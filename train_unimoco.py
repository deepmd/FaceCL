import argparse
import logging
import math
import os

import torch
from torch import distributed
from torch.utils.tensorboard import SummaryWriter

from backbones import get_model
from data import get_dataloader, get_transform

from unimoco import UniMoCo, UnifiedContrastive
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging


try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main(args):
    torch.cuda.set_device(args.local_rank)
    cfg = get_config(args.config)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)
    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    # set deterministic training for reproducibility
    if cfg.seed is not None:
        import random
        import numpy
        random.seed(cfg.seed)
        numpy.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        # setting the following flags degrade performance considerably!
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # define train_loader
    train_transform = get_transform()
    train_loader = get_dataloader(
        cfg.rec, local_rank=args.local_rank, batch_size=cfg.batch_size, label_group_size=cfg.samples_per_label,
        dali=cfg.dali, transform=train_transform)

    # define model
    model = UniMoCo(
        base_encoder=lambda: get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size),
        dim=cfg.moco_dim, K=cfg.moco_k, m=cfg.moco_m, T=cfg.moco_t, mlp=cfg.moco_mlp)
    model = torch.nn.parallel.DistributedDataParallel(
        module=model.cuda(), broadcast_buffers=False, device_ids=[args.local_rank])
    model.train()

    # define loss function (criterion) and optimizer
    criterion = UnifiedContrastive().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay)

    cfg.total_step = len(train_loader) * cfg.epochs

    logging.info("world_size: %d" % world_size)
    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=summary_writer
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    start_epoch = 0
    global_step = 0
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.epochs):
        lr = adjust_learning_rate(optimizer, epoch, cfg)

        train_loader.batch_sampler.set_epoch(epoch)
        # train for one epoch
        for _, (images_q, images_k, labels) in enumerate(train_loader):
            global_step += 1

            output, target = model(im_q=images_q, im_k=images_k, labels=labels)
            loss = criterion(output, target)

            if cfg.fp16:
                amp.scale(loss).backward()
                amp.step(optimizer)
                amp.update()
            else:
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr, amp)

                if global_step % cfg.verbose == 0 and global_step > 200:
                    callback_verification(global_step, model)

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(model.module.state_dict(), path_module)

        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(model.module.state_dict(), path_module)
    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())
