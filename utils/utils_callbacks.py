import logging
import math
import os
import time
from typing import List

import torch

from eval import verification
from utils.utils_logging import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from torch import distributed


class CallBackVerification(object):
    
    def __init__(self, val_targets, rec_prefix, summary_writer=None, image_size=(112, 112)):
        self.rank: int = distributed.get_rank()
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        if self.rank is 0:
            self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

        self.summary_writer = summary_writer

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                self.ver_list[i], backbone, 10, 10)
            logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))

            self.summary_writer: SummaryWriter
            self.summary_writer.add_scalar(tag=self.ver_name_list[i], scalar_value=acc2, global_step=global_step, )

            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info(
                '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, num_update, backbone: torch.nn.Module):
        if self.rank is 0 and num_update > 0:
            backbone.eval()
            self.ver_test(backbone, num_update)
            backbone.train()


class CallBackLogging(object):
    def __init__(self, frequent, total_step, batch_size, num_labels, label_queue, writer=None):
        self.frequent: int = frequent
        self.rank: int = distributed.get_rank()
        self.world_size: int = distributed.get_world_size()
        self.time_start = time.time()
        self.total_step: int = total_step
        self.batch_size: int = batch_size
        self.writer = writer
        self.label_queue = label_queue
        self.num_labels = num_labels

        self.init = False
        self.tic = 0

    @torch.no_grad()
    def _get_queue_stats(self):
        valid_items = self.label_queue[self.label_queue != -1]
        num_items = len(valid_items)
        labels_counts = torch.bincount(valid_items, minlength=self.num_labels)
        assert len(labels_counts) == self.num_labels  # assumed that label ids are from 0 to (num_labels-1)
        std, mean = torch.std_mean(labels_counts.float())
        mini = torch.min(labels_counts)
        maxi = torch.max(labels_counts)
        zeros_count = self.num_labels - torch.count_nonzero(labels_counts)
        probs = labels_counts / num_items  # label probabilities
        cross_entropy = -torch.sum((1/self.num_labels) * torch.log(probs + 1e-20))  # cross entropy of label probabilities relative to balanced label probabilities (1/c)
        balanced_entropy = math.log(self.num_labels)  # entropy of balanced label probabilities (1/c)
        balance_score = balanced_entropy / cross_entropy  # in range of (0, 1]
        return num_items, mean.item(), std.item(), mini.item(), maxi.item(), zeros_count.item(), balance_score.item()

    def __call__(self,
                 global_step: int,
                 loss: AverageMeter,
                 epoch: int,
                 fp16: bool,
                 learning_rate: float,
                 grad_scaler: torch.cuda.amp.GradScaler,
                 positives: AverageMeter):
        if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float('inf')

                time_now = (time.time() - self.time_start) / 3600
                time_total = time_now / ((global_step + 1) / self.total_step)
                time_for_end = time_total - time_now
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    self.writer.add_scalar('learning_rate', learning_rate, global_step)
                    self.writer.add_scalar('loss', loss.avg, global_step)
                if fp16:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.4f   Epoch: %d   Global Step: %d   " \
                          "Fp16 Grad Scale: %2.f   Required: %1.f hours" % (
                              speed_total, loss.avg, learning_rate, epoch, global_step,
                              grad_scaler.get_scale(), time_for_end
                          )
                else:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.4f   Epoch: %d   Global Step: %d   " \
                          "Required: %1.f hours" % (
                              speed_total, loss.avg, learning_rate, epoch, global_step, time_for_end
                          )

                # log queue stats
                num_items, mean, std, mini, maxi, zeros_count, balance_score = self._get_queue_stats()
                msg += "  |  QueueStats: " \
                       "#Items %d   #Zeros %d   Mean %.2f   Std %.2f   Min %d   Max %d   Balance %.3f   AvgPos %.2f" % (
                          num_items, zeros_count, mean, std, mini, maxi, balance_score, positives.avg
                       )
                if self.writer is not None:
                    self.writer.add_scalar('queue/#items', num_items, global_step)
                    self.writer.add_scalar('queue/avg_per_label', mean, global_step)
                    self.writer.add_scalar('queue/balance_score', balance_score, global_step)
                    self.writer.add_scalar('queue/avg_positives', positives.avg, global_step)

                logging.info(msg)
                loss.reset()
                positives.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()
