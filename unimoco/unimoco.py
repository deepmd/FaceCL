"""
unimoco model definition.
"""
from itertools import chain

import torch
from torch import nn

class UniMoCo(nn.Module):
    """
    build a UniMoCo model with the same hyper-parameter with MoCo
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(UniMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.encoder_q, dim_mlp = base_encoder()
        self.encoder_k, _ = base_encoder()
        self.embedding_size = dim_mlp

        if not mlp:
            self.projection_q = nn.Linear(dim_mlp, dim)
            self.projection_k = nn.Linear(dim_mlp, dim)
        else:
            self.projection_q = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
            self.projection_k = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))

        for param_q, param_k in zip(chain(self.encoder_q.parameters(), self.projection_q.parameters()),
                                    chain(self.encoder_k.parameters(), self.projection_k.parameters())):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # initialize a label queue with shape K.
        # all the label is -1 by default.
        self.register_buffer("label_queue", torch.zeros(self.K).long() - 1)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(chain(self.encoder_q.parameters(), self.projection_q.parameters()),
                                    chain(self.encoder_k.parameters(), self.projection_k.parameters())):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys and labels before updating queue
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)

        batch_size = keys.shape[0]
        # print(batch_size)
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity
 
        # replace the keys and labels at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T # this queue is feature queue
        self.label_queue[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    
    def forward(self, im_q, im_k=None, labels=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            labels: a batch of label for images (-1 for unsupervised images)
        Output:
            logits: with shape NxK
            targets: with shape NxK
        """
        # if only im_q is provided, just return output of encoder_q
        if im_k is None and labels is None:
            return self.encoder_q(im_q)

        # compute query features
        q = self.projection_q(self.encoder_q(im_q))  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.projection_k(self.encoder_k(im_k))  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, labels)

        # compute logits
        # Einstein sum is more intuitive
        # logits: NxK
        logits = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # apply temperature
        logits /= self.T

        # find same label images from label queue
        # for the query with -1, all 
        targets = ((labels[:, None] == self.label_queue[None, :]) & (labels[:, None] != -1)).float().cuda()

        return logits, targets

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output