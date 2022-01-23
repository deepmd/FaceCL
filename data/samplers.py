import math
import os
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class DistributedClassGroupSampler(Sampler):
    r"""Batch sampler that group items of each label into batches and also restricts data loading
    to a subset of the dataset.

    Arguments:
        labels_path (str): Path to the file containing label information of all items.
        batch_size (int): Size of mini-batch. It should be divisible by group_size.
        group_size (int): Number of items of each label in the mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        seed (int, optional): random seed used to shuffle the sampler. This number
            should be identical across all processes in the distributed group. Default: ``0``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.
    """

    def __init__(self, labels_path, batch_size, group_size, drop_last=False,
                 num_replicas=None, rank=None, seed=0):
        if batch_size % group_size != 0:
            raise ValueError(f"Batch size {batch_size} should be divisible by group size {group_size}")
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.batch_size = batch_size
        self.group_size = group_size
        self.drop_last = drop_last
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed

        if not os.path.exists(labels_path):
            raise ValueError(f'{labels_path} does not exist.')
        with open(labels_path, 'r') as f:
            samples = f.readlines()
        labels = [int(sample.strip('\n').split('\t')[1]) for sample in samples]

        self.label2id = defaultdict(lambda: [])
        for index, label in enumerate(labels):
            self.label2id[label].append(index)

        total_groups = 0
        for label, ids in self.label2id.items():
            self.label2id[label] = torch.as_tensor(ids)
            total_groups += math.ceil(len(ids) / self.group_size)
        per_replica_groups = math.ceil(total_groups / self.num_replicas)
        self.total_groups = per_replica_groups * self.num_replicas
        self.per_replica_samples = per_replica_groups * self.group_size

    def create_index_list(self):
        # deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        # shuffle both indices of each label and order of grouped indices.
        # add extra samples if required to make number of indices of each label divisible by self.group_size
        # and number of groups divisible by self.num_replicas (to ensure each rank receives the same amount of data).
        grouped_ids = []
        for ids in self.label2id.values():
            random_indices = torch.randperm(len(ids), generator=g)  # shuffle indices of each label
            extra_needed = -len(ids) % self.group_size
            if extra_needed > 0:  # randomly pad indices until divisible by self.group_size
                extra_ids = torch.randint(high=len(ids), size=(extra_needed,), generator=g)
                random_indices = torch.cat((random_indices, extra_ids))
            ids = torch.reshape(ids[random_indices], shape=(-1, self.group_size))  # group shuffled and padded indices into rows
            grouped_ids.append(ids)
        grouped_ids = torch.vstack(grouped_ids)
        random_group_indices = torch.randperm(grouped_ids.shape[0], generator=g)  # shuffle rows (groups)
        extra_needed = -grouped_ids.shape[0] % self.num_replicas
        if extra_needed > 0:  # randomly pad groups until divisible by self.num_replicas
            extra_group_indices = torch.randint(high=grouped_ids.shape[0], size=(extra_needed,), generator=g)
            random_group_indices = torch.cat((random_group_indices, extra_group_indices))
        grouped_ids = grouped_ids[random_group_indices]
        assert grouped_ids.shape[0] == self.total_groups

        # subsample groups for current rank
        grouped_ids = grouped_ids[self.rank:self.total_groups:self.num_replicas].flatten()
        assert len(grouped_ids) == self.per_replica_samples

        return grouped_ids

    def __iter__(self):
        index_list = self.create_index_list()
        for i in range(0, len(index_list), self.batch_size):
            batch = index_list[i:i+self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self):
        if self.drop_last:
            return self.per_replica_samples // self.batch_size
        else:
            return (self.per_replica_samples + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. This ensures all replicas use a different random ordering
        for each epoch. Otherwise, the next iteration of this sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


if __name__ == "__main__":
    batch_size = 256
    group_size = 4
    num_replicas = 3
    group_samplers = []
    for r in range(num_replicas):
        group_samplers.append(DistributedClassGroupSampler('synth_labels.txt', batch_size, group_size, False,
                                                           num_replicas=num_replicas, rank=r))
    print(tuple([len(gs) for gs in group_samplers]))

    # rank_batches = [[] for _ in range(num_replicas)]
    for batch_idx, xs in enumerate(zip(*group_samplers)):
        print(f"Batch {batch_idx}:")
        for r, x in enumerate(xs):
            print(f"  rank_{r}: {x.tolist()} --> {len(x)}")
            # rank_batches[r].append(x)

    # import itertools
    # import numpy
    # for r in range(num_replicas):
    #     rank_batches[r] = torch.hstack(rank_batches[r]).numpy()
    # for r1, r2 in itertools.combinations(range(num_replicas), 2):
    #     intersection = numpy.intersect1d(rank_batches[r1], rank_batches[r2]).tolist()
    #     print(f"Intersection of batches between {r1}({len(rank_batches[r1])}) and {r2}({len(rank_batches[r2])}):"
    #           f"{intersection} --> {len(intersection)}")
