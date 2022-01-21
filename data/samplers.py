from torch.utils.data import Sampler
import os
from collections import defaultdict
import random


class SingleClassGroupSampler(Sampler):
    def __init__(self, file_path, batch_size, group_size, drop_last=False):
        if batch_size % group_size != 0:
            raise ValueError(f"Batch size {batch_size} should be divisible by group size {group_size}")
        self.batch_size = batch_size
        self.group_size = group_size
        self.drop_last = drop_last

        if not os.path.exists(file_path):
            raise ValueError(f'{file_path} does not exist.')
        with open(file_path, 'r') as f:
            samples = f.readlines()

        samples_labels = [int(sample.strip('\n').split('\t')[1]) for sample in samples]

        label2id = defaultdict(lambda: [])
        for index, label in enumerate(samples_labels):
            label2id[label].append(index)

        total_samples = 0
        for label, indices in label2id.items():
            if len(indices) % self.group_size != 0:
                required_items = self.group_size - len(indices) % self.group_size
                if len(indices) >= required_items:
                    items = random.sample(indices, required_items)
                else:
                    items = random.choices(indices, k=required_items)
                indices.extend(items)
            total_samples += len(indices)

        self.label2id = label2id
        self.total_samples = total_samples

    def create_index_list(self):
        index_list = []
        for ids in self.label2id.values():
            random.shuffle(ids)
            index_list.extend([ids[i:i+self.group_size] for i in range(0, len(ids), self.group_size)])

        random.shuffle(index_list)
        index_list = [item for sublist in index_list for item in sublist]

        return index_list

    def __iter__(self):
        index_list = self.create_index_list()
        for i in range(0, len(index_list), self.batch_size):
            batch = index_list[i:i+self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self):
        if self.drop_last:
            return self.total_samples // self.batch_size
        else:
            return (self.total_samples + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":
    batch_size = 9
    group_size = 3
    group_sampler = SingleClassGroupSampler('labels.txt', batch_size, group_size, False)
    print(len(group_sampler))

    for idx, x in enumerate(group_sampler):
        print(f"{idx} x: {x} --> {len(x)}")
