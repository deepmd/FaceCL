import os
import numbers
import sys

import numpy as np
import mxnet as mx

root_dir = sys.argv[1]
path_imgrec = os.path.join(root_dir, 'train.rec')
path_imgidx = os.path.join(root_dir, 'train.idx')
imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
s = imgrec.read_idx(0)
header, _ = mx.recordio.unpack(s)
if header.flag > 0:
    header0 = (int(header.label[0]), int(header.label[1]))
    imgidx = np.array(range(1, int(header.label[0])))
    print(int(header.label[0]))

labels_path = os.path.join(root_dir, 'labels.txt')
if os.path.exists(labels_path):
    print(f"labels file {labels_path} already exists!")
    exit()

with open(labels_path, 'a') as f:
    for idx in imgidx:
        s = imgrec.read_idx(idx)
        header, _ = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = label.astype(int)
        f.writelines(f"{idx}\t{label}\n")
        print(idx)
