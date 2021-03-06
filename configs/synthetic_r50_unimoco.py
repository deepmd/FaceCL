from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.seed = None
config.network = "r50"
config.output = None
config.embedding_size = 512
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 8  # should be divisible by samples_per_label | total_batch_size = batch_size * num_gpus
config.samples_per_label = 2
config.lr = 0.2
config.epochs = 25
config.scheduler = "cos"  # cos: CosineAnnealingLR, poly: PolyScheduler
config.verbose = 2000
config.frequent = 10
config.dali = False
config.loss_margin = 0
config.augmentation = "MoCo2_NoCrop"  # MoCo2, MoCo2_NoCrop, ArcFace

config.moco_dim = 128
config.moco_k = 65536  # should be divisible by total_batch_size
config.moco_m = 0.999
config.moco_t = 0.07
config.moco_mlp = None  # same_in: same as embedding_size, same_out: same as moco_dim, None: a single linear layer

config.rec = "synthetic"
config.num_image = 100000
config.num_classes = 1000
config.val_targets = []
