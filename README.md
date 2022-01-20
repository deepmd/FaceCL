# FaceCL: Contrastive Representation Learning for Face Recognition
 
This is an implementation of contrastive representation learning for face recognition.

## Requirements

- Install [PyTorch](http://pytorch.org) (torch>=1.6.0).
- (Optional) Install [DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/).
- `pip install -r requirements.txt`.

## How to Training

To train a model, run `train_unimoco.py` with the path to the configs.  
The example commands below show how to run distributed training.

### 1. To run on a machine with 8 GPUs:

```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 train_unimoco.py configs/ms1mv3_r50_unimoco
```

### 2. To run on 2 machines with 8 GPUs each:

Node 0:

```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="ip1" --master_port=12581 train_unimoco.py configs/ms1mv3_r50_unimoco
```

Node 1:

```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="ip1" --master_port=12581 train_unimoco.py configs/ms1mv3_r50_unimoco
```

## Download Datasets or Prepare Datasets

- [MS1MV3](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-retinaface) (93k IDs, 5.2M images)
- [Glint360K](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc#4-download) (360k IDs, 17.1M images)
- [WebFace42M](docs/prepare_webface42m.md) (2M IDs, 42.5M images)

## Performance on IJB-C and [**ICCV2021-MFR**](https://github.com/deepinsight/insightface/blob/master/challenges/mfr/README.md)

ICCV2021-MFR testset consists of non-celebrities so we can ensure that it has very few overlap with public available face 
recognition training set, such as MS1M and CASIA as they mostly collected from online celebrities. 
As the result, we can evaluate the FAIR performance for different algorithms.  

For **ICCV2021-MFR-ALL** set, TAR is measured on all-to-all 1:1 protocal, with FAR less than 0.000001(e-6). The 
globalised multi-racial testset contains 242,143 identities and 1,624,305 images. 



| Datasets                 | Backbone   | **MFR-ALL** | IJB-C(1E-4) | IJB-C(1E-5) | Training Throughout | log                                                                                                                                                                                                         |
|:-------------------------|:-----------|:-------------|:------------|:------------|:--------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MS1MV3                   | r50        | -            | -           | -           | -                   | log\|config                                                                                                                                                                                                 |
