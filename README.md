# 3DIoUMatch-PVRCNN

## Introduction

In this repository, we provide 3DIoUMatch implementation (with Pytorch) based on PV-RCNN. This is not a general implementation of SSL-training on KITTI but tailored to PVRCNN.

The implementation based on VoteNet is [here](https://github.com/THU17cyz/3DIoUMatch). Our arXiv report is [here](https://arxiv.org/abs/2012.04355v3).

## Notice

In the current version of our paper, the experiments are using a problematic setting -- we used the complete `gt_database` for `gt_sampling` data augmentation, causing the baseline performance of 1% and 2% data to be too high (which means we used 100% bounding boxes and the points enclosed in them for data augmentation, for more information please refer to [database_sample.py](https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/datasets/augmentor/database_sampler.py)). Now we limit the `gt_database` to be also 1%, for example, if the training data is only 1%. We also used train-time RoI selection stradegy in pseudo-label generation before, which we now change to test-time RoI selection strategy to avoid involving ground truth knowledge. Corrected experiments show our method can still achieve large improvements. Part of the experiment results are shown below. The paper on arXiv has already been updated. Sincere apologies for this problem and thank [Andy Yuan](https://github.com/AndyYuan96) very much for helping with this issue.

![image](https://user-images.githubusercontent.com/52420115/122535862-8873fe00-d056-11eb-9ad3-bd41d76f6af9.png) 

The above results are achieved with IoU thresholds 0.5, 0.25, 0.25 for car, pedestrian, and cyclist, respectively. The classification threshold is 0.4. We repeat the traverse of labeled data for 5 times in each epoch and we train 60 epochs.


## Installation

Please refer to the origin [README.md](./README_OpenPCDet.md) for installation and usage of OpenPCDet.

## Data Preparation and Training

#### Data preparation

Please first generate the data splits or use the data splits we provide.

```bash
cd data/kitti/ImageSets
python split.py <label_ratio> <split_num>
cd ../../..
```

For example:

```bash
cd data/kitti/ImageSets
python split.py 0.01 4
cd ../../..
```

Then generate the `infos` and `dbinfos`, and rename `kitti_dbinfos_train_3712.pkl`.

```
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos \
tools/cfgs/dataset_configs/kitti_dataset.yaml
mv data/kitti/kitti_dbinfos_train_3712.pkl data/kitti/kitti_dbinfos_train.pkl
```

Then generate the new `gt_database` based on the data split.

```bash
python -m pcdet.datasets.kitti.kitti_dataset create_part_dbinfos \
tools/cfgs/dataset_configs/kitti_dataset.yaml <split_name_Without_txt>
```

For example:

```bash
python -m pcdet.datasets.kitti.kitti_dataset create_part_dbinfos \
tools/cfgs/dataset_configs/kitti_dataset.yaml train_0.01_1
```

#### Pre-training

```bash
GPUS_PER_NODE=<num_gpus> sh scripts/slurm_pretrain.sh <partition> \
<job_name> <num_gpus> --cfg_file ./cfgs/kitti_models/pv_rcnn.yaml \
--split <split_name_without_txt> --extra_tag <log_folder_name> \
--ckpt_save_interval <ckpt_save_interval> \
--repeat <number_of_traverses_of_dataset_in_one_epoch> \
--dbinfos <pkl_name_of_dbinfos>
```

For example:

```bash
GPUS_PER_NODE=8 sh scripts/slurm_pretrain.sh p1 pretrain_0.01_1 8 \
--cfg_file ./cfgs/kitti_models/pv_rcnn.yaml --split train_0.01_1 \
--extra_tag split_0.01_1 --ckpt_save_interval 4 --repeat 10 \
--dbinfos kitti_dbinfos_train_0.01_1_37.pkl
```

#### Training

```bash
GPUS_PER_NODE=<num_gpus> sh scripts/slurm_train.sh <partition> \
<job_name> <num_gpus> --cfg_file ./cfgs/kitti_models/pv_rcnn_ssl_60.yaml \
--split <split_name_without_txt> --extra_tag <log_folder_name> \
--ckpt_save_interval <ckpt_save_interval> --pretrain_model <path_to_pretrain_model> \
--repeat <number_of_traverses_of_dataset_in_one_epoch> --thresh <iou_thresh> \
--sem_thresh <sem_cls_thresh> --dbinfos <pkl_name_of_dbinfos>
```

For example:

```bash
GPUS_PER_NODE=8 sh scripts/slurm_train.sh p1 train_0.01_1 8 \
--cfg_file ./cfgs/kitti_models/pv_rcnn_ssl_60.yaml --split train_0.01_1 \
--extra_tag split_0.01_1 --ckpt_save_interval 2 \
--pretrained_model "../output/cfgs/kitti_models/pv_rcnn/split_0.01_1/ckpt/checkpoint_epoch_80.pth" \
--repeat 5 --thresh '0.5,0.25,0.25' --sem_thresh '0.4,0.0,0.0' \
--dbinfos kitti_dbinfos_train_0.01_1_37.pkl
```

Note: Currently only the first element of `sem_thresh` is used (class-agnostic). And the batch size per GPU card is currently hardcoded to be 1+1 (labeled+unlabeled).

## Acknowledgement

This codebase is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) (commit a7cf5368d9cbc3969b4613c9e61ba4dcaf217517).

