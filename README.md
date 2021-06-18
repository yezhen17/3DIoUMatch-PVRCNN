# 3DIoUMatch-PVRCNN

## Notice

In the current version of our paper, the experiments are using a problematic setting -- we used the complete `gt_database` for `gt_sampling` data augmentation, causing the baseline performance of 1% and 2% data to be too high (which means we used 100% bounding boxes and the points enclosed in them for data augmentation, for more information please refer to [database_sample.py](https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/datasets/augmentor/database_sampler.py)). Now we limited the `gt_database` to be also 1%, for example, if the training data is only 1%. We also used train-time RoI selection stradegy in pseudo-label generation before, which we now changed to test-time RoI selection stradegy to avoid involving ground truth knowledge. Corrected experiments still show our method can achieve big improvements. Part of the experiment results are shown below. The paper on arXiv and the complete experiment results will be updated very soon. Sincere apologies for this problem and thank [Andy Yuan](https://github.com/AndyYuan96) very much for helping with this issue.

![image](https://user-images.githubusercontent.com/52420115/122535862-8873fe00-d056-11eb-9ad3-bd41d76f6af9.png) 

More instructions on how to use the repo will be updated very soon.

## Codebase

This codebase is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) (commit a7cf5368d9cbc3969b4613c9e61ba4dcaf217517), please refer to the origin [README.md](./README_OpenPCDet.md) for installation and usage of OpenPCDet.
