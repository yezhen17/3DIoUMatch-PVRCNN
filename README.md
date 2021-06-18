# 3DIoUMatch-PVRCNN

## Notice

In the current version of our paper, the experiments are using a problematic setting -- we used the complete `gt_database` for `db_sample` data augmentation, causing the baseline performance of 1% and 2% data to be too high. Now we limited the `gt_database` to be also 1%, for example, if the training data is only 1%. We also used train-time RoI selection stradegy in pseudo-label generation before, which we now changed to test-time RoI selection stradegy to avoid involving ground truth knowledge. Corrected experiments still show our method can achieve big improvements. The paper on arXiv will be updated very soon.

## Codebase

This codebase is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)(commit a7cf5368d9cbc3969b4613c9e61ba4dcaf217517), please refer to the origin [README.md](./README_OpenPCDet.md) for installation and usage of OpenPCDet.
