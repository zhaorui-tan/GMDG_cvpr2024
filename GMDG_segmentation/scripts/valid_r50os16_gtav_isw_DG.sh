#!/usr/bin/env bash
#echo "Running inference on" ${1}
     python -m torch.distributed.launch --nproc_per_node=1 valid.py \
        --val_dataset bdd100k cityscapes  mapillary  synthia gtav\
        --arch network.deepv3.DeepR50V3PlusD \
        --wt_layer 0 0 2 2 2 0 0 \
        --date 0101 \
        --exp r50os16_gtav_isw \
        --snapshot /path/to/project/segmentation/RobustNet-main/logs/0101/r50os16_gtav_isw/09_19_17_t1_t2_final/best_cityscapes_epoch_9_mean-iu_0.38618.pth
