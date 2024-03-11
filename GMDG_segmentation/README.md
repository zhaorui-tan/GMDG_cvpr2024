# Applying General mDG objective to segmentation task. 

This code is modified form RobustNet (CVPR 2021 Oral) https://arxiv.org/abs/2103.15597. 
code link: https://github.com/shachoi/RobustNet

To run this code, please follow RobustNet to prepare the environment and datasets.

You can train the model with following commands.
```angular2html
<!--RobustNet + T1 -->
$ CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_r50os16_gtav_isw_DG_t1.sh # Train: GTAV, Cityscapes, Test: BDD100K, Synthia, Mapillary / ResNet50
<!--RobustNet + T2-->
$ CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_r50os16_gtav_isw_DG4_t2.sh # Train: GTAV, Cityscapes, Test: BDD100K, Synthia, Mapillary / ResNet50
<!--RobustNet + T1 + T2-->
$ CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_r50os16_gtav_isw_DG3_t1_t2.sh # Train: GTAV, Cityscapes, Test: BDD100K, Synthia, Mapillary / ResNet50
```

Inference and validation
```angular2html
<!--Inference -->
$ CUDA_VISIBLE_DEVICES=0,1 ./scripts/infer_r50os16_cty_isw_DG.sh.sh 
<!--Validation-->
$ CUDA_VISIBLE_DEVICES=0,1 ./scripts/valid_r50os16_gtav_isw_DG.sh
```


## Preparation 
### Installation
Clone this repository.
```
git clone https://github.com/shachoi/RobustNet.git
cd RobustNet
```
Install following packages.
```
conda create --name robustnet python=3.7
conda activate robustnet
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install scipy==1.1.0
conda install tqdm==4.46.0
conda install scikit-image==0.16.2
pip install tensorboardX
pip install thop
pip install kmeans1d
imageio_download_bin freeimage
```
### How to Run RobustNet
We evaludated RobustNet on [Cityscapes](https://www.cityscapes-dataset.com/), [BDD-100K](https://bair.berkeley.edu/blog/2018/05/30/bdd/),[Synthia](https://synthia-dataset.net/downloads/) ([SYNTHIA-RAND-CITYSCAPES](http://synthia-dataset.net/download/808/)), [GTAV](https://download.visinf.tu-darmstadt.de/data/from_games/) and [Mapillary Vistas](https://www.mapillary.com/dataset/vistas?pKey=2ix3yvnjy9fwqdzwum3t9g&lat=20&lng=0&z=1.5).

We adopt Class uniform sampling proposed in [this paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Improving_Semantic_Segmentation_via_Video_Propagation_and_Label_Relaxation_CVPR_2019_paper.pdf) to handle class imbalance problems. [GTAVUniform](https://github.com/shachoi/RobustNet/blob/0538c69954c030273b3df952f90347572ecac53b/datasets/gtav.py#L306) and [CityscapesUniform](https://github.com/shachoi/RobustNet/blob/0538c69954c030273b3df952f90347572ecac53b/datasets/cityscapes.py#L324) are the datasets to which Class Uniform Sampling is applied.


1. For Cityscapes dataset, download "leftImg8bit_trainvaltest.zip" and "gtFine_trainvaltest.zip" from https://www.cityscapes-dataset.com/downloads/<br>
Unzip the files and make the directory structures as follows.
```
cityscapes
 └ leftImg8bit_trainvaltest
   └ leftImg8bit
     └ train
     └ val
     └ test
 └ gtFine_trainvaltest
   └ gtFine
     └ train
     └ val
     └ test
```
```
bdd-100k
 └ images
   └ train
   └ val
   └ test
 └ labels
   └ train
   └ val
```
```
mapillary
 └ training
   └ images
   └ labels
 └ validation
   └ images
   └ labels
 └ test
   └ images
   └ labels
```

#### We used [GTAV_Split](https://download.visinf.tu-darmstadt.de/data/from_games/code/read_mapping.zip) to split GTAV dataset into training/validation/test set. Please refer the txt files in [split_data](https://github.com/shachoi/RobustNet/tree/main/split_data).

```
GTAV
 └ images
   └ train
     └ folder
   └ valid
     └ folder
   └ test   
     └ folder
 └ labels
   └ train
     └ folder
   └ valid
     └ folder
   └ test   
     └ folder
```

#### We randomly splitted [Synthia dataset](http://synthia-dataset.net/download/808/) into train/val set. Please refer the txt files in [split_data](https://github.com/shachoi/RobustNet/tree/main/split_data).

```
synthia
 └ RGB
   └ train
   └ val
 └ GT
   └ COLOR
     └ train
     └ val
   └ LABELS
     └ train
     └ val
```

2. You should modify the path in **"<path_to_robustnet>/config.py"** according to your dataset path.
```
#Cityscapes Dir Location
__C.DATASET.CITYSCAPES_DIR = <YOUR_CITYSCAPES_PATH>
#Mapillary Dataset Dir Location
__C.DATASET.MAPILLARY_DIR = <YOUR_MAPILLARY_PATH>
#GTAV Dataset Dir Location
__C.DATASET.GTAV_DIR = <YOUR_GTAV_PATH>
#BDD-100K Dataset Dir Location
__C.DATASET.BDD_DIR = <YOUR_BDD_PATH>
#Synthia Dataset Dir Location
__C.DATASET.SYNTHIA_DIR = <YOUR_SYNTHIA_PATH>
```


