# Applying General mDG objective to classification task. 

This code is modified form MIRO: Mutual Information Regularization with Oracle (ECCV'22) [Domain Generalization by Mutual-Information Regularization with Pre-trained Models](https://arxiv.org/abs/2203.10789).
 
## Reproduce the main results of the paper

We provide the commands to reproduce the classification results of the paper.
Note that every result is averaged over three trials.


```angular2html
<!--Use ResNet-50 with SWAD:-->
--dataset TerraIncognita --ld 0.1 --shift 0.1 --d_shift 0.2 --lr_mult 12.5
--dataset OfficeHome --ld 0.1 --lr 3e-5 --resnet_dropout 0.1 --weight_decay 1e-6 --shift 0.001 --d_shift 0.1 --lr_mult 20.0
--dataset VLCS --checkpoint_freq 50 --tolerance_ratio 0.2 --lr 1e-5 --resnet_dropout 0.5 --weight_decay 1e-6 --ld 0.01 --shift 0.001 --d_shift 0.1 --lr_mult 10 
--dataset PACS --ld 0.01 --shift 0.01 --d_shift 0.01 --lr_mult 25.
--dataset DomainNet --ld 0.1 --checkpoint_freq 500 --shift 0.1 --d_shift 0.1 --lr_mult 7.5

<!--Use ResNet-50 with SWAD:-->
--dataset TerraIncognita --ld 0.1 --swad True --shift 0.001 --d_shift 0.01 --lr_mult 10
--dataset OfficeHome --swad True --ld 0.1 --shift 0.1 --d_shift 0.3 --lr_mult 10.
--dataset VLCS --checkpoint_freq 50 --tolerance_ratio 0.2 --ld 0.01 --swad True --swad True --shift 0.001 --d_shift 0.1 --lr_mult 10.
--dataset PACS --ld 0.01 --swad True --shift 0.001 --d_shift 0.1 --lr_mult 20.
--dataset DomainNet --ld 0.1 --checkpoint_freq 500 --swad True --shift 0.1 --d_shift 0.1 --lr_mult 7.5

<!--Use RegNetY-16GF with and without SWAD-->
--dataset TerraIncognita --ld 0.01 --model swag_regnety_16gf --batch_size 16 --checkpoint_freq 200 --swad True --shift 0.01 --d_shift 0.01 --lr_mult 2.5
--dataset OfficeHome --ld 0.01 --model swag_regnety_16gf --batch_size 16 --swad True --shift 0.1 --d_shift 0.1 --lr_mult 0.1
--dataset VLCS --ld 0.01 --checkpoint_freq 50 --tolerance_ratio 0.2 --model swag_regnety_16gf --batch_size 16 --swad True --shift 0.01 --d_shift 0.1 --lr_mult 2.
--dataset PACS --ld 0.01 --model swag_regnety_16gf --batch_size 16 --swad True --shift 0.1 --d_shift 0.1 --lr_mult 0.1
--dataset DomainNet --ld 0.1 --checkpoint_freq 500 --model swag_regnety_16gf --batch_size 14 --swad True --shift 0.1 --d_shift 0.1
```

## Preparation

### Dependencies

```sh
pip install -r requirements.txt
```

### Datasets

```sh
python -m domainbed.scripts.download --data_dir=/my/datasets/path
```

### Environments

Environment details used for the main experiments. Every main experiment is conducted on a single NVIDIA V100 GPU.

```
Environment:
	Python: 3.7.7
	PyTorch: 1.7.1
	Torchvision: 0.8.2
	CUDA: 10.1
	CUDNN: 7603
	NumPy: 1.21.4
	PIL: 7.2.0
```

## How to Run

`train_all.py` script conducts multiple leave-one-out cross-validations for all target domain.

```sh
python train_all.py exp_name --dataset PACS --data_dir /my/dataset/path --algorithm GMDG
```


