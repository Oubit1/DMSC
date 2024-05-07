# Data Augmentation Strategies for Semi-Supervised Medical Image Segmentation
by Jiahui Wang, Mingfeng Jiang*, Dongsheng Ruan, Yang Li, Zefeng Wang, Yongquan Wu, Tao Tan, Guang Yang, Senior Member, IEEE, Ling Xia
## Introduction
This repository is for our paper '[Dynamic Mask Splicing-guided Mixing Consistency for Semi-supervised 3D Medical Image Segmentation]
## Requirements
This repository is based on Pytorch 1.9.1, CUDA11.1 and Python 3.6.5
## Usage

### Dataset
We use [the dataset of 2018 Atrial Segmentation Challenge](http://atriaseg2018.cardiacatlas.org/).
We use [the dataset of Pancreas-CT](https://drive.google.com/file/d/1qzFUtkHx-46kFvHE7RAMhjAdo6dmn4iT/view?usp=sharing/).
### Preprocess
If you want to process .nrrd data into .h5 data, you can use `code/dataloaders/preprocess.py`.


## Acknowledgements
Our code is origin from [UAMT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [DTC](https://github.com/HiLab-git/DTC), and [MC-Net+](https://github.com/ycwu1997/MC-Net),and [CC-Net+](https://github.com/Cuthbert-Huang/CC-Net) Thanks to these authors for their excellent work.
