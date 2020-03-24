This repository contains the supplementary Python code for paper **"Background Data Resampling for Outlier-Aware Classification"** to appear at CVPR 2020. 

## Prerequisites

### Environment
The code was tested with Python 3.7+ using PyTorch v1.2+. Additional Python libraries may be required, as specified in `requirements.txt`.

### Datasets
The pre-training and evaluation code makes use of the following datasets:

Dataset | URL
-- | --
CIFAR-10/100 | `https://www.cs.toronto.edu/~kriz/cifar.html`
Tiny ImageNet | `https://tiny-imagenet.herokuapp.com/`
Textures | `https://www.robots.ox.ac.uk/~vgg/data/dtd/`
LSUN | `https://www.yf.io/p/lsun`
SVHN | `http://ufldl.stanford.edu/housenumbers/`
Places | `http://places.csail.mit.edu/`

Training with background data further requires ILSVRC'12 and/or Tiny Images:

Dataset | URL
-- | --
ILSVRC 2012 | `http://www.image-net.org/challenges/LSVRC/2012/`
80M Tiny Images | `https://groups.csail.mit.edu/vision/TinyImages/`


All datasets must be downloaded and prepared under paths specified in `datasets/__init__.py`. ILSVRC'12 data should be further processed into LMDB format for faster data loading.

## Instructions

### Pre-training
Use `train.py` to pretrain models on the in-distribution datasets. Models are automatically saved under `checkpoints/` folder. Example:
```
python train.py --gpus 0 -a wrn40 -d cifar10 --epochs 100
```

### OOD detection
Use `test_ood.py` to evaluate OOD detection of trained models on one or more test sets. Example:
```
python test_ood.py --gpus 0 -a wrn40 -id cifar10 -od gaussian uniform textures lsun svhn places --out-ratio 0.2 --load-path path/to/model.pth
```

### Fine-tuning with background data
Use `train_bg.py` to finetune models using background data. Supports full background dataset, uniformly subsampled background, or resampled background using learned weights. Example:
```
python train_bg.py --gpus 0 -a wrn40 -id cifar10 -od ilsvrc --epochs 50 --load-path path/to/model.pth
python train_bg.py --gpus 0 -a wrn40 -id cifar10 -od ilsvrc --epochs 50 --load-path path/to/model.pth --resample random -p 0.1
python train_bg.py --gpus 0 -a wrn40 -id cifar10 -od ilsvrc --epochs 50 --load-path path/to/model.pth --resample path/to/resample/weights.pth -p 0.1
```

### Adversarial background resampling
Use `train_bg_resample.py` to perform adversarial resampling on background dataset. Resampling weights are automatically saved under `checkpoints/` folder. Example:
```
python train_bg_resample.py --gpus 0 -a wrn40 -id cifar10 -od ilsvrc --epochs 50 --load-path path/to/model.pth
```