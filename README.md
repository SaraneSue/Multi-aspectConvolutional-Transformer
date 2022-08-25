# Multi-aspect Convolutional-Transformer

This repository is an implementation of "Multi-Aspect Convolutional-Transformer Network for SAR Automatic Target Recognition", in Remote Sensing, 2022. 

> [Multi-Aspect Convolutional-Transformer Network for SAR Automatic Target Recognition](https://www.mdpi.com/2072-4292/14/16/3924)
> *Remote Sens.* 2021.
> Siyuan Li, [Zongxu Pan](http://people.ucas.ac.cn/~panzx), Yuxin Hu.

## Abstract

In recent years, synthetic aperture radar (SAR) automatic target recognition (ATR) has been widely used in both military and civilian fields. Due to the sensitivity of SAR images to the observation azimuth, the multi-aspect SAR image sequence contains more information for recognition than a single-aspect one. Nowadays, multi-aspect SAR target recognition methods mainly use recurrent neural networks (RNN), which rely on the order between images and thus suffer from information loss. At the same time, the training of the deep learning model also requires a lot of training data, but multi-aspect SAR images are expensive to obtain. Therefore, this paper proposes a multi-aspect SAR recognition method based on self-attention, which is used to find the correlation between the semantic information of images. Simultaneously, in order to improve the anti-noise ability of the proposed method and reduce the dependence on a large amount of data, the convolutional autoencoder (CAE) used to pretrain the feature extraction part of the method is designed. The experimental results using the MSTAR dataset show that the proposed multi-aspect SAR target recognition method is superior in various working conditions, performs well with few samples and also has a strong ability of anti-noise.

![](https://github.com/SaraneSue/Multi-aspectConvolutional-Transformer/blob/main/images/network.png)

## Requirements

-Python 3.6

-Pytorch 1.9.1

-cuda 10.2

-sklearn 0.24.2

-tensorboard 1.15.0

-pillow 8.13.2

## Preparing the data

1.The MSTAR dataset can be downloaded from: https://www.sdms.afrl.af.mil/

2.Run data_processing.py to read the raw data to generate images.

3.Run sequence_construction.py to construct multi-aspect SAR image sequences.

## Training and testing

1.Configure the associated hyperparameters and file paths in the following files:

```
train.py -- file paths and training parameters
cct.py -- parameters of the proposed network structure
encoder_decoder.py -- parameters of the CAE structure
```

2.Execute the following code to pretrain CAE： 

```bash
python encoder_decoder.py
```

3.Execute the following code for training and testing： 

```bash
python train.py
```

## Citation

 If you find this repository/work helpful in your research, welcome to cite the paper. 

```bibtex
@Article{rs14163924,
    AUTHOR = {Li, Siyuan and Pan, Zongxu and Hu, Yuxin},
    TITLE = {Multi-Aspect Convolutional-Transformer Network for SAR Automatic Target Recognition},
    JOURNAL = {Remote Sensing},
    VOLUME = {14},
    YEAR = {2022},
    NUMBER = {16},
    ARTICLE-NUMBER = {3924},
    URL = {https://www.mdpi.com/2072-4292/14/16/3924},
    ISSN = {2072-4292},
    DOI = {10.3390/rs14163924}
}
```

