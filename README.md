# UltraMamba: Mamba-based Multimodal Ultrasound Image Adaptive Fusion for Breast Lesion Segmentation

🎉 This work is published in [IEEE Transactions on Medical Imaging](https://ieeexplore.ieee.org/document/11348936)

[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-deepang/ultramamba-yellow)](https://huggingface.co/deepang/ultramamba)&nbsp;

# Network Architecture
![Visualization](./figures/fig_framewor.png)

# Data Description
Dataset Name: MUB2024

Modality: SWT, SWV, BUS

Challenge: Breast Lesion Segmentation Challenge

The dataset will be publicly released within one week of paper acceptance.

The BreLS dataset was meticulously assembled from diagnostic imaging performed on 506 breast cancer patients
undergoing neoadjuvant therapy at Sun Yat-sen University Cancer Center and Sun Yat-sen Memorial Hospital between
October 2016 and May 2023. This robust dataset employs a Siemens Acuson S2000 ultrasound diagnostic instrument
equipped with a 9L4 linear array probe, operating within a frequency range of 4.0MHz to 9.0MHz, to ensure high-
resolution image capture.

Data samples can be found in the folder ./data

# Benchmark
Performance comparative analysis of different network architectures for breast lesion segmentation on the BreLS dataset.
![Visualization](./figures/fig_benchmark.png)


# Visualization
Error map visualizations of the UltraMamba and baseline methods in the BreLS datasets. The correct segmentation
position and the incorrect or missing portion of the detection are in green and red, respectively.
![Visualization](./figures/fig_errormap.png)

# Training
The ```config.yml``` is the global parameters control file.


## Training from scratch on single GPU
Adjust specific training parameters in ```  config.yml ```, and:
``` bash
python train.py
```

## Training from scratch on multi-GPU
Adjust the specific training parameters in ```config.yml``` and decide the devices in ```train.sh```, and:
``` bash
sh train.sh
```

# Evaluation
Evaluate the segmentation performance of UltraMamba on a single GPU: 
``` bash
python verify.py
```
Evaluate the segmentation performance of UltraMamba on multi-GPU:
``` bash
sh verify.sh
```

# Bixtex
```bib
@ARTICLE{11348936,
  author={Huang, Jiahui and Huang, Jiaxin and Zhang, Mingdu and Wang, Qiong and Pei, Xiao-Qing and Hu, Ying and Chen, Hao and Pang, Yan},
  journal={IEEE Transactions on Medical Imaging}, 
  title={UltraMamba: Mamba-based Multimodal Ultrasound Image Adaptive Fusion for Breast Lesion Segmentation}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Lesions;Image segmentation;Breast;Ultrasonic imaging;Accuracy;Feature extraction;Attention mechanisms;Image color analysis;Elastography;Decoding;Multimodal Ultrasound Imaging;Breast Lesion Segmentation;B-mode Ultrasound;Shear Wave Velocity;Shear Wave Time},
  doi={10.1109/TMI.2026.3653779}}
```
 