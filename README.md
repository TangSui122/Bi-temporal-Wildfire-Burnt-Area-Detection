📌 **Note:** A newer and improved version of this project is available at:  
[https://github.com/TangSui122/Bi-temporal-Wildfire-Burnt-Area-Detection](https://github.com/TangSui122/Bi-temporal-Wildfire-Burnt-Area-Detection)

# Project 

Bi-temporal Convolutional Neural Network based on U-Net Architecture for Wildfire Burnt Area Detection.
The purpose of this project is to provide more accurate prediction results compared to publicly available products, utilizing a relaxed dataset requirement and achieving higher spatial resolution. The model utilizes two temporal images of the same location, pre and post-wildfire occurrences, in the form of false-color images for better distinction between burnt and unburnt areas.

## [File: train_AttentionUnet.py](segmentation/train_AttentionUnet.py)

The `train_AttentionUnet.py` contains scripts for model training and testing. The neural network's hyperparameters can be adjusted using different parameters. It supports the following models and loss functions:

- U-Net
- Bi-temporal U-Net
- U-Net with attention gate
- Bi-temporal U-Net with attention gate
- Various loss functions

## [File: AttentionUnet.py](segmentation/models/AttentionUnet.py)

The `AttentionUnet.py` file contains the architecture of the neural network model. This script defines the structure of the Attention U-Net used for burnt area detection caused by wildfires.

## [File: BADMDataset_set.py](segmentation/utils/BADMDataset_set.py)

The `BADMDataset_set.py` script initializes the dataset required for training and testing. It deals with dual temporal images, each having dimensions of 256x256x3, and produces predictions of size 128x128x1.

## Dataset

If you require access to the dataset or have any related inquiries, please feel free to reach out to me via email:

Email: tsui5@wisc.edu
