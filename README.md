# Carvana Image Masking Using UNet Model

This repository contains a PyTorch implementation of a UNet model for image masking. The model is trained on the Carvana dataset and is designed to predict the mask for a given image.

## Overview
- The Carvana dataset consists of images of cars with their respective masks.
- The goal is to develop a model that can predict the mask for a given image.
- This repository provides a PyTorch implementation of a UNet model that achieves this goal.

## Components
`UNet Model:`
The UNet model is a deep learning architecture designed for image segmentation tasks. It consists of a series of convolutional and pooling layers followed by upsampling and convolutional layers.

`Carvana Dataset:`
The Carvana dataset is used to train and validate the model. It consists of images and their corresponding masks.

`Training:`
The model is trained using the AdamW optimizer and the binary cross-entropy loss function.

`Testing:`
The trained model is used to predict the mask for a given image.

## Usage

Training: Run the training script to train the model on the Carvana dataset.

Testing: Use the trained model to predict the mask for a given image.

## Results
The model achieves a high accuracy on the validation set, indicating its effectiveness in predicting the mask for a given image.

## License
Lmao no License
