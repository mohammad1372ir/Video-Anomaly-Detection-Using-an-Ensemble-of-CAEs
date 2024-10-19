# Video-Anomaly-Detection-Using-an-Ensemble-of-CAEs
This repository contains python scripts for training and testing ensembles of convolutional auto-encoders for video anomaly detection.



There are four files in this repository. You need to run these files in the following order:

- spatial_features.py
- train_caes.py
- test_caes.py
- inference_caes.py

The first file uses **vit_keras** library. To run this file, you need to have the following libraries:
- tensorflow 2.13
- keras 2.13.1
- keras_applications 1.0.8
- tensorflow_addons
- vit_keras

The first file extracts spatial features using a vision transformer

Usigng the outputs of the first file, the second code trains an ensemble convolutional auto-encoders (CAEs).

The third code uses the trained CAEs and calculate results of the test data.

the fourth file uses the outputs of the third file and find normalities and anomalies.
