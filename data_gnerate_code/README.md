## Overview
This floder provides audio data augmentation used in Adaptive Mixture of Low-Rank Experts for Robust Audio Spoofing Detection. It implements various augmentation methods to enhance the robustness of anti-spoofing systems by generating diverse training data.

## Features
### Noise Augmentation (`apply_Noise.py`)
- Gaussian noise
- Gaussian SNR noise
- Colored noise
- Configurable noise intensity levels

### Filter Augmentation (`apply_filter.py`)
- Band-pass filter
- High-pass filter
- Low-pass filter
- Band-stop filter
- High-shelf filter
- Low-shelf filter
- Peaking filter

### Combined Augmentation (`noise_and_filter.py`)
- Noise and filter combinations (21 different combinations)
- Two processing orders: (1)Filter-first: Apply filter before adding noise. (2)Noise-first: Add noise before applying filter

### Rawboost Augmentation (`rawboost.py`)
- Enhance data using different Rawboost algorithms

### Core Augmentation Functions (`aug_wav.py`)
- Base implementation of all augmentation methods
- Customizable parameters for each method
- Support for both training and evaluation datasets

