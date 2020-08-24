# Instructions for SURE based training for DnCNN

**Shakarim Soltanayev**

## Getting Started

This code was tested with Python 2.7. It is highly recommended to use the GPU version of Tensorflow for fast training.

### Prerequisites
```
numpy==1.16.4
Pillow==6.0.0
scipy==1.2.1
tensorflow==1.13.1
```

## Single noise level denoising

### Generating the datasets
You should generate the datasets before starting training.

If there is no 'img_clean_pats.npy' file in the ./data folder, then generate the file using:
```
python2 generate_patches.py
```
The noisy patches will be generated automatically during training.

### Training the network
You can control paramaters such as batch size, epsilon value. More info inside main.py.

Example to train DnCNN with SURE:
```
python2 main.py --phase train --cost sure --sigma 25.0 --eps 0.0035
```
```
python2 main.py --phase train --cost sure --sigma 50.0 --eps 0.007
```
Example to train DnCNN with MSE:
```
python2 main.py --phase train --cost mse --sigma 25.0
```

### Testing using the trained network

Set the --phase to test and run the main.py file Be sure to correctly set other arguments such as cost, sigma etc., otherwise checkpoints files will not be found. Denoised images are saved in ./test folder.

The noisy test images for BSD68 and Set12 are already in the ./data/test folder. If you want to create noisy images from your own dataset and for other noise levels use the create_noisy_test_sets.py file.

## Blind denoising (sigma=[0,55])

### Generating the datasets
You should generate the datasets before starting training.

If there is no 'img_clean_pats_blind.npy' file in the ./data folder, then generate the file using:
```
python2 generate_patches_blind.py
```
If there is no 'img_noisy_pats_blind.py' file in the ./data folder, then generate the file using:
```
python2 create_noisy_training_set_blind.py
```

### Training the network
You can control paramaters such as batch size, epsilon value. More info inside main_blind.py.

Example to train DnCNN-B with SURE:
```
python2 main_blind.py --phase train --cost sure
```
```
python2 main_blind.py --phase train --cost sure
```
Example to train DnCNN-B with MSE:
```
python2 main_blind.py --phase train --cost mse
```
You can also specify sigma to evaluate with specific sigma level during training.

### Testing using the trained network

Set the --phase to test and run the main_blind.py file. Be sure to correctly set other arguments such as cost, sigma etc., otherwise checkpoints files will not be found. Denoised images are saved in ./test folder.

The noisy test images for BSD68 and Set12 are already in the ./data/test folder. If you want to create noisy images from your own dataset and for other noise levels use the create_noisy_test_sets.py file.

### Tensorboard summaries
```
tensorboard --logdir=./logs
```


