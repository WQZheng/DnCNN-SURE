import numpy as np
from utils import *
import os.path

data_path = './data/img_clean_pats_blind.npy'
data = np.load(data_path).astype(np.float32) / 255.0

#generating sigma values in range [0.1, 55]
#We set sigma>0.1, to avoid having too small epsilon values
sigma_vector = np.random.uniform(0.1,55.0, data.shape[0]).astype('float32')

#generating corrupted training images
data_noisy = data + np.array([np.random.normal(0, sigma/255.0, (50,50,1)) for sigma in sigma_vector], dtype='float32')

np.save('./data/sigma_vector_0_55.npy', sigma_vector)
np.save('./data/img_noisy_pats_blind.npy', data_noisy)
