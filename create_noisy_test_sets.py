import numpy as np
from utils import *
from scipy.misc import imsave as ims
from glob import glob
import natsort
import os.path

sigma=25
dataset = 'Set12'

dataset_path = './data/test/%s' % (dataset)
save_path = '%s/sigma%d' % (dataset_path, int(sigma))
if not os.path.exists(save_path):
    os.makedirs(save_path)

eval_files = natsort.natsorted(glob('%s/*.png' % (dataset_path)))

for idx in xrange(len(eval_files)):
    clean_image = load_images(eval_files[idx]).astype(np.float32) / 255.0
    noisy_image = clean_image + np.random.normal(0, sigma/255.0, np.shape(clean_image)).astype('float32')
    
    img_name, img_ext = os.path.splitext(os.path.basename(eval_files[idx]))
    
    name = '%s/%s.npy' % (save_path, img_name)
    
    np.save(name, noisy_image)
