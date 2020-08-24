from __future__ import print_function
from __future__ import division
import argparse
from glob import glob
import natsort

import tensorflow as tf
 
from model import denoiser
from utils import *
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for sgd')
parser.add_argument('--sigma', dest='sigma', type=float, default=25.0, help='noise level')
parser.add_argument('--eps', dest='eps', type=float, default=0.0035, help='epsilon')

parser.add_argument('--data', dest='data', default='./data/img_clean_pats.npy', help='training data path')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--log_dir', dest='log_dir', default='./logs', help='tensorboard logs are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--eval_set', dest='eval_set', default='BSD68', help='dataset for eval in training')
parser.add_argument('--test_set', dest='test_set', default='Set12', help='dataset for testing')

parser.add_argument('--cost', dest='cost', default='sure', help='cost to minimize')
parser.add_argument('--gpu', dest='gpu', default='1', help='which gpu to use')
parser.add_argument('--type', dest='type', default='', help='arg to give unique names to realizations')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')

args = parser.parse_args()


def denoiser_train(denoiser, lr):
    eval_files = natsort.natsorted(glob('./data/test/{}/*.png'.format(args.eval_set)))
    
    denoiser.train(args.data, eval_files, batch_size=args.batch_size, epoch=args.epoch, lr=lr)


def denoiser_test(denoiser, save_dir):
    noisy_test_set_path = './data/test/{}/sigma{}'.format(args.test_set, int(args.sigma))
    assert os.path.exists(noisy_test_set_path) == True, 'No noisy test set available! Create using create_noisy_test_sets.py'
    noisy_files = natsort.natsorted(glob('{}/*.npy'.format(noisy_test_set_path)))
    test_files = natsort.natsorted(glob('./data/test/{}/*.png'.format(args.test_set)))
    
    denoiser.test(test_files, noisy_files, save_dir)

def main(_):
    
    #the following string is attached to checkpoint, log and image folder names
    eps_str = str(int(np.log10(args.eps)))
    name = "DnCNN_" + args.cost + str(args.type) + "_sigma" + str(int(args.sigma))
    
    ckpt_dir = args.ckpt_dir + "/" + name
    sample_dir = args.sample_dir + "/" + name
    test_dir = args.test_dir + "/" + name
    log_dir = args.log_dir + "/" + name
    
    
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    lr = args.lr * np.ones([args.epoch])
    lr[40:] = lr[0] / 10.0 #lr decay

    
    if args.use_gpu:
        # added to control the gpu memory
        print("GPU\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess, sigma=args.sigma, eps=args.eps, cost_str=args.cost, ckpt_dir=ckpt_dir, sample_dir=sample_dir, log_dir=log_dir)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model, test_dir)
            else:
                print('[!]Unknown phase')
                exit(0)
    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = denoiser(sess, sigma=args.sigma, eps=args.eps, cost_str=args.cost, ckpt_dir=ckpt_dir, sample_dir=sample_dir, log_dir=log_dir)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model, test_dir)
            else:
                print('[!]Unknown phase')
                exit(0)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    tf.app.run()
