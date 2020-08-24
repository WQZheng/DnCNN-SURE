from __future__ import print_function
from __future__ import division
import time

from utils import *
#from scipy.misc import imsave as ims

def dncnn(input, is_training=True, output_channels=1):
    with tf.variable_scope('DnCNN', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('block1'):
            output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
        for layers in range(2, 16 + 1):
            with tf.variable_scope('block%d' % layers):
                output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
                output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training, name='bn%d' % layers))
        with tf.variable_scope('block17'):
            output = tf.layers.conv2d(output, output_channels, 3, padding='same')
    return input-output


class denoiser(object):
    def __init__(self, sess, sigma, eps, cost_str, ckpt_dir, sample_dir, log_dir):
        self.sess = sess
        self.sigma = sigma
        self.eps = eps
        
        
        self.ckpt_dir = ckpt_dir
        self.sample_dir = sample_dir
        self.log_dir = log_dir
        
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        
        # build model
        #placeholders for clean and noisy image batches
        self.X = tf.placeholder(tf.float32, [None, None, None, 1], name='clean_image')
        
        self.Y = tf.placeholder(tf.float32, [None, None, None, 1], name='noisy_image')
        #self.Y = self.X + tf.random_normal(shape=tf.shape(self.X), stddev=self.sigma / 255.0)  # noisy images
        
        self.is_training = tf.placeholder(tf.bool, name='is_training') #for batchnorm

        #forward propagation
        self.Y_ = dncnn(self.Y, is_training=self.is_training)
            
        #forward propagation of the perturbed input
        self.n = tf.random_normal(shape=tf.shape(self.Y), stddev=1.0)
        self.Z = self.Y + self.n*self.eps

        self.Z_ = dncnn(self.Z, is_training=self.is_training)
        
        self.dim1 = tf.cast(tf.shape(self.Y)[1], tf.float32) #height
        self.dim2 = tf.cast(tf.shape(self.Y)[2], tf.float32) #width
        
        self.var = (self.sigma/255.0)**2
        batch = tf.cast(tf.shape(self.Y)[0], tf.float32) #size of the minibatch

        #######################################-COST FUNCTIONS-#######################################
        self.loss = (1.0 / batch) * tf.nn.l2_loss(self.Y_ - self.X) #MSE


        self.divergence = (1.0/self.eps)*(tf.reduce_sum(tf.multiply(self.n, (self.Z_-self.Y_))))
        self.sure = (1.0 / batch)*(tf.nn.l2_loss(self.Y - self.Y_)
            - batch*self.dim1*self.dim2*self.var/2.0 + self.var*self.divergence) #MC-SURE

        #which cost function to use for training
        if cost_str=='sure':
            self.cost = self.sure
        elif cost_str=='mse':
            self.cost = self.loss
        else:
            print("UNKNOWN COST")

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr = tf.placeholder(tf.float32, name='eva_psnr')

        #optimizer
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        #for batchnorm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.cost)

        #checkpoint saver
        self.saver = tf.train.Saver(max_to_keep=10)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    #function to evaluate the performance after each epoch
    def evaluate(self, test_files, iter_num, summary_merged, summary_writer):
        
        psnr_sum = 0
        for idx in range(len(test_files)):
            clean_image = load_images(test_files[idx]).astype(np.float32) / 255.0
            noisy_image = clean_image + np.random.normal(0, self.sigma/255.0, np.shape(clean_image)).astype('float32')
            output_clean_image = self.sess.run(self.Y_, feed_dict={self.X: clean_image, self.Y: noisy_image, self.is_training: False})
            
            groundtruth = np.clip(255 * clean_image, 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            if idx<10:
                print("%s PSNR: %.2f" % (os.path.basename(test_files[idx]), psnr))
            
            psnr_sum += psnr
            img_name, img_ext = os.path.splitext(os.path.basename(test_files[idx]))
            save_images(os.path.join(self.sample_dir, 'denoised_%s_%d.png' % (img_name, iter_num)), groundtruth, noisyimage, outputimage)


        avg_psnr = psnr_sum / len(test_files)

        print("---- Validation ---- Average PSNR %.2f ---" % avg_psnr)

        psnr_summary = self.sess.run(summary_merged, feed_dict={self.eva_psnr:avg_psnr})
        summary_writer.add_summary(psnr_summary, iter_num)


    def train(self, data_path, eval_files, batch_size, epoch, lr):
        # normalize the data to 0-1
        data = np.load(data_path).astype(np.float32) / 255.0
        print("DATA RANGE:")
        print(np.amax(data))
        print(np.amin(data))
        numData = np.shape(data)[0]
        numBatch = int(numData / batch_size)


        # load pretrained model
        load_model_status, global_step = self.load(self.ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = (global_step) // numBatch
            start_step = (global_step) % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")
        # make summary
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('sure', self.sure)
        tf.summary.scalar('lr', self.lr)
        writer = tf.summary.FileWriter(self.log_dir+"/", self.sess.graph)
        merged = tf.summary.merge_all()
        summary_psnr = tf.summary.scalar('eva_psnr', self.eva_psnr)
        
        #generating corrupted images (only once)
        data_noisy = data + np.random.normal(0, self.sigma/255.0, np.shape(data)).astype(np.float32)
        #data_noisy = np.load('./data/img_noisy_pats_25.npy', mmap_mode='r')
        print("NOISY DATA RANGE:")
        print(np.amax(data_noisy))
        print(np.amin(data_noisy))
        
        
        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        self.evaluate(eval_files, iter_num, summary_merged=summary_psnr, summary_writer=writer)
        
        tf.get_default_graph().finalize() #making sure that the graph is fixed at this point
        
        print("Training set shape:")
        print(np.shape(data_noisy))
        #training loop
        for epoch in range(start_epoch, epoch):
            print("Model: %s" % (self.ckpt_dir))
            print("Learning rate: {}".format(lr[epoch]))
            rand_inds=np.random.choice(numData, numData,replace=False) # for shuffling
            
            for batch_id in range(0, numBatch):
                batch_rand_inds = rand_inds[batch_id * batch_size:(batch_id + 1) * batch_size]
                batch_images = data[batch_rand_inds]
                batch_images_corrupt = data_noisy[batch_rand_inds]
                feed_dict_train = {self.X: batch_images, self.Y: batch_images_corrupt, self.lr: lr[epoch], self.is_training: True}
                
                # training optimization step
                self.sess.run(self.train_op, feed_dict=feed_dict_train)
                
                # display the losses
                if (iter_num)%100==0:
                    feed_dict_test = {self.X: batch_images, self.Y: batch_images_corrupt, self.lr: lr[epoch], self.is_training: False}
                    loss, sure, summary = self.sess.run([self.loss, self.sure, merged],feed_dict=feed_dict_test)
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f"
                          % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time))
                    print("TRAIN, true: %.6f, sure: %.6f" % (loss, sure))
                    print("\n")
                    writer.add_summary(summary, iter_num)

                if (iter_num+1)%(numBatch//2)==0:
                    self.evaluate(eval_files, iter_num+1, summary_merged=summary_psnr, summary_writer=writer)
                    self.save(iter_num+1, self.ckpt_dir)
                iter_num += 1
        print("[*] Finish training.")
    
    #checkpoint saver
    def save(self, iter_num, ckpt_dir, model_name='DnCNN-tensorflow'):
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        self.saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)
    
    #checkpoint loader
    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            self.saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    
    def test(self, test_files, noisy_files, save_dir):
        """Test DnCNN"""
        # init variables
        tf.initialize_all_variables().run()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        load_model_status, global_step = self.load(self.ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print("[*] Load weights SUCCESS...")

        psnr_sum = 0
        print("Model: %s" % (save_dir))
        print("[*] " + 'noise level: ' + str(self.sigma) + " start testing...")
        for idx in range(len(test_files)):
            clean_image = load_images(test_files[idx]).astype(np.float32) / 255.0
            noisy_image = np.load(noisy_files[idx]).astype(np.float32)
            #noisy_image = clean_image + np.random.normal(0, self.sigma/255.0, np.shape(clean_image))
            output_clean_image = self.sess.run(self.Y_, feed_dict={self.X: clean_image, self.Y: noisy_image, self.is_training: False})

            groundtruth = np.clip(255 * clean_image, 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            # calculate PSNR
            noisy_psnr = cal_psnr(noisyimage, outputimage)
            psnr = cal_psnr(groundtruth, outputimage)
            
            img_name, img_ext = os.path.splitext(os.path.basename(test_files[idx]))
            print("%s Noisy PSNR: %.2f, Denoised PSNR: %.2f" % (os.path.basename(test_files[idx]), noisy_psnr, psnr))
            psnr_sum += psnr
            save_images(os.path.join(save_dir, 'noisy_%s.png' % img_name), noisyimage)
            save_images(os.path.join(save_dir, 'denoised_%s.png' % img_name), outputimage)

        avg_psnr = psnr_sum / len(test_files)

        print("---- Test ---- Average PSNR %.2f ---" % avg_psnr)
