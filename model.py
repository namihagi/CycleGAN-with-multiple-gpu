import json
import time
from glob import glob

from tqdm import tqdm

from module import *
from utils import *

CONFIG_PATH = 'config.json'


class Model(object):
    def __init__(self, sess, args):
        self.sess = sess
        config = json.load(open(CONFIG_PATH))
        self.model_params = config['model_params']
        self.input_params = config['input_params']
        self.dir_params = config['dir_params']

        # model setting
        self.gpu_num = args.gpu_num
        self.GLOBAL_BATCH_SIZE = args.global_batch_size
        assert self.GLOBAL_BATCH_SIZE % self.gpu_num == 0
        self.BATCH_SIZE_PER_GPU = self.GLOBAL_BATCH_SIZE // self.gpu_num
        self.g_dim = self.model_params['g_dim']
        self.d_dim = self.model_params['d_dim']
        self.L1_lambda = self.model_params['L1_lambda']
        self.beta1 = self.model_params['beta1']
        self.lr = self.model_params['lr']
        self.epoch = self.model_params['epoch']
        self.epoch_step = self.model_params['epoch_step']

        # input setting
        self.dataset_name = self.input_params['dataset_name']
        self.image_size = self.input_params['image_size']
        self.input_c_dim = self.input_params['input_c_dim']
        self.output_c_dim = self.input_params['output_c_dim']

        self.options = {'g_dim': self.g_dim, 'd_dim': self.d_dim,
                        'input_c_dim': self.input_c_dim, 'output_c_dim': self.output_c_dim}

        # network setting
        self.generator = generator_resnet
        self.discriminator = discriminator
        self.criterionGAN = mae_criterion

        # directory setting
        self.dataset_dir = os.path.join(self.dir_params['dataset_dir'], self.dataset_name)
        self.sample_dir = os.path.join(self.dir_params['sample_dir'], args.sub_dir)
        self.test_dir = os.path.join(self.dir_params['test_dir'], args.sub_dir)
        self.log_dir = os.path.join(self.dir_params['log_dir'], args.sub_dir)
        self.checkpoint_dir = os.path.join(self.dir_params['checkpoint_dir'], args.sub_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # initialize graph
        self._build_model()
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter('{}'.format(self.log_dir), self.sess.graph)

    def _build_model(self):
        # placeholder
        self.real_A = tf.placeholder(tf.float32,
                                     [self.GLOBAL_BATCH_SIZE, self.image_size,
                                      self.image_size, self.input_c_dim],
                                     name="real_A")
        self.real_B = tf.placeholder(tf.float32,
                                     [self.GLOBAL_BATCH_SIZE, self.image_size,
                                      self.image_size, self.input_c_dim],
                                     name="real_B")
        # divide real images
        self.real_A_per_gpu = tf.split(self.real_A, self.gpu_num)
        self.real_B_per_gpu = tf.split(self.real_B, self.gpu_num)

        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [self.gpu_num, self.BATCH_SIZE_PER_GPU, self.image_size,
                                             self.image_size, self.input_c_dim],
                                            name="fake_sample")
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [self.gpu_num, self.BATCH_SIZE_PER_GPU, self.image_size,
                                             self.image_size, self.input_c_dim],
                                            name="fake_sample")

        self.g_losses = []
        self.d_losses = []
        self.fake_A_list = []
        self.fake_B_list = []
        for gpu_id in range(int(self.gpu_num)):
            reuse = (gpu_id > 0)
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                    # related to generator update
                    # A > B > A
                    self.fake_B = self.generator(self.real_A_per_gpu[gpu_id], self.options,
                                                 reuse=reuse, name="generatorA2B")
                    self.fake_A_return = self.generator(self.fake_B, self.options,
                                                        reuse=reuse, name="generatorB2A")
                    # B > A > B
                    self.fake_A = self.generator(self.real_B_per_gpu[gpu_id], self.options,
                                                 reuse=True, name="generatorB2A")
                    self.fake_B_return = self.generator(self.fake_A, self.options,
                                                        reuse=True, name="generatorA2B")

                    # loss for generator
                    self.DB_fake = self.discriminator(self.fake_B, self.options,
                                                      reuse=reuse, name="discriminatorB")
                    self.DA_fake = self.discriminator(self.fake_A, self.options,
                                                      reuse=reuse, name="discriminatorA")
                    self.loss_A_cyc = self.L1_lambda * abs_criterion(self.real_A_per_gpu[gpu_id], self.fake_A_return)
                    self.loss_B_cyc = self.L1_lambda * abs_criterion(self.real_B_per_gpu[gpu_id], self.fake_B_return)
                    self.g_loss_per_gpu = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
                                          + self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
                                          + self.loss_A_cyc + self.loss_B_cyc
                    self.g_losses.append(self.g_loss_per_gpu)

                    # store fake image for discriminator
                    self.fake_A_list.append(self.fake_A)
                    self.fake_B_list.append(self.fake_B)

                    # related to discriminator update
                    self.DA_real = self.discriminator(self.real_A_per_gpu[gpu_id], self.options,
                                                      reuse=True, name="discriminatorA")
                    self.DB_real = self.discriminator(self.real_B_per_gpu[gpu_id], self.options,
                                                      reuse=True, name="discriminatorB")
                    self.DA_fake_sample = self.discriminator(self.fake_A_sample[gpu_id], self.options,
                                                             reuse=True, name="discriminatorA")
                    self.DB_fake_sample = self.discriminator(self.fake_B_sample[gpu_id], self.options,
                                                             reuse=True, name="discriminatorB")

                    # discriminatorA loss
                    self.d_A_real_loss = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_fake))
                    self.d_A_fake_loss = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
                    self.d_A_loss = (self.d_A_real_loss + self.d_A_fake_loss) / 2.0
                    # discriminatorB loss
                    self.d_B_real_loss = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_fake))
                    self.d_B_fake_loss = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
                    self.d_B_loss = (self.d_B_real_loss + self.d_B_fake_loss) / 2.0
                    # all discriminator loss
                    self.d_loss = self.d_A_loss + self.d_B_loss
                    self.d_losses.append(self.d_loss)

        # compute all loss over gpu
        self.g_total_loss = tf.reduce_mean(tf.stack(self.g_losses, axis=0))
        self.d_total_loss = tf.reduce_mean(tf.stack(self.d_losses, axis=0))
        self.g_total_loss_sum = tf.summary.scalar("generator_loss", self.g_total_loss)
        self.d_total_loss_sum = tf.summary.scalar("discriminator_loss", self.d_total_loss)

        # get variables
        self.t_vars = tf.trainable_variables()
        self.g_vars = [var for var in self.t_vars if 'generator' in var.name]
        self.d_vars = [var for var in self.t_vars if 'discriminator' in var.name]
        for var in self.t_vars:
            print(var.name)

        # optimizer
        self.lr_ph = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr_ph, beta1=self.beta1) \
            .minimize(self.d_total_loss, var_list=self.d_vars, colocate_gradients_with_ops=True)
        self.g_optim = tf.train.AdamOptimizer(self.lr_ph, beta1=self.beta1) \
            .minimize(self.g_total_loss, var_list=self.g_vars, colocate_gradients_with_ops=True)

    def train(self):
        # initialize variables
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # load train filenames
        A_files = glob('{}/trainA/*.jpg'.format(self.dataset_dir))
        B_files = glob('{}/trainB/*.jpg'.format(self.dataset_dir))
        batch_idxs = min(len(A_files), len(B_files)) // self.GLOBAL_BATCH_SIZE

        # training loop
        counter = 0
        start_time = time.time()
        for epoch in range(self.epoch):
            # lr decay
            if epoch < self.epoch_step:
                lr = np.float32(self.lr)
            else:
                lr = self.lr * (self.epoch - epoch) / (self.epoch - self.epoch_step)
            np.random.shuffle(A_files)
            np.random.shuffle(B_files)
            with tqdm(range(batch_idxs)) as bar:
                bar.set_description('epoch: {:>03d} '.format(epoch))
                for idx in bar:
                    # load images
                    batch_A_files = A_files[idx * self.GLOBAL_BATCH_SIZE:(idx + 1) * self.GLOBAL_BATCH_SIZE]
                    batch_B_files = B_files[idx * self.GLOBAL_BATCH_SIZE:(idx + 1) * self.GLOBAL_BATCH_SIZE]
                    batch_A_images = [load_test_data(batch_file, self.image_size) for batch_file in batch_A_files]
                    batch_B_images = [load_test_data(batch_file, self.image_size) for batch_file in batch_B_files]
                    batch_A_images = np.array(batch_A_images).astype(np.float32)
                    batch_B_images = np.array(batch_B_images).astype(np.float32)

                    # update G
                    fake_A_list, fake_B_list, _, g_sum = self.sess.run([self.fake_A_list, self.fake_B_list,
                                                                        self.g_optim, self.g_total_loss_sum],
                                                                       feed_dict={self.real_A: batch_A_images,
                                                                                  self.real_B: batch_B_images,
                                                                                  self.lr_ph: lr})
                    self.writer.add_summary(g_sum, counter)

                    # update D
                    _, d_sum = self.sess.run([self.d_optim, self.d_total_loss_sum],
                                             feed_dict={self.fake_A_sample: fake_A_list,
                                                        self.fake_B_sample: fake_B_list,
                                                        self.real_A: batch_A_images,
                                                        self.real_B: batch_B_images,
                                                        self.lr_ph: lr})
                    self.writer.add_summary(d_sum, counter)

                    # save samples and checkpoints
                    if (idx + 1) == batch_idxs:
                        self.save_samples(epoch)
                        self.save(self.checkpoint_dir, counter)
                    counter += 1

        # save model when finish training
        self.save(self.checkpoint_dir, counter)
        print('the total time is %4.4f' % (time.time() - start_time))

    def test(self):
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        self.save_samples(self.test_dir)
        pass

    def save(self, checkpoint_dir, counter):
        model_name = "{}.model".format(self.dataset_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=counter)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def save_samples(self, epoch):
        # load test filenames
        A_files = glob('{}/testA/*.jpg'.format(self.dataset_dir))
        B_files = glob('{}/testB/*.jpg'.format(self.dataset_dir))
        # load images
        batch_A_files = A_files[:self.GLOBAL_BATCH_SIZE]
        batch_B_files = B_files[:self.GLOBAL_BATCH_SIZE]
        batch_A_images = [load_test_data(batch_file, self.image_size) for batch_file in batch_A_files]
        batch_B_images = [load_test_data(batch_file, self.image_size) for batch_file in batch_B_files]
        batch_A_images = np.array(batch_A_images).astype(np.float32)
        batch_B_images = np.array(batch_B_images).astype(np.float32)
        fake_A_sample, fake_B_sample = self.sess.run([self.fake_A_list, self.fake_B_list],
                                                     feed_dict={self.real_A: batch_A_images,
                                                                self.real_B: batch_B_images})
        fake_A_sample = np.concatenate(fake_A_sample, axis=0)
        fake_B_sample = np.concatenate(fake_B_sample, axis=0)

        # save images
        save_images(batch_A_files, fake_A_sample, self.sample_dir, "B2A", epoch)
        save_images(batch_B_files, fake_B_sample, self.sample_dir, "A2B", epoch)
