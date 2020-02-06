import json
import time
from glob import glob

from tqdm import tqdm

from input_pipeline import ImagePipeline, NpyPipeline
from module import *
from utils import *


class Model(object):
    def __init__(self, sess, args):
        self.sess = sess

        # model setting
        self.gpu_num = args.gpu_num
        self.GLOBAL_BATCH_SIZE = args.global_batch_size
        assert self.GLOBAL_BATCH_SIZE % self.gpu_num == 0
        self.epoch = args.epoch
        self.lr_decay_epoch = args.lr_decay_epoch
        self.lr = args.lr
        self.hidden_dim = args.hidden_dim
        self.L1_lambda = args.L1_lambda
        self.beta1 = args.hidden_dim

        # input setting
        self.image_size = args.image_size
        self.B_range = args.B_range

        # network setting
        self.G_A2B = generator_resnet(name='generatorA2B')
        self.G_B2A = generator_resnet(name='generatorB2A')
        self.D_A = discriminator(name='discriminatorA', hidden_dim=self.hidden_dim)
        self.D_B = discriminator(name='discriminatorB', hidden_dim=self.hidden_dim)
        self.criterionGAN = mae_criterion

        # directory setting
        self.dataset_dir = args.dataset_dir
        self.datasetA = args.datasetA
        self.datasetB = args.datasetB
        if args.sub_dir is None:
            self.sub_dir = self.datasetB + '2' + self.datasetA
        else:
            self.sub_dir = args.sub_dir
        assert os.path.exists(self.dataset_dir), "not exists {}".format(self.dataset_dir)
        self.sample_dir = os.path.join('./sample', args.sub_dir)
        self.result_dir = os.path.join('./result', args.sub_dir)
        self.log_dir = os.path.join('./log', args.sub_dir)
        self.checkpoint_dir = os.path.join('./checkpoint', args.sub_dir)
        for dir_path in [self.sample_dir, self.result_dir, self.log_dir, self.checkpoint_dir]:
            makedirs(dir_path=dir_path)

        # sample data
        self.sample_A, self.sample_B = self.get_sample_data()

        # initialize graph
        self._build_model()
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        # initialize variables
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def _build_model(self):
        # placeholder
        self.real_img_A = tf.placeholder(tf.float32,
                                         [self.GLOBAL_BATCH_SIZE, self.image_size, self.image_size, 3],
                                         name="real_img_A")
        self.real_feat_B = tf.placeholder(tf.float32,
                                          [self.GLOBAL_BATCH_SIZE, 32, 32, 128],
                                          name="real_feat_B")
        # divide real images
        self.real_img_A_per_gpu = tf.split(self.real_img_A, self.gpu_num)
        self.real_feat_B_per_gpu = tf.split(self.real_feat_B, self.gpu_num)

        self.g_losses = []
        self.d_losses = []
        self.fake_A_list = []
        self.fake_B_list = []
        for gpu_id in range(int(self.gpu_num)):
            reuse = (gpu_id > 0)
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                # generator
                # A > B > A
                self.real_feat_A = FeatureExtractor(self.real_img_A_per_gpu[gpu_id], reuse=reuse, name='FE_A')
                self.fake_feat_B = self.G_A2B(self.real_feat_A, reuse=reuse)
                self.fake_feat_A_return = self.G_B2A(self.fake_feat_B, reuse=reuse)
                tf.add_to_collection('fake_feat_B', self.fake_feat_B)
                # B > A > B
                self.fake_feat_A = self.G_B2A(self.real_feat_B_per_gpu[gpu_id], reuse=True)
                self.fake_feat_B_return = self.G_A2B(self.fake_feat_A, reuse=True)
                tf.add_to_collection('fake_feat_A', self.fake_feat_A)

                # discriminator
                self.DA_fake_A_score = self.D_A(self.fake_feat_A, reuse=reuse)
                self.DA_real_A_score = self.D_A(self.real_feat_A, reuse=True)
                self.DB_fake_B_score = self.D_B(self.fake_feat_B, reuse=reuse)
                self.DB_real_B_score = self.D_B(self.real_feat_B_per_gpu[gpu_id], reuse=True)

                # loss for generator
                self.loss_A_cyc = abs_criterion(self.real_feat_A, self.fake_feat_A_return)
                self.loss_B_cyc = abs_criterion(self.real_feat_B_per_gpu[gpu_id], self.fake_feat_B_return)
                self.g_loss_per_gpu = self.criterionGAN(self.DA_fake_A_score, tf.ones_like(self.DA_fake_A_score)) \
                                      + self.criterionGAN(self.DB_fake_B_score, tf.ones_like(self.DB_fake_B_score)) \
                                      + self.L1_lambda * (self.loss_A_cyc + self.loss_B_cyc)
                tf.add_to_collection('G_loss', self.g_loss_per_gpu)

                # loss for D_A
                self.DA_fake_A_loss = self.criterionGAN(self.DA_fake_A_score, tf.zeros_like(self.DA_fake_A_score))
                self.DA_real_A_loss = self.criterionGAN(self.DA_real_A_score, tf.ones_like(self.DA_real_A_score))
                self.DA_loss = self.DA_fake_A_loss + self.DA_real_A_loss
                # loss for D_B
                self.DB_fake_B_loss = self.criterionGAN(self.DB_fake_B_score, tf.zeros_like(self.DB_fake_B_score))
                self.DB_real_B_loss = self.criterionGAN(self.DB_real_B_score, tf.ones_like(self.DB_real_B_score))
                self.DB_loss = self.DB_fake_B_loss + self.DB_real_B_loss
                # all discriminator loss
                self.D_loss = self.DA_loss + self.DB_loss
                tf.add_to_collection('D_loss', self.D_loss)

        # output for sample
        self.sample_feat_A = tf.get_collection('fake_feat_A')
        self.sample_feat_B = tf.get_collection('fake_feat_B')

        # compute all loss over gpu
        self.G_total_loss = tf.reduce_mean(tf.get_collection('G_loss'))
        self.D_total_loss = tf.reduce_mean(tf.get_collection('D_loss'))
        self.G_total_loss_sum = tf.summary.scalar("generator_loss", self.G_total_loss)
        self.D_total_loss_sum = tf.summary.scalar("discriminator_loss", self.D_total_loss)

        # get variables
        self.t_vars = tf.trainable_variables()
        self.g_vars = [var for var in self.t_vars if 'generator' or 'FE_A' in var.name]
        self.d_vars = [var for var in self.t_vars if 'discriminator' in var.name]
        for var in self.g_vars + self.d_vars:
            print(var.name)

        # optimizer
        self.lr_ph = tf.placeholder(tf.float32, name='learning_rate')
        self.D_optim = tf.train.AdamOptimizer(self.lr_ph, beta1=self.beta1) \
            .minimize(self.D_total_loss, var_list=self.d_vars, colocate_gradients_with_ops=True)
        self.G_optim = tf.train.AdamOptimizer(self.lr_ph, beta1=self.beta1) \
            .minimize(self.G_total_loss, var_list=self.g_vars, colocate_gradients_with_ops=True)

    def train(self):
        A_pipeline = ImagePipeline(os.path.join(self.dataset_dir, self.datasetA),
                                   batch_size=self.GLOBAL_BATCH_SIZE, image_size=self.image_size, shuffle=True)
        A_init_op, A_next_el = A_pipeline.get_init_op_and_next_el()
        B_pipeline = NpyPipeline(os.path.join(self.dataset_dir, self.datasetB),
                                 batch_size=self.GLOBAL_BATCH_SIZE, max_range=self.B_range, shuffle=True)
        B_init_op, B_next_el = B_pipeline.get_init_op_and_next_el()

        # get num of iteration
        max_iter = min(A_pipeline.get_file_num(), B_pipeline.get_file_num()) // self.GLOBAL_BATCH_SIZE

        # training loop
        counter = 0
        start_time = time.time()
        with tqdm(range(self.epoch)) as bar_epoch:
            for epoch in bar_epoch:
                bar_epoch.set_description('epoch')
                # lr decay
                if epoch < self.lr_decay_epoch:
                    lr = np.float32(self.lr)
                else:
                    lr = np.float32(self.lr) * (self.epoch - epoch) / (self.epoch - self.lr_decay_epoch)

                # initialize dataset iterator
                self.sess.run([A_init_op, B_init_op])

                with tqdm(range(max_iter), leave=False) as bar_iter:
                    for idx in bar_iter:
                        bar_iter.set_description('iteration')

                        # load data
                        A_image_paths, A_images = self.sess.run(A_next_el)
                        B_filenames, B_features = self.sess.run(B_next_el)

                        # update G
                        _, G_sum = self.sess.run([self.G_optim, self.G_total_loss_sum],
                                                 feed_dict={self.real_img_A: A_images,
                                                            self.real_feat_B: B_features,
                                                            self.lr_ph: lr})
                        self.writer.add_summary(G_sum, counter)

                        # update D
                        _, D_sum = self.sess.run([self.D_optim, self.D_total_loss_sum],
                                                 feed_dict={self.real_img_A: A_images,
                                                            self.real_feat_B: B_features,
                                                            self.lr_ph: lr})
                        self.writer.add_summary(D_sum, counter)

                        counter += 1

                # save samples
                if (epoch+1) % (self.epoch // 10) == 0:
                    self.save_samples(epoch)

        # save model when finish training
        self.save(self.checkpoint_dir, counter)
        self.save_config()
        print('the total time is %4.4f' % (time.time() - start_time))

    def test(self):
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # load dataset
        A_pipeline = ImagePipeline(os.path.join(self.dataset_dir, self.datasetA),
                                   batch_size=self.GLOBAL_BATCH_SIZE, image_size=self.image_size, shuffle=False)
        A_init_op, A_next_el = A_pipeline.get_init_op_and_next_el()

        # get num of iteration
        max_iter = A_pipeline.get_file_num() // self.GLOBAL_BATCH_SIZE

        # initialize dataset iterator
        self.sess.run(A_init_op)

        with tqdm(range(max_iter)) as bar_iter:
            for idx in bar_iter:
                # load data
                A_image_paths, A_images = self.sess.run(A_next_el)

                fake_B = self.sess.run(self.sample_feat_B, feed_dict={self.real_img_A: A_images})



    def save(self, checkpoint_dir, counter):
        model_name = "{}.ckpt".format(self.datasetB + '2' + self.datasetA)

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

    def save_config(self):
        save_dict = {
            "gpu_num": self.gpu_num,
            "GLOBAL_BATCH_SIZE": self.GLOBAL_BATCH_SIZE,
            "epoch": self.epoch,
            "lr": self.lr,
            "hidden_dim": self.hidden_dim,
            "L1_lambda": self.L1_lambda,
            "beta1": self.beta1,
            "image_size": self.image_size,
            "B_range": self.B_range,
            "datasetA": self.datasetA,
            "datasetB": self.datasetB
        }
        json.dump(save_dict, open(os.path.join(self.result_dir, 'config.json'), 'w'))

    def save_samples(self, epoch):
        num_sample = len(self.sample_A)
        max_iter = num_sample // self.GLOBAL_BATCH_SIZE
        fake_A_list = []
        fake_B_list = []
        for idx in range(max_iter):
            A = self.sample_A[idx*self.GLOBAL_BATCH_SIZE:(idx+1)*self.GLOBAL_BATCH_SIZE]
            B = self.sample_B[idx*self.GLOBAL_BATCH_SIZE:(idx+1)*self.GLOBAL_BATCH_SIZE]
            fake_A, fake_B = self.sess.run([self.sample_feat_A, self.sample_feat_B],
                                           feed_dict={self.real_img_A: A, self.real_feat_B: B})
            fake_A_list.append(fake_A[0])
            fake_B_list.append(fake_B[0])
        save_sample_npy(npy_array=fake_A_list, max_range=self.B_range,
                        output_dir=os.path.join(self.sample_dir, "%04d" % epoch, 'fake_B2A'))
        save_sample_npy(npy_array=fake_B_list, max_range=self.B_range,
                        output_dir=os.path.join(self.sample_dir, "%04d" % epoch, 'fake_A2B'))

    def get_sample_data(self):
        A_pipeline = ImagePipeline(os.path.join(self.dataset_dir, self.datasetA),
                                   batch_size=1, image_size=self.image_size, shuffle=False)
        A_init_op, A_next_el = A_pipeline.get_init_op_and_next_el()
        B_pipeline = NpyPipeline(os.path.join(self.dataset_dir, self.datasetB),
                                 batch_size=1, max_range=self.B_range, shuffle=False)
        B_init_op, B_next_el = B_pipeline.get_init_op_and_next_el()
        self.sess.run([A_init_op, B_init_op])

        num_sample = 0
        while num_sample < 9:
            num_sample += self.GLOBAL_BATCH_SIZE

        sample_A = []
        for idx in range(num_sample):
            _, imageA = self.sess.run(A_next_el)
            sample_A.append(imageA[0])

        sample_B = []
        for idx in range(num_sample):
            _, featureB = self.sess.run(B_next_el)
            sample_B.append(featureB[0])

        del A_pipeline
        del B_pipeline
        save_sample_image(image_array=np.array(sample_A, dtype=np.float32),
                          output_dir=os.path.join(self.sample_dir, "base", 'real_A'))
        save_sample_npy(npy_array=sample_B, max_range=self.B_range,
                        output_dir=os.path.join(self.sample_dir, "base", 'real_B'))
        return sample_A, sample_B
