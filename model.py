import json
import time
from glob import glob

from tqdm import tqdm

from detection_src import Detector, FeatureExtractor, AnchorGenerator, makedirs
from detection_src.input_pipeline import Pipeline
from detection_src.result_util import output_prediction_from_image
from module import *
from utils import *


class Model(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.is_training = (args.phase == 'train')
        self.class_name = args.class_name

        # model setting
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.lr_decay_epoch = args.lr_decay_epoch
        self.lr = args.lr
        self.hidden_dim = args.hidden_dim
        self.output_c_dim = args.output_c_dim
        self.cons_lambda = args.cons_lambda
        self.dete_lambda = args.dete_lambda
        self.beta1 = args.beta1
        self.B_range = args.B_range

        # network setting
        self.G_A2B = generator_resnet(hidden_dim=self.hidden_dim, output_c_dim=self.output_c_dim, name='generatorA2B')
        self.G_B2A = generator_resnet(hidden_dim=self.hidden_dim, output_c_dim=self.output_c_dim, name='generatorB2A')
        self.D_A = discriminator(hidden_dim=64, name='discriminatorA')
        self.D_B = discriminator(hidden_dim=64, name='discriminatorB')
        self.criterionGAN = mae_criterion
        self.detector_params = {
            "weight_decay": 1e-3,
            "score_threshold": 0.3, "iou_threshold": 0.3, "max_boxes": 200,
            "localization_loss_weight": 1.0, "classification_loss_weight": 1.0,
            "loss_to_use": "classification",
            "loc_loss_weight": 0.0, "cls_loss_weight": 1.0,
            "num_hard_examples": 500, "nms_threshold": 0.99,
            "max_negatives_per_positive": 3.0, "min_negatives_per_image": 30,
        }

        # directory setting
        self.train_A_path = args.train_A_path
        self.train_B_path = args.train_B_path
        self.test_A_path = args.test_A_path
        self.test_B_path = args.test_B_path
        self.sub_dir = args.sub_dir
        assert self.sub_dir is not None
        self.result_dir = os.path.join('./result', args.sub_dir)
        self.log_dir = os.path.join('./log', args.sub_dir)
        self.checkpoint_dir = os.path.join('./checkpoint', args.sub_dir)
        for dir_path in [self.result_dir, self.log_dir, self.checkpoint_dir]:
            makedirs(dir_path=dir_path)

        # initialize graph
        if self.is_training:
            self._train_model()
        else:
            self._test_model()
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        # initialize variables
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def _train_model(self):
        # placeholder
        self.real_A = tf.placeholder(tf.float32, [self.batch_size, 32, 32, 128], name="real_A")
        self.real_B = tf.placeholder(tf.float32, [self.batch_size, 32, 32, 128], name="real_B")
        self.B_boxes = tf.placeholder(tf.float32, [self.batch_size, None, 4], name="boxes_B")
        self.B_num_boxes = tf.placeholder(tf.int32, [self.batch_size], name="num_boxes_B")

        with tf.device('/gpu:0'):
            # generator
            # A > B > A
            self.fake_B = self.G_A2B(self.real_A, reuse=False)
            self.fake_A_return = self.G_B2A(self.fake_B, reuse=False)
            # B > A > B
            self.fake_A = self.G_B2A(self.real_B, reuse=True)
            self.fake_B_return = self.G_A2B(self.fake_A, reuse=True)

            # discriminator
            self.DA_fake_A_score = self.D_A(self.fake_A, reuse=False)
            self.DA_real_A_score = self.D_A(self.real_A, reuse=True)
            self.DB_fake_B_score = self.D_B(self.fake_B, reuse=False)
            self.DB_real_B_score = self.D_B(self.real_B, reuse=True)

            # detection model
            with tf.variable_scope('student'):
                self.feature_extractor_fn = FeatureExtractor(is_training=False)
                self.anchor_generator_fn = AnchorGenerator()
                self.inverse_fake_B_return = self._inverse_before_detector(self.fake_B_return)
                self.detector = Detector(self.inverse_fake_B_return, tf.zeros([self.batch_size, 1024, 1024, 3]),
                                         self.feature_extractor_fn, self.anchor_generator_fn)

                with tf.name_scope('student_supervised_loss'):
                    labels = {'boxes': self.B_boxes, 'num_boxes': self.B_num_boxes}
                    losses = self.detector.loss(labels, self.detector_params)
                    self.localization_loss = \
                        self.detector_params['localization_loss_weight'] * losses['localization_loss']
                    self.classification_loss = \
                        self.detector_params['classification_loss_weight'] * losses['classification_loss']
                    self.supervised_total_loss = self.localization_loss + self.classification_loss

            # loss for generator
            self.loss_A_cyc = abs_criterion(self.real_A, self.fake_A_return)
            self.loss_B_cyc = abs_criterion(self.real_B, self.fake_B_return)
            self.G_loss = self.criterionGAN(self.DA_fake_A_score, tf.ones_like(self.DA_fake_A_score)) \
                          + self.criterionGAN(self.DB_fake_B_score, tf.ones_like(self.DB_fake_B_score)) \
                          + self.cons_lambda * (self.loss_A_cyc + self.loss_B_cyc) \
                          + self.dete_lambda * self.supervised_total_loss

            # loss for D_A
            self.DA_fake_A_loss = self.criterionGAN(self.DA_fake_A_score, tf.zeros_like(self.DA_fake_A_score))
            self.DA_real_A_loss = self.criterionGAN(self.DA_real_A_score, tf.ones_like(self.DA_real_A_score))
            self.DA_loss = (self.DA_fake_A_loss + self.DA_real_A_loss) / 2
            # loss for D_B
            self.DB_fake_B_loss = self.criterionGAN(self.DB_fake_B_score, tf.zeros_like(self.DB_fake_B_score))
            self.DB_real_B_loss = self.criterionGAN(self.DB_real_B_score, tf.ones_like(self.DB_real_B_score))
            self.DB_loss = (self.DB_fake_B_loss + self.DB_real_B_loss) / 2
            # all discriminator loss
            self.D_loss = self.DA_loss + self.DB_loss

        # summary for loss
        self.loss_A_cyc_sum = tf.summary.scalar("loss_A_cyc", self.loss_A_cyc)
        self.loss_B_cyc_sum = tf.summary.scalar("loss_B_cyc", self.loss_B_cyc)
        self.supervised_total_loss_sum = tf.summary.scalar("supervised_total_loss", self.supervised_total_loss)
        self.G_loss_sum = tf.summary.scalar("generator_loss", self.G_loss)
        self.G_sum = tf.summary.merge([self.loss_A_cyc_sum, self.loss_B_cyc_sum,
                                       self.supervised_total_loss_sum, self.G_loss_sum])
        self.DA_loss_sum = tf.summary.scalar("DA_loss", self.DA_loss)
        self.DB_loss_sum = tf.summary.scalar("DB_loss", self.DB_loss)
        self.D_loss_sum = tf.summary.scalar("discriminator_loss", self.D_loss)
        self.D_sum = tf.summary.merge([self.DA_loss_sum, self.DB_loss_sum, self.D_loss_sum])

        # get variables
        self.t_vars = tf.trainable_variables()
        self.g_vars = [var for var in self.t_vars if 'generator' in var.name]
        self.d_vars = [var for var in self.t_vars if 'discriminator' in var.name]
        print('----- g_vars -----')
        for var in self.g_vars:
            print(var.name)
        print('----- g_vars -----')
        print('----- d_vars -----')
        for var in self.d_vars:
            print(var.name)
        print('----- d_vars -----')
        self.s_vars = tf.global_variables(scope='student')
        print('----- s_vars -----')
        for var in self.s_vars:
            print(var.name)
        print('----- s_vars -----')

        # optimizer
        self.lr_ph = tf.placeholder(tf.float32, name='learning_rate')
        self.D_optim = tf.train.AdamOptimizer(self.lr_ph, beta1=self.beta1).minimize(self.D_loss, var_list=self.d_vars)
        self.G_optim = tf.train.AdamOptimizer(self.lr_ph, beta1=self.beta1).minimize(self.G_loss, var_list=self.g_vars)

    def _test_model(self):
        # placeholder
        self.real_A = tf.placeholder(tf.float32, [self.batch_size, 32, 32, 128], name="real_A")
        # convert feature
        self.fake_B = self.G_A2B(self.real_A, reuse=False)
        # detection model
        with tf.variable_scope('student'):
            self.feature_extractor_fn = FeatureExtractor(is_training=False)
            self.anchor_generator_fn = AnchorGenerator()
            self.inverse_fake_B = self._inverse_before_detector(self.fake_B)
            self.detector = Detector(self.inverse_fake_B, tf.zeros([self.batch_size, 1024, 1024, 3]),
                                     self.feature_extractor_fn, self.anchor_generator_fn)

            with tf.name_scope('student_prediction'):
                self.prediction = self.detector.get_predictions(
                    score_threshold=0.5,
                    iou_threshold=0.5,
                    max_boxes=200
                )

    def _inverse_before_detector(self, feature):
        inverse_feature = (feature + 1.0) * (self.B_range / 2.0)
        return tf.clip_by_value(inverse_feature, clip_value_min=0.0, clip_value_max=self.B_range)

    def train(self):
        if self.load_detector():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        A_init_op, A_next_el, A_file_num = self.get_input_fn(self.train_A_path, is_training=True)
        B_init_op, B_next_el, B_file_num = self.get_input_fn(self.train_B_path,
                                                             max_range=self.B_range,  is_training=True)

        # get num of iteration
        max_iter = min(A_file_num, B_file_num) // self.batch_size

        # training loop
        counter = 0
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
                        A_image, A_img_shape, A_boxes, A_num_boxes, A_filename = self.sess.run(A_next_el)
                        B_image, B_img_shape, B_boxes, B_num_boxes, B_filename = self.sess.run(B_next_el)

                        # update G
                        _, G_sum = self.sess.run([self.G_optim, self.G_sum],
                                                 feed_dict={self.real_A: A_image,
                                                            self.real_B: B_image,
                                                            self.B_boxes: B_boxes,
                                                            self.B_num_boxes: B_num_boxes,
                                                            self.lr_ph: lr})
                        # self.B_num_boxes: B_num_boxes,
                        self.writer.add_summary(G_sum, counter)

                        # update D
                        _, D_sum = self.sess.run([self.D_optim, self.D_sum],
                                                 feed_dict={self.real_A: A_image,
                                                            self.real_B: B_image,
                                                            self.lr_ph: lr})
                        self.writer.add_summary(D_sum, counter)
                        counter += 1

        # save model when finish training
        self.save()
        self.save_config()

    def test(self):
        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # load dataset
        A_init_op, A_next_el, A_file_num = self.get_input_fn(self.test_A_path, is_training=self.is_training)

        # initialize dataset iterator
        self.sess.run(A_init_op)

        with tqdm(range(A_file_num)) as bar_iter:
            for idx in bar_iter:
                # load data
                A_image, A_img_shape, A_boxes, A_num_boxes, A_filename = self.sess.run(A_next_el)
                h, w, c = A_image[0].shape

                # prediction
                predictions = self.sess.run(self.prediction, feed_dict={self.real_A: A_image})
                # extract prediction
                num_boxes = predictions['num_boxes'][0]
                boxes = predictions['boxes'][0][:num_boxes]
                scores = predictions['scores'][0][:num_boxes]

                scaler = np.array([h, w, h, w], dtype='float32')
                boxes = boxes * scaler

                output_dir = os.path.join(self.result_dir, 'prediction')
                makedirs(output_dir)
                output_prediction_from_image(pred_scores=scores, pred_boxes=boxes, img_w=w, img_h=h,
                                             class_name=self.class_name, output_dir=output_dir,
                                             file_path=A_filename[0])

    def save(self):
        model_name = "{}.ckpt".format(self.sub_dir)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, model_name))

    def load(self):
        print(" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def load_detector(self):
        print(" [*] Reading detector checkpoint...")
        s_saver = tf.train.Saver(var_list=self.s_vars)

        checkpoint_dir = './ckpt_for_detector'
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            s_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def get_input_fn(self, dataset_path, is_training=True, max_range=None):
        filenames = os.listdir(dataset_path)
        filenames = [n for n in filenames if n.endswith('.tfrecords')]
        filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

        with tf.device('/cpu:0'), tf.name_scope('input_pipeline'):
            pipeline = Pipeline(filenames, batch_size=self.batch_size, max_range=max_range, shuffle=is_training)
        return pipeline.get_init_op_and_next_el()

    def save_config(self):
        save_dict = {
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "lr": self.lr,
            "lr_decay_epoch": self.lr_decay_epoch,
            "cons_lambda": self.cons_lambda,
            "dete_lambda": self.dete_lambda,
            "beta1": self.beta1,
            "hidden_dim": self.hidden_dim,
            "output_c_dim": self.output_c_dim,
            "B_range": self.B_range,
            "train_A_path": self.train_A_path,
            "train_B_path": self.train_B_path,
            "test_A_path": self.test_A_path,
            "test_B_path": self.test_B_path
        }
        json.dump(save_dict, open(os.path.join(self.result_dir, 'config.json'), 'w'))
