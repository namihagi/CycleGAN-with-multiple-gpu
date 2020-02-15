import json
from glob import glob

import cv2
from tqdm import tqdm

from detection_src import makedirs, Detector, AnchorGenerator, FeatureExtractor
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
        self.load_size = args.load_size
        self.fine_size = args.fine_size
        self.output_c_dim = args.output_c_dim
        self.input_c_dim = args.input_c_dim
        self.cons_lambda = args.cons_lambda
        self.dete_lambda = args.dete_lambda
        self.beta1 = args.beta1

        # network setting
        self.G_A2B = generator_resnet_256(hidden_dim=self.hidden_dim,
                                          output_c_dim=self.output_c_dim, name='generatorA2B')
        self.G_B2A = generator_resnet_256(hidden_dim=self.hidden_dim,
                                          output_c_dim=self.output_c_dim, name='generatorB2A')
        self.D_A = discriminator_256(hidden_dim=64, name='discriminatorA')
        self.D_B = discriminator_256(hidden_dim=64, name='discriminatorB')
        # self.G_A2B = generator_resnet(hidden_dim=self.hidden_dim, output_c_dim=self.output_c_dim, name='generatorA2B')
        # self.G_B2A = generator_resnet(hidden_dim=self.hidden_dim, output_c_dim=self.output_c_dim, name='generatorB2A')
        # self.D_A = discriminator(hidden_dim=64, name='discriminatorA')
        # self.D_B = discriminator(hidden_dim=64, name='discriminatorB')
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
        self.sample_dir = os.path.join('./result', args.sub_dir, 'sample')
        self.log_dir = os.path.join('./log', args.sub_dir)
        self.checkpoint_dir = os.path.join('./checkpoint', args.sub_dir)
        for dir_path in [self.result_dir, self.sample_dir, self.log_dir, self.checkpoint_dir]:
            makedirs(dir_path=dir_path)

        # initialize graph
        if args.phase == 'train':
            self._train_model()
            self.A_test_init_op, self.A_test_next_el, self.A_test_file_num = \
                self.get_input_fn(self.test_A_path, is_training=False)
            self.B_test_init_op, self.B_test_next_el, self.B_test_file_num = \
                self.get_input_fn(self.test_B_path, is_training=False)
            # get num of iteration
            self.test_max_iter = min(self.A_test_file_num, self.B_test_file_num, 10) // self.batch_size
        elif args.phase == 'input_directly':
            self._input_directly_model()
        elif args.phase == 'test_only_detection':
            self._detection_only_model()
        else:
            self._test_model()
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        # initialize variables
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def _train_model(self):
        # placeholder
        self.real_A = tf.placeholder(tf.float32,
                                     [self.batch_size, self.fine_size, self.fine_size, self.input_c_dim],
                                     name="real_A")
        self.real_B = tf.placeholder(tf.float32,
                                     [self.batch_size, self.fine_size, self.fine_size, self.input_c_dim],
                                     name="real_B")
        self.B_boxes = tf.placeholder(tf.float32, [self.batch_size, None, 4], name="boxes_B")
        self.B_num_boxes = tf.placeholder(tf.int32, [self.batch_size], name="num_boxes_B")

        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [self.batch_size, self.fine_size, self.fine_size, self.input_c_dim],
                                            name="fake_A_sample")
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [self.batch_size, self.fine_size, self.fine_size, self.input_c_dim],
                                            name="fake_B_sample")

        with tf.device('/gpu:0'):
            # ---- network to update generator ------
            # A > B > A
            self.fake_B = self.G_A2B(self.real_A, reuse=False)
            self.fake_A_return = self.G_B2A(self.fake_B, reuse=False)
            # B > A > B
            self.fake_A = self.G_B2A(self.real_B, reuse=True)
            self.fake_B_return = self.G_A2B(self.fake_A, reuse=True)

            # discriminator
            self.DA_fake_A_score = self.D_A(self.fake_A, reuse=False)
            self.DB_fake_B_score = self.D_B(self.fake_B, reuse=False)

            # # detection model
            # with tf.variable_scope('student'):
            #     self.feature_extractor_fn = FeatureExtractor(is_training=False)
            #     self.anchor_generator_fn = AnchorGenerator()
            #     upsampling_fake_B_return = tf.keras.layers.UpSampling2D()(self.fake_B_return)
            #     self.detector = Detector(upsampling_fake_B_return,
            #                              self.feature_extractor_fn, self.anchor_generator_fn)
            #
            #     with tf.name_scope('student_supervised_loss'):
            #         labels = {'boxes': self.B_boxes, 'num_boxes': self.B_num_boxes}
            #         losses = self.detector.loss(labels, self.detector_params)
            #         self.localization_loss = \
            #             self.detector_params['localization_loss_weight'] * losses['localization_loss']
            #         self.classification_loss = \
            #             self.detector_params['classification_loss_weight'] * losses['classification_loss']
            #         self.detector_loss = self.localization_loss + self.classification_loss

            # loss for generator
            self.loss_A_cyc = abs_criterion(self.real_A, self.fake_A_return)
            self.loss_B_cyc = abs_criterion(self.real_B, self.fake_B_return)
            self.G_loss = self.criterionGAN(self.DA_fake_A_score, tf.ones_like(self.DA_fake_A_score)) \
                          + self.criterionGAN(self.DB_fake_B_score, tf.ones_like(self.DB_fake_B_score)) \
                          + self.cons_lambda * (self.loss_A_cyc + self.loss_B_cyc)
            # ---- network to update generator ------

            # ---- network to update discriminator ------
            self.DA_real_A_score = self.D_A(self.real_A, reuse=True)
            self.DB_real_B_score = self.D_B(self.real_B, reuse=True)
            self.DA_fake_sample_score = self.D_A(self.fake_A_sample, reuse=True)
            self.DB_fake_sample_score = self.D_B(self.fake_B_sample, reuse=True)

            # loss for D_A
            self.DA_fake_A_loss = self.criterionGAN(self.DA_fake_sample_score, tf.zeros_like(self.DA_fake_A_score))
            self.DA_real_A_loss = self.criterionGAN(self.DA_real_A_score, tf.ones_like(self.DA_real_A_score))
            self.DA_loss = (self.DA_fake_A_loss + self.DA_real_A_loss) / 2
            # loss for D_B
            self.DB_fake_B_loss = self.criterionGAN(self.DB_fake_sample_score, tf.zeros_like(self.DB_fake_B_score))
            self.DB_real_B_loss = self.criterionGAN(self.DB_real_B_score, tf.ones_like(self.DB_real_B_score))
            self.DB_loss = (self.DB_fake_B_loss + self.DB_real_B_loss) / 2
            # all discriminator loss
            self.D_loss = self.DA_loss + self.DB_loss

        # summary for loss
        self.loss_A_cyc_sum = tf.summary.scalar("loss_A_cyc", self.loss_A_cyc)
        self.loss_B_cyc_sum = tf.summary.scalar("loss_B_cyc", self.loss_B_cyc)
        # self.detector_loss_sum = tf.summary.scalar("detector_loss", self.detector_loss)
        self.G_loss_sum = tf.summary.scalar("G_loss", self.G_loss)
        self.G_sum = tf.summary.merge([self.loss_A_cyc_sum, self.loss_B_cyc_sum, self.G_loss_sum])
        self.DA_loss_sum = tf.summary.scalar("DA_loss", self.DA_loss)
        self.DB_loss_sum = tf.summary.scalar("DB_loss", self.DB_loss)
        self.D_loss_sum = tf.summary.scalar("D_loss", self.D_loss)
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
        # print('----- s_vars -----')
        # for var in self.s_vars:
        #     print(var.name)
        # print('----- s_vars -----')

        # optimizer
        self.lr_ph = tf.placeholder(tf.float32, name='learning_rate')
        self.D_optim = tf.train.AdamOptimizer(self.lr_ph, beta1=self.beta1).minimize(self.D_loss, var_list=self.d_vars)
        self.G_optim = tf.train.AdamOptimizer(self.lr_ph, beta1=self.beta1).minimize(self.G_loss, var_list=self.g_vars)
        # self.G_detector_optim = tf.train.AdamOptimizer(self.lr_ph, beta1=self.beta1) \
        #     .minimize(self.detector_loss, var_list=self.g_vars)

    def _test_model(self):
        # placeholder
        self.real_A = tf.placeholder(tf.float32, [self.batch_size, self.fine_size, self.fine_size, self.input_c_dim],
                                     name="real_A")
        self.real_B = tf.placeholder(tf.float32, [self.batch_size, self.fine_size, self.fine_size, self.input_c_dim],
                                     name="real_B")
        # convert feature
        self.fake_B = self.G_A2B(self.real_A, reuse=False)
        self.fake_A = self.G_B2A(self.real_B, reuse=False)

        # zero padding
        fake_B_padding = tf.image.pad_to_bounding_box(
            self.fake_B, offset_height=0, offset_width=0,
            target_height=1024, target_width=1024
        )

        with tf.variable_scope('student'):
            self.feature_extractor_fn = FeatureExtractor(is_training=False)
            self.anchor_generator_fn = AnchorGenerator()
            self.detector = Detector(fake_B_padding, self.feature_extractor_fn, self.anchor_generator_fn)

            with tf.name_scope('student_prediction'):
                self.prediction = self.detector.get_predictions(
                    score_threshold=0.5,
                    iou_threshold=0.5,
                    max_boxes=200
                )

        # get variables
        self.t_vars = tf.trainable_variables()
        self.g_vars = [var for var in self.t_vars if 'generator' in var.name]
        self.s_vars = tf.global_variables(scope='student')

    def _input_directly_model(self):
        self.real_A = tf.placeholder(tf.float32, [self.batch_size, self.fine_size, self.fine_size, self.input_c_dim])

        self.fake_B = self.G_A2B(self.real_A, reuse=False)

        # zero padding
        fake_B_padding = tf.image.pad_to_bounding_box(
            self.fake_B, offset_height=0, offset_width=0,
            target_height=1024, target_width=1024
        )

        # detection model
        with tf.variable_scope('student'):
            self.feature_extractor_fn = FeatureExtractor(is_training=False)
            self.anchor_generator_fn = AnchorGenerator()
            self.detector = Detector(fake_B_padding, self.feature_extractor_fn, self.anchor_generator_fn)

            with tf.name_scope('student_prediction'):
                self.prediction = self.detector.get_predictions(
                    score_threshold=0.5,
                    iou_threshold=0.5,
                    max_boxes=200
                )

        # get variables
        self.t_vars = tf.trainable_variables()
        self.g_vars = [var for var in self.t_vars if 'generator' in var.name]
        self.s_vars = tf.global_variables(scope='student')

    def _detection_only_model(self):
        self.real_A = tf.placeholder(tf.float32, [self.batch_size, self.fine_size, self.fine_size, self.input_c_dim])

        # zero padding
        real_A_padding = tf.image.pad_to_bounding_box(
            self.real_A, offset_height=0, offset_width=0,
            target_height=1024, target_width=1024
        )

        # detection model
        with tf.variable_scope('student'):
            self.feature_extractor_fn = FeatureExtractor(is_training=False)
            self.anchor_generator_fn = AnchorGenerator()
            self.detector = Detector(real_A_padding, self.feature_extractor_fn, self.anchor_generator_fn)

            with tf.name_scope('student_prediction'):
                self.prediction = self.detector.get_predictions(
                    score_threshold=0.8,
                    iou_threshold=0.8,
                    max_boxes=200
                )

        # get variables
        self.s_vars = tf.global_variables(scope='student')

    def train(self):
        # if self.load_detector():
        #     print(" [*] Load SUCCESS")
        # else:
        #     print(" [!] Load failed...")

        A_init_op, A_next_el, A_file_num = self.get_input_fn(self.train_A_path, is_training=True)
        B_init_op, B_next_el, B_file_num = self.get_input_fn(self.train_B_path, is_training=True)

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
                        fake_A_sample, fake_B_sample, _, G_sum = \
                            self.sess.run([self.fake_A, self.fake_B, self.G_optim, self.G_sum],
                                          feed_dict={self.real_A: A_image, self.real_B: B_image, self.lr_ph: lr})
                        # self.B_num_boxes: B_num_boxes,
                        self.writer.add_summary(G_sum, counter)

                        # # update G by detector loss
                        # if epoch >= self.epoch // 3:
                        #     _, detector_loss_sum = self.sess.run([self.G_detector_optim, self.detector_loss_sum],
                        #                                          feed_dict={self.real_B: B_image,
                        #                                                     self.B_boxes: B_boxes,
                        #                                                     self.B_num_boxes: B_num_boxes,
                        #                                                     self.lr_ph: lr})
                        #     self.writer.add_summary(detector_loss_sum, counter)

                        # update D
                        _, D_sum = self.sess.run([self.D_optim, self.D_sum],
                                                 feed_dict={self.real_A: A_image,
                                                            self.real_B: B_image,
                                                            self.fake_A_sample: fake_A_sample,
                                                            self.fake_B_sample: fake_B_sample,
                                                            self.lr_ph: lr})
                        self.writer.add_summary(D_sum, counter)
                        counter += 1

                if (epoch + 1) % 20 == 0:
                    self.save_sample(epoch + 1)

        # save model when finish training
        self.save()
        self.save_config()

    def save_sample(self, epoch):
        output_dir = os.path.join(self.sample_dir, str(epoch))
        makedirs(output_dir)
        # initialize dataset iterator
        self.sess.run([self.A_test_init_op, self.B_test_init_op])

        for idx in range(self.test_max_iter):
            # load data
            A_image, A_img_shape, A_boxes, A_num_boxes, A_filename = self.sess.run(self.A_test_next_el)
            B_image, B_img_shape, B_boxes, B_num_boxes, B_filename = self.sess.run(self.B_test_next_el)
            fake_A, fake_B = self.sess.run([self.fake_A, self.fake_B],
                                           feed_dict={self.real_A: A_image, self.real_B: B_image})

            fake_A = (fake_A + 1.0) / 2.0
            fake_A = (fake_A * 255).astype(np.uint8)
            fake_B = (fake_B + 1.0) / 2.0
            fake_B = (fake_B * 255).astype(np.uint8)

            A_filename = A_filename[0].decode()
            B_filename = B_filename[0].decode()

            fake_A_size = B_img_shape[0].tolist()
            save_fake_A = Image.fromarray(fake_A[0], mode='RGB')
            save_fake_A = save_fake_A.resize((int(fake_A_size[1]), int(fake_A_size[0])))
            save_fake_A.save(os.path.join(output_dir, 'B2A_{}'.format(B_filename)))

            fake_B_size = A_img_shape[0].tolist()
            save_fake_B = Image.fromarray(fake_B[0], mode='RGB')
            save_fake_B = save_fake_B.resize((int(fake_B_size[1]), int(fake_B_size[0])))
            save_fake_B.save(os.path.join(output_dir, 'A2B_{}'.format(A_filename)))

    def test(self):
        if self.load(var_list=self.g_vars):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        if self.load_detector():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        direction = {
            self.train_A_path: {
                'image_dir': 'train/A2B',
                'prediction_dir': 'train/A2B_predictions',
                'placeholder': self.real_A,
                'output': self.fake_B
            },
            self.train_B_path: {
                'image_dir': 'train/B2A',
                'placeholder': self.real_B,
                'output': self.fake_A
            },
            self.test_A_path: {
                'image_dir': 'test/A2B',
                'prediction_dir': 'test/A2B_predictions',
                'placeholder': self.real_A,
                'output': self.fake_B
            },
            self.test_B_path: {
                'image_dir': 'test/B2A',
                'placeholder': self.real_B,
                'output': self.fake_A
            }
        }

        for target_dir in direction.keys():
            setting = direction[target_dir]
            # load dataset
            init_op, next_el, file_num = self.get_input_fn(str(target_dir), is_training=self.is_training)
            self.sess.run([init_op])

            for idx in tqdm(range(file_num)):
                # load data
                image, img_shape, boxes, num_boxes, filename = self.sess.run(next_el)
                img_shape = img_shape[0].tolist()
                filename = filename[0].decode()
                if 'prediction_dir' in setting.keys():
                    if idx == 0:
                        output_prediction_dir = os.path.join(self.result_dir, setting['prediction_dir'])
                        makedirs(output_prediction_dir)
                    output, prediction = self.sess.run([setting['output'], self.prediction],
                                                       feed_dict={setting['placeholder']: image})
                    # extract prediction
                    num_boxes = prediction['num_boxes'][0]
                    boxes = prediction['boxes'][0][:num_boxes]
                    scores = prediction['scores'][0][:num_boxes]

                    scaler = np.array([1024, 1024, 1024, 1024], dtype='float32')
                    boxes = boxes * scaler

                    # output prediction
                    json_filename = filename.replace('.jpg', '.txt')
                    h, w, c = img_shape
                    with open(os.path.join(output_prediction_dir, json_filename), 'w') as fout:
                        for score, box in zip(scores, boxes):
                            left, top, right, bottom = float(box[1]), float(box[0]), float(box[3]), float(box[2])
                            if right > 256 or top > 256:
                                continue
                            fout.write('%s %f %d %d %d %d\n'
                                       % (self.class_name, float(score), int(left), int(top), int(right), int(bottom)))
                else:
                    output = self.sess.run(setting['output'],
                                           feed_dict={setting['placeholder']: image})

                output = (output + 1.0) / 2.0
                output = (output * 255).astype(np.uint8)

                output_image_dir = os.path.join(self.result_dir, setting['image_dir'])
                makedirs(output_image_dir)
                save_image = Image.fromarray(output[0], mode='RGB')
                # save_image = save_image.resize((int(img_shape[1]), int(img_shape[0])))
                save_image.save(os.path.join(output_image_dir, filename))

    def test_input_directly(self, args):
        if self.load(var_list=self.g_vars):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        if self.load_detector():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        makedirs(args.output_image_dir)
        makedirs(args.output_prediction_dir)

        image_paths = glob(os.path.join(args.input_dir, '*.jpg'))
        with tqdm(image_paths) as bar_iter:
            for file_path in bar_iter:
                filename = file_path.split('/')[-1]
                # load image
                image_array = cv2.imread(file_path)
                h, w, c = image_array.shape
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                image_array = cv2.resize(image_array, (self.fine_size, self.fine_size))
                assert image_array.shape == (256, 256, 3)
                image = image_array.astype(np.float32) / 255.0
                # now pixel values are scaled to [0, 1] range
                image = (image * 2.0) - 1.0
                # now pixel values are scaled to [-1, 1] range
                image = np.expand_dims(image, 0)
                # prediction
                output, predictions = self.sess.run([self.fake_B, self.prediction], feed_dict={self.real_A: image})
                # extract prediction
                num_boxes = predictions['num_boxes'][0]
                boxes = predictions['boxes'][0][:num_boxes]
                scores = predictions['scores'][0][:num_boxes]

                scaler = np.array([1024, 1024, 1024, 1024], dtype='float32')
                boxes = boxes * scaler

                # output prediction
                json_filename = filename.replace('.jpg', '.txt')
                with open(os.path.join(args.output_prediction_dir, json_filename), 'w') as fout:
                    for score, box in zip(scores, boxes):
                        left, top, right, bottom = float(box[1]), float(box[0]), float(box[3]), float(box[2])
                        if right > 256 or top > 256:
                            continue
                        fout.write('%s %f %d %d %d %d\n'
                                   % (self.class_name, float(score), int(left), int(top), int(right), int(bottom)))

                # save image
                output = (output + 1.0) / 2.0
                output = (output * 255).astype(np.uint8)

                makedirs(args.output_image_dir)
                save_image = Image.fromarray(output[0], mode='RGB')
                save_image.save(os.path.join(args.output_image_dir, filename))

    def test_only_detection(self, args):
        if self.load_detector():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        makedirs(args.output_prediction_dir)

        image_paths = glob(os.path.join(args.input_dir, '*.jpg'))
        with tqdm(image_paths) as bar_iter:
            for file_path in bar_iter:
                filename = file_path.split('/')[-1]
                # load image
                image_array = cv2.imread(file_path)
                h, w, c = image_array.shape
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                image_array = cv2.resize(image_array, (self.fine_size, self.fine_size))
                assert image_array.shape == (256, 256, 3)
                image = image_array.astype(np.float32) / 255.0
                # now pixel values are scaled to [0, 1] range
                image = (image * 2.0) - 1.0
                # now pixel values are scaled to [-1, 1] range
                image = np.expand_dims(image, 0)
                # prediction
                predictions = self.sess.run(self.prediction, feed_dict={self.real_A: image})
                # extract prediction
                num_boxes = predictions['num_boxes'][0]
                boxes = predictions['boxes'][0][:num_boxes]
                scores = predictions['scores'][0][:num_boxes]

                scaler = np.array([1024, 1024, 1024, 1024], dtype='float32')
                boxes = boxes * scaler

                # output prediction
                json_filename = filename.replace('.jpg', '.txt')
                with open(os.path.join(args.output_prediction_dir, json_filename), 'w') as fout:
                    for score, box in zip(scores, boxes):
                        left, top, right, bottom = float(box[1]), float(box[0]), float(box[3]), float(box[2])
                        if right > self.fine_size or top > self.fine_size:
                            continue
                        left = max(int(left * (w / self.fine_size)), 0)
                        right = min(int(right * (w / self.fine_size)), w)
                        top = min(int(top * (h / self.fine_size)), h)
                        bottom = max(int(bottom * (h / self.fine_size)), 0)
                        if right > 256 or top > 256:
                            continue
                        fout.write('%s %f %d %d %d %d\n'
                                   % (self.class_name, float(score), int(left), int(top), int(right), int(bottom)))

    def save(self):
        model_name = "{}.ckpt".format(self.sub_dir)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, model_name))

    def load(self, var_list=None):
        print(" [*] Reading checkpoint...")
        if var_list is not None:
            saver = tf.train.Saver(var_list=var_list)
        else:
            saver = self.saver

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
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

    def get_input_fn(self, dataset_path, is_training=True):
        filenames = os.listdir(dataset_path)
        filenames = [n for n in filenames if n.endswith('.tfrecords')]
        filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

        with tf.device('/cpu:0'), tf.name_scope('input_pipeline'):
            pipeline = Pipeline(filenames, is_training=is_training, batch_size=self.batch_size,
                                load_size=self.load_size, fine_size=self.fine_size, shuffle=is_training)
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
            "load_size": self.load_size,
            "fine_size": self.fine_size,
            "train_A_path": self.train_A_path,
            "train_B_path": self.train_B_path,
            "test_A_path": self.test_A_path,
            "test_B_path": self.test_B_path
        }
        json.dump(save_dict, open(os.path.join(self.result_dir, 'config.json'), 'w'))
