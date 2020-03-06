import argparse
import os
from glob import glob

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from detection_src import FeatureExtractor, AnchorGenerator, Detector, makedirs


class Model(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.ckpt_for_detector = './ckpt_for_detector'
        self.input_image_dir = args.input_dir
        self.output_prediction_dir = args.output_prediction_dir
        makedirs(self.output_prediction_dir)
        self.class_name = args.class_name

        self._build_model()

    def _build_model(self):
        self.input_images = tf.placeholder(tf.float32, [1, None, None, 3], name='input_images')
        resized_images = tf.image.resize(self.input_images, [1024, 1024])

        with tf.variable_scope('student'):
            self.feature_extractor_fn = FeatureExtractor(is_training=False)
            self.anchor_generator_fn = AnchorGenerator()
            self.detector = Detector(resized_images, self.feature_extractor_fn, self.anchor_generator_fn)

            with tf.name_scope('student_prediction'):
                self.prediction = self.detector.get_predictions(
                    score_threshold=0.7,
                    iou_threshold=0.5,
                    max_boxes=200
                )

        self.s_vars = tf.global_variables(scope='student')
        for var in self.s_vars:
            print(var.name)

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

    def get_predictions(self):
        if self.load_detector():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        image_path_list = glob(os.path.join(self.input_image_dir, '*.jpg'))
        for image_path in tqdm(image_path_list):
            filename = image_path.split('/')[-1]

            # load image
            pil_image = Image.open(image_path, mode='r')
            image = np.expand_dims(np.asarray(pil_image, dtype=np.float32), axis=0)
            assert image.shape[0] == 1 and image.shape[3] == 3
            img_h, img_w = image.shape[1:3]

            # input_image = np.zeros((1, 1024, 1024, 3), dtype=np.float32)
            # input_image[0,:img_h,:img_w,:] = image

            predictions = self.sess.run(self.prediction, feed_dict={self.input_images: image})

            # extract prediction
            num_boxes = predictions['num_boxes'][0]
            boxes = predictions['boxes'][0][:num_boxes]
            scores = predictions['scores'][0][:num_boxes]

            scaler = np.array([img_h, img_w, img_h, img_w], dtype=np.float32)
            # scaler = np.array([1024, 1024, 1024, 1024], dtype=np.float32)
            boxes = boxes * scaler

            # output prediction
            pred_filename = filename.replace('.jpg', '.txt')
            with open(os.path.join(self.output_prediction_dir, pred_filename), 'w') as fout:
                for score, box in zip(scores, boxes):
                    top, left, bottom, right = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    if bottom > img_h or right > img_w:
                        continue
                    top = max(0, top)
                    left = max(0, left)
                    bottom = min(img_h, bottom)
                    right = min(img_w, right)
                    fout.write('%s %f %d %d %d %d\n' % (self.class_name, float(score), left, top, right, bottom))


if __name__ == '__main__':
    tf.set_random_seed(19)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_dir', dest='input_dir')
    parser.add_argument('--output_prediction_dir', dest='output_prediction_dir')
    parser.add_argument('--class_name', dest='class_name', type=str, default='face')
    args = parser.parse_args()

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        model = Model(sess, args)
        model.get_predictions()
