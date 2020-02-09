import tensorflow as tf
import tensorflow.contrib.slim as slim
from detection_src.constants import BATCH_NORM_MOMENTUM


class FeatureExtractor:
    def __init__(self, is_training):
        self.is_training = is_training

    def __call__(self, image):
        # rapidly digested convolutional layers
        params = {
            'padding': 'SAME',
            'activation_fn': lambda x: tf.nn.crelu(x, axis=3),
            'normalizer_fn': self.batch_norm, 'data_format': 'NHWC'
        }
        with slim.arg_scope([slim.conv2d], **params):
            with slim.arg_scope([slim.max_pool2d], stride=2, padding='SAME', data_format='NHWC'):
                x = slim.conv2d(image, 24, (7, 7), stride=4, scope='conv1')
                x = slim.max_pool2d(x, (3, 3), scope='pool1')
                x = slim.conv2d(x, 64, (5, 5), stride=2, scope='conv2')
                x = slim.max_pool2d(x, (3, 3), scope='pool2')

        # multiple scale convolutional layers
        params = {
            'padding': 'SAME', 'activation_fn': tf.nn.relu,
            'normalizer_fn': self.batch_norm, 'data_format': 'NHWC'
        }
        with slim.arg_scope([slim.conv2d], **params):
            features = []  # extracted feature maps
            x = inception_module(x, scope='inception1')
            x = inception_module(x, scope='inception2')
            x = inception_module(x, scope='inception3')
            features.append(x)  # scale 0
            x = slim.conv2d(x, 128, (1, 1), scope='conv3_1')
            x = slim.conv2d(x, 256, (3, 3), stride=2, scope='conv3_2')
            features.append(x)  # scale 1
            x = slim.conv2d(x, 128, (1, 1), scope='conv4_1')
            x = slim.conv2d(x, 256, (3, 3), stride=2, scope='conv4_2')
            features.append(x)  # scale 2

        return features

    def batch_norm(self, x):
        x = tf.layers.batch_normalization(
            x, axis=3, center=True, scale=True,
            momentum=BATCH_NORM_MOMENTUM, epsilon=0.001,
            training=self.is_training, fused=True,
            name='batch_norm'
        )
        return x


def preprocess(images):
    """Transform images before feeding them to the network."""
    return (2.0 * images) - 1.0


def inception_module(x, scope):
    # path 1
    x1 = slim.conv2d(x, 32, (1, 1), scope=scope + '/conv_1x1_path1')
    # path 2
    y = slim.max_pool2d(x, (3, 3), stride=1, padding='SAME', scope=scope + '/pool_3x3_path2')
    x2 = slim.conv2d(y, 32, (1, 1), scope=scope + '/conv_1x1_path2')
    # path 3
    y = slim.conv2d(x, 24, (1, 1), scope=scope + '/conv_1x1_path3')
    x3 = slim.conv2d(y, 32, (3, 3), scope=scope + '/conv_3x3_path3')
    # path 4
    y = slim.conv2d(x, 24, (1, 1), scope=scope + '/conv_1x1_path4')
    y = slim.conv2d(y, 32, (3, 3), scope=scope + '/conv_3x3_path4')
    x4 = slim.conv2d(y, 32, (3, 3), scope=scope + '/conv_3x3_second_path4')
    return tf.concat([x1, x2, x3, x4], axis=3, name=scope + '/concat')
