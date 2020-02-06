import tensorflow as tf
from tensorflow.contrib import slim

from ops import deconv2d, instance_norm, lrelu, conv2d, residule_block


def FeatureExtractor(input, reuse, name='feature_extractor'):
    # rapidly digested convolutional layers
    params = {
        'padding': 'SAME',
        'activation_fn': lambda x: tf.nn.crelu(x, axis=3),
        'normalizer_fn': instance_norm, 'data_format': 'NHWC'
    }
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        with slim.arg_scope([slim.conv2d], **params):
            with slim.arg_scope([slim.max_pool2d], stride=2, padding='SAME', data_format='NHWC'):
                x = slim.conv2d(input, 24, (7, 7), stride=2, scope='conv1')
                x = slim.max_pool2d(x, (3, 3), scope='pool1')
                x = slim.conv2d(x, 64, (5, 5), stride=2, scope='conv2')
                x = slim.max_pool2d(x, (3, 3), scope='pool2')
    # output shape == (32 x 32 x 128)
    x = tf.nn.tanh(x)
    return tf.identity(x, name=name + '_output')


class generator_resnet:
    def __init__(self, name="generator"):
        self.name = name

    def __call__(self, input, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            act = tf.nn.relu
            _, h, w, c = input.shape.as_list()

            # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
            # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
            # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
            # input shape == (32 x 32 x 128)
            c1 = act(instance_norm(conv2d(input, c, 7, 1, name='g_e1_c'), 'g_e1_bn'))
            # c1 shape == (32 x 32 x 128)
            c2 = act(instance_norm(conv2d(c1, c * 2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
            # c2 shape == (16 x 16 x 256)
            c3 = act(instance_norm(conv2d(c2, c * 4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
            # c3 shape == (8 x 8 x 512)

            # define G network with 9 resnet blocks
            r1 = residule_block(c3, c * 4, name='g_r1')
            r2 = residule_block(r1, c * 4, name='g_r2')
            r3 = residule_block(r2, c * 4, name='g_r3')
            r4 = residule_block(r3, c * 4, name='g_r4')
            r5 = residule_block(r4, c * 4, name='g_r5')
            r6 = residule_block(r5, c * 4, name='g_r6')
            r7 = residule_block(r6, c * 4, name='g_r7')
            r8 = residule_block(r7, c * 4, name='g_r8')
            r9 = residule_block(r8, c * 4, name='g_r9')

            d1 = act(instance_norm(deconv2d(r9, c * 2, 3, 2, name='g_d1_dc'), 'g_d1_bn'))
            # d1 shape == (16 x 16 x 256)
            d2 = act(instance_norm(deconv2d(d1, c, 3, 2, name='g_d2_dc'), 'g_d2_bn'))
            # d1 shape == (32 x 32 x 128)
            output = tf.nn.tanh(deconv2d(d2, c, 7, 1, name='g_output_dc'))
            # output shape == (32, 32, 128)

            return tf.identity(output, name=self.name + '_output')


class discriminator:
    def __init__(self, name="discriminator", hidden_dim=64):
        self.name = name
        self.hidden_dim = hidden_dim

    def __call__(self, feature, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            h0 = lrelu(conv2d(feature, 64, name='d_h0_conv'))
            # h0 is (16 x 16 x 64)
            h1 = lrelu(instance_norm(conv2d(h0, self.hidden_dim * 2, name='d_h1_conv'), 'd_bn1'))
            # h1 is (8 x 8 x self.hidden_dim*2)
            h2 = lrelu(instance_norm(conv2d(h1, self.hidden_dim * 4, name='d_h2_conv'), 'd_bn2'))
            # h2 is (4 x 4 x self.hidden_dim*4)
            h3 = lrelu(instance_norm(conv2d(h2, self.hidden_dim * 8, s=1, name='d_h3_conv'), 'd_bn3'))
            # h3 is (4 x 4 x self.hidden_dim*8)
            h4 = tf.nn.sigmoid(conv2d(h3, 1, s=1, name='d_h3_pred'))
            # h4 is (4 x 4 x 1)
        return tf.identity(h4, name=self.name+'_output')


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_ - target) ** 2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
