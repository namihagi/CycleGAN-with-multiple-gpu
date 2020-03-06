import tensorflow as tf
from tensorflow.contrib import slim

from ops import deconv2d, instance_norm, lrelu, conv2d, residule_block


class generator_resnet:
    def __init__(self, output_c_dim=128, hidden_dim=64, name="generator"):
        self.name = name
        self.output_c_dim = output_c_dim
        self.hidden_dim = hidden_dim

    def __call__(self, input, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
            # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
            # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
            c0 = tf.pad(input, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            # 1024
            c1 = tf.nn.relu(instance_norm(conv2d(c0, self.hidden_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
            # 1024
            c2 = tf.nn.relu(instance_norm(conv2d(c1, self.hidden_dim * 2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
            # 512
            c3 = tf.nn.relu(instance_norm(conv2d(c2, self.hidden_dim * 4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
            # 256
            c4 = tf.nn.relu(instance_norm(conv2d(c3, self.hidden_dim * 4, 3, 2, name='g_e4_c'), 'g_e4_bn'))
            # 128
            c5 = tf.nn.relu(instance_norm(conv2d(c4, self.hidden_dim * 4, 3, 2, name='g_e5_c'), 'g_e5_bn'))
            # 64
            # define G network with 9 resnet blocks
            r1 = residule_block(c5, self.hidden_dim * 4, name='g_r1')
            r2 = residule_block(r1, self.hidden_dim * 4, name='g_r2')
            r3 = residule_block(r2, self.hidden_dim * 4, name='g_r3')
            r4 = residule_block(r3, self.hidden_dim * 4, name='g_r4')
            r5 = residule_block(r4, self.hidden_dim * 4, name='g_r5')
            r6 = residule_block(r5, self.hidden_dim * 4, name='g_r6')
            r7 = residule_block(r6, self.hidden_dim * 4, name='g_r7')
            r8 = residule_block(r7, self.hidden_dim * 4, name='g_r8')
            r9 = residule_block(r8, self.hidden_dim * 4, name='g_r9')
            # 64

            d1 = deconv2d(r9, self.hidden_dim * 2, 3, 2, name='g_d1_dc')
            d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
            # 128
            d2 = deconv2d(d1, self.hidden_dim, 3, 2, name='g_d2_dc')
            d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
            # 256
            d3 = deconv2d(d2, self.hidden_dim, 3, 2, name='g_d3_dc')
            d3 = tf.nn.relu(instance_norm(d3, 'g_d3_bn'))
            # 512
            d4 = deconv2d(d3, self.hidden_dim, 3, 2, name='g_d4_dc')
            d4 = tf.nn.relu(instance_norm(d4, 'g_d4_bn'))
            # 1024
            d5 = tf.pad(d4, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            pred = tf.nn.tanh(conv2d(d5, self.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))
            # 1024

            return pred


class generator_resnet_256:
    def __init__(self, output_c_dim=128, hidden_dim=64, name="generator"):
        self.name = name
        self.output_c_dim = output_c_dim
        self.hidden_dim = hidden_dim

    def __call__(self, input, reuse=False):
        with tf.variable_scope(self.name):
            # image is 256 x 256 x input_c_dim
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
            # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
            # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
            c0 = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            c1 = tf.nn.relu(instance_norm(conv2d(c0, self.hidden_dim, 3, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
            c2 = tf.nn.relu(instance_norm(conv2d(c1, self.hidden_dim * 2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
            c3 = tf.nn.relu(instance_norm(conv2d(c2, self.hidden_dim * 4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
            # define G network with 9 resnet blocks
            r1 = residule_block(c3, self.hidden_dim * 4, name='g_r1')
            r2 = residule_block(r1, self.hidden_dim * 4, name='g_r2')
            r3 = residule_block(r2, self.hidden_dim * 4, name='g_r3')
            r4 = residule_block(r3, self.hidden_dim * 4, name='g_r4')
            r5 = residule_block(r4, self.hidden_dim * 4, name='g_r5')
            r6 = residule_block(r5, self.hidden_dim * 4, name='g_r6')
            r7 = residule_block(r6, self.hidden_dim * 4, name='g_r7')
            r8 = residule_block(r7, self.hidden_dim * 4, name='g_r8')
            r9 = residule_block(r8, self.hidden_dim * 4, name='g_r9')

            d1 = deconv2d(r9, self.hidden_dim * 2, 3, 2, name='g_d1_dc')
            d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
            d2 = deconv2d(d1, self.hidden_dim, 3, 2, name='g_d2_dc')
            d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
            d2 = tf.pad(d2, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            pred = tf.nn.tanh(conv2d(d2, self.output_c_dim, 3, 1, padding='VALID', name='g_pred_c'))

            return pred


class discriminator:
    def __init__(self, hidden_dim=64, name="discriminator"):
        self.name = name
        self.hidden_dim = hidden_dim

    def __call__(self, input, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            h0 = lrelu(conv2d(input, self.hidden_dim, name='d_h0_conv'))
            # 512
            h1 = lrelu(instance_norm(conv2d(h0, self.hidden_dim * 2, name='d_h1_conv'), 'd_bn1'))
            # 256
            h2 = lrelu(instance_norm(conv2d(h1, self.hidden_dim * 4, name='d_h2_conv'), 'd_bn2'))
            # 128
            h3 = lrelu(instance_norm(conv2d(h2, self.hidden_dim * 8, name='d_h3_conv'), 'd_bn3'))
            # 64
            h4 = lrelu(instance_norm(conv2d(h3, self.hidden_dim * 8, name='d_h4_conv'), 'd_bn4'))
            # 32
            h5 = conv2d(h4, 1, s=1, name='d_h4_pred')
            # 32
        return h5


class discriminator_256:
    def __init__(self, hidden_dim=64, name="discriminator"):
        self.name = name
        self.hidden_dim = hidden_dim

    def __call__(self, input, reuse=False):
        with tf.variable_scope(self.name):
            # image is 256 x 256 x input_c_dim
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            h0 = lrelu(conv2d(input, self.hidden_dim, name='d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(instance_norm(conv2d(h0, self.hidden_dim * 2, name='d_h1_conv'), 'd_bn1'))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(instance_norm(conv2d(h1, self.hidden_dim * 4, name='d_h2_conv'), 'd_bn2'))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(instance_norm(conv2d(h2, self.hidden_dim * 8, s=1, name='d_h3_conv'), 'd_bn3'))
            # h3 is (32 x 32 x self.df_dim*8)
            h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
            # h4 is (32 x 32 x 1)
            return h4


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_ - target) ** 2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
