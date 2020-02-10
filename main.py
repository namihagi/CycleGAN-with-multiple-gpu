import argparse

import tensorflow as tf

from model import Model

tf.set_random_seed(19)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')

# dir setting
parser.add_argument('--train_A_path', dest='train_A_path', default=None, help='train_A_path')
parser.add_argument('--train_B_path', dest='train_B_path', default=None, help='train_B_path')
parser.add_argument('--test_A_path', dest='test_A_path', default=None, help='test_A_path')
parser.add_argument('--test_B_path', dest='test_B_path', default=None, help='test_B_path')
parser.add_argument('--class_name', dest='class_name', default=None, help='class name for test')
parser.add_argument('--sub_dir', dest='sub_dir', default=None, help='sub directory name')

# model setting
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='overall batch size')
parser.add_argument('--epoch', dest='epoch', type=int, default=300, help='the num of epoch')
parser.add_argument('--lr_decay_epoch', dest='lr_decay_epoch', type=int, default=150)
parser.add_argument('--lr', dest='lr', type=float, default=0.0002)
parser.add_argument('--cons_lambda', dest='cons_lambda', type=float, default=10.0)
parser.add_argument('--dete_lambda', dest='dete_lambda', type=float, default=5.0)
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=128)
parser.add_argument('--output_c_dim', dest='output_c_dim', type=int, default=128)
parser.add_argument('--B_range', dest='B_range', type=float, default=None)

args = parser.parse_args()


def main(_):
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        model = Model(sess, args)
        if args.phase == 'train':
            model.train()
        # else:
        #     model.test()


if __name__ == '__main__':
    tf.app.run()
