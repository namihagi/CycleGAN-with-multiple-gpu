import argparse

import tensorflow as tf

from model import Model

tf.set_random_seed(19)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--sub_dir', dest='sub_dir', help='sub directory name')
parser.add_argument('--gpu_num', dest='gpu_num', type=int, default=2, help='the number of gpu you use')
parser.add_argument('--global_batch_size', dest='global_batch_size',
                    type=int, default=16, help='overall batch size')
args = parser.parse_args()


def main(_):
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    usable_gpu = "0"
    for idx in range(args.gpu_num - 1):
        usable_gpu += ",{}".format(idx + 1)
    tfconfig.gpu_options.visible_device_list = usable_gpu
    with tf.Session(config=tfconfig) as sess:
        model = Model(sess, args)
        model.train() if args.phase == 'train' \
            else model.test()


if __name__ == '__main__':
    tf.app.run()
