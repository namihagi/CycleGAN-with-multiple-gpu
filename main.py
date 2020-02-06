import argparse

import tensorflow as tf

from model import Model

tf.set_random_seed(19)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')

# dir setting
parser.add_argument('--datasetA', dest='datasetA', default='ball', help='datasetA name')
parser.add_argument('--datasetB', dest='datasetB', default='wider-face', help='datasetB name')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='./datasets', help='path in which datasets exists')
parser.add_argument('--sub_dir', dest='sub_dir', default=None, help='sub directory name')

# model setting
parser.add_argument('--gpu_num', dest='gpu_num', type=int, default=1, help='the number of gpu you use')
parser.add_argument('--global_batch_size', dest='global_batch_size', type=int, default=1, help='overall batch size')
parser.add_argument('--epoch', dest='epoch', type=int, default=300, help='the num of epoch')
parser.add_argument('--lr_decay_epoch', dest='lr_decay_epoch', type=int, default=150)
parser.add_argument('--lr', dest='lr', type=float, default=0.0002)
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0)
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=64)
parser.add_argument('--B_range', dest='B_range', type=float, default=15.1)

# input setting
parser.add_argument('--image_size', dest='image_size', type=int, default=512)

args = parser.parse_args()


def main(_):
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # gpu setting
    usable_gpu = "0"
    for idx in range(args.gpu_num - 1):
        usable_gpu += ",{}".format(idx + 1)
    tfconfig.gpu_options.visible_device_list = usable_gpu

    with tf.Session(config=tfconfig) as sess:
        model = Model(sess, args)
        if args.phase == 'train':
            model.train()
        # else:
        #     model.test()


if __name__ == '__main__':
    tf.app.run()
