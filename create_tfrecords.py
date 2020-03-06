import os
import shutil
from glob import glob

import numpy as np
import tensorflow as tf
from tqdm import tqdm


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _string_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring()]))


def dict_to_tf_example(npy_path):
    filename = npy_path.split('/')[-1]
    feature_array = np.load(npy_path).astype(np.float32)

    example = tf.train.Example(features=tf.train.Features(feature={
        'filename': _bytes_feature(filename.encode()),
        'feature': _string_feature(feature_array)
    }))
    return example


def main():
    npy_dir = '../pretrain_for_detection/test/ball_disk020_resnet_vggcentering/features/test'
    output_dir = './datasets/ball-web-disk020-features/test'

    print('Reading npys from:', npy_dir)
    examples_list = glob(os.path.join(npy_dir, '*.npy'))
    num_examples = len(examples_list)
    print('Number of npys:', num_examples)

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)

    for idx, example in tqdm(enumerate(examples_list)):
        shard_path = os.path.join(output_dir, 'shard-%04d.tfrecords' % idx)
        writer = tf.python_io.TFRecordWriter(shard_path)
        tf_example = dict_to_tf_example(example)
        writer.write(tf_example.SerializeToString())
        writer.close()

    print('Result is here:', output_dir)


main()
