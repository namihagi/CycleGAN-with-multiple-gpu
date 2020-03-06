import argparse
import io
import json
import math
import os
import shutil

import PIL.Image
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _string_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring()]))


def dict_to_tf_example(example, npy_dir):
    filename = example.replace('.npy', '.jpg')
    path = os.path.join(npy_dir, example)
    npy_array = np.load(path).astype(np.float32)
    npy_array = np.reshape(npy_array, [32*32*128])

    example = tf.train.Example(features=tf.train.Features(feature={
        'filename': _bytes_feature(filename.encode()),
        'npy_array': _float_list_feature(npy_array),
    }))
    return example


def main():
    parser = argparse.ArgumentParser(description='')
    # dir setting
    parser.add_argument('--npy_dir', dest='npy_dir', default=None)
    parser.add_argument('--output_dir', dest='output_dir', default=None)
    args = parser.parse_args()

    print('Reading npy from:', args.npy_dir)

    examples_list = os.listdir(args.npy_dir)
    num_examples = len(examples_list)
    print('Number of images:', num_examples)

    num_shards = len(examples_list)
    shard_size = math.ceil(num_examples / num_shards)
    print('Number of images per shard:', shard_size)

    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir)

    shard_id = 0
    num_examples_written = 0
    writer = None
    for example in tqdm(examples_list):

        if num_examples_written == 0:
            shard_path = os.path.join(args.output_dir, 'shard-%04d.tfrecords' % shard_id)
            writer = tf.python_io.TFRecordWriter(shard_path)

        tf_example = dict_to_tf_example(example, args.npy_dir)
        writer.write(tf_example.SerializeToString())
        num_examples_written += 1

        if num_examples_written == shard_size:
            shard_id += 1
            num_examples_written = 0
            writer.close()

    if num_examples_written != shard_size and num_examples % num_shards != 0:
        writer.close()

    print('Result is here:', args.output_dir)


main()
