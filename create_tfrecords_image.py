import io
import os
import shutil
from glob import glob

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


def dict_to_tf_example(image_path):
    filename = image_path.split('/')[-1]

    # load image
    image_path = os.path.join(image_path)
    with tf.gfile.GFile(image_path, 'rb') as f:
        encoded_jpg = f.read()

    # check image format
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG!')

    example = tf.train.Example(features=tf.train.Features(feature={
        'filename': _bytes_feature(filename.encode()),
        'image': _bytes_feature(encoded_jpg)
    }))
    return example


def main():
    image_dir = '../datasets/anime_face/train'
    output_dir = './datasets/anime_face_tfr/train'

    print('Reading images from:', image_dir)
    examples_list = glob(os.path.join(image_dir, '*.jpg'))
    num_examples = len(examples_list)
    print('Number of images:', num_examples)

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)

    for idx, example in tqdm(enumerate(examples_list)):
        print(example)
        shard_path = os.path.join(output_dir, 'shard-%04d.tfrecords' % idx)
        writer = tf.python_io.TFRecordWriter(shard_path)
        tf_example = dict_to_tf_example(example)
        writer.write(tf_example.SerializeToString())
        writer.close()

    print('Result is here:', output_dir)


main()
