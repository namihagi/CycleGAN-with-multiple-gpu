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


def dict_to_tf_example(annotation, class_name, image_dir, npy_dir):
    """Convert dict to tf.Example proto.

    Notice that this function normalizes the bounding
    box coordinates provided by the raw data.

    Arguments:
        data: a dict.
        image_dir: a string, path to the image directory.
    Returns:
        an instance of tf.Example.
    """
    image_name = annotation['filename']
    assert image_name.endswith('.jpg') or image_name.endswith('.jpeg')

    # load image
    image_path = os.path.join(image_dir, image_name)
    with tf.gfile.GFile(image_path, 'rb') as f:
        encoded_jpg = f.read()

    # check image format
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG!')

    # get image size
    width = int(annotation['size']['width'])
    height = int(annotation['size']['height'])
    assert width > 0 and height > 0
    assert image.size[0] == width and image.size[1] == height
    ymin, xmin, ymax, xmax = [], [], [], []

    just_name = image_name[:-4] if image_name.endswith('.jpg') else image_name[:-5]
    annotation_name = just_name + '.json'
    if len(annotation['object']) == 0:
        print(annotation_name, 'is without any objects!')

    for obj in annotation['object']:
        if obj['name'] == class_name:
            a = float(obj['bndbox']['ymin']) / height
            b = float(obj['bndbox']['xmin']) / width
            c = float(obj['bndbox']['ymax']) / height
            d = float(obj['bndbox']['xmax']) / width
            assert (a < c) and (b < d)
            ymin.append(a)
            xmin.append(b)
            ymax.append(c)
            xmax.append(d)

    example = tf.train.Example(features=tf.train.Features(feature={
        'filename': _bytes_feature(image_name.encode()),
        'image': _bytes_feature(encoded_jpg),
        'xmin': _float_list_feature(xmin),
        'xmax': _float_list_feature(xmax),
        'ymin': _float_list_feature(ymin),
        'ymax': _float_list_feature(ymax),
    }))
    return example


def main():
    parser = argparse.ArgumentParser(description='')
    # dir setting
    parser.add_argument('--image_dir', dest='image_dir', default=None)
    parser.add_argument('--annotations_dir', dest='annotations_dir', default=None)
    parser.add_argument('--npy_dir', dest='npy_dir', default=None)
    parser.add_argument('--output_dir', dest='output_dir', default=None)
    parser.add_argument('--class_name', dest='class_name', default=None)
    args = parser.parse_args()

    print('Reading images from:', args.image_dir)
    print('Reading annotations from:', args.annotations_dir, '\n')

    examples_list = os.listdir(args.annotations_dir)
    num_examples = len(examples_list)
    print('Number of images:', num_examples)

    num_shards = len(examples_list)
    shard_size = math.ceil(num_examples / num_shards)
    print('Number of images per shard:', shard_size)

    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir)

    shard_id = 0
    num_examples_written = 0
    for example in tqdm(examples_list):

        if num_examples_written == 0:
            shard_path = os.path.join(args.output_dir, 'shard-%04d.tfrecords' % shard_id)
            writer = tf.python_io.TFRecordWriter(shard_path)

        path = os.path.join(args.annotations_dir, example)
        annotation = json.load(open(path))
        tf_example = dict_to_tf_example(annotation, class_name, args.image_dir, args.npy_dir)
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
