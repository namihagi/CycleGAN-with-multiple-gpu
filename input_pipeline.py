import os
from glob import glob

import tensorflow as tf


class ImagePipeline:
    def __init__(self, dataset_dir, batch_size=1, image_size=512, shuffle=False):
        self.file_paths = glob(os.path.join(dataset_dir, '*.jpg'))
        self.num_files = len(self.file_paths)
        self.batch_size = batch_size
        self.image_size = image_size

        dataset = tf.data.Dataset.from_tensor_slices(self.file_paths)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.num_files)
        dataset = dataset.map(self._read_and_resize_image, num_parallel_calls=8)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=self.batch_size * 2)
        self.iter = dataset.make_initializable_iterator()

    def __del__(self):
        pass

    def get_file_num(self):
        return self.num_files

    def get_init_op_and_next_el(self):
        return self.iter.initializer, self.iter.get_next()

    def _read_and_resize_image(self, image_path):
        image = tf.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize_images(image, [self.image_size, self.image_size])
        image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=255)
        image = tf.cast(image, tf.float32)
        image = image / 127.5 - 1.0
        # image range is [-1.0, 1.0]
        return image_path, image


class NpyPipeline:
    def __init__(self, dataset_dir, batch_size=1, max_range=1.0, shuffle=False):
        self.file_paths = glob(os.path.join(dataset_dir, '*.tfrecords'))
        self.num_files = len(self.file_paths)
        self.batch_size = batch_size
        self.max_range = max_range

        dataset = tf.data.Dataset.from_tensor_slices(self.file_paths)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.num_files)
        dataset = dataset.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.map(self._read_npy, num_parallel_calls=8)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=self.batch_size * 2)
        self.iter = dataset.make_initializable_iterator()

    def __del__(self):
        pass

    def get_file_num(self):
        return self.num_files

    def get_init_op_and_next_el(self):
        return self.iter.initializer, self.iter.get_next()

    def _read_npy(self, example_proto):
        features = {
            'filename': tf.FixedLenFeature([], tf.string),
            'feature': tf.FixedLenFeature([], tf.string)
        }
        parsed_features = tf.parse_single_example(example_proto, features)

        filename = parsed_features['filename']
        feature = tf.decode_raw(parsed_features['feature'], tf.float32)
        feature = tf.reshape(feature, [32, 32, 128])
        feature = tf.clip_by_value(tf.to_float(feature), clip_value_min=0.0, clip_value_max=self.max_range)
        feature = (feature / (self.max_range / 2)) - 1.0
        # feature range is [-1.0, 1.0]

        return filename, feature
