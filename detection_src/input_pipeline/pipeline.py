import tensorflow as tf

from detection_src.constants import SHUFFLE_BUFFER_SIZE, NUM_THREADS, RESIZE_METHOD
from detection_src.input_pipeline.random_image_crop import random_image_crop
from detection_src.input_pipeline.other_augmentations import random_color_manipulations, \
    random_flip_left_right, random_pixel_value_scale, random_jitter_boxes


class Pipeline:
    """Input pipeline for training or evaluating object detectors."""

    def __init__(self, filenames, batch_size, max_range=None, shuffle=False):
        if max_range is None:
            self.max_range = 1.0
            self.min_range = -1.0
        else:
            self.max_range = max_range
            self.min_range = 0.0

        self.batch_size = batch_size

        def get_num_samples(filename):
            return sum(1 for _ in tf.python_io.tf_record_iterator(filename))

        num_examples = 0
        for filename in filenames:
            num_examples_in_file = get_num_samples(filename)
            assert num_examples_in_file > 0
            num_examples += num_examples_in_file
        self.num_examples = num_examples
        assert self.num_examples > 0

        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        self.num_shards = len(filenames)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.num_shards)

        dataset = dataset.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.prefetch(buffer_size=batch_size * 2)

        # if shuffle:
        #     dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        dataset = dataset.map(self._parse_and_preprocess, num_parallel_calls=NUM_THREADS)

        # we need batches of fixed size
        padded_shapes = ([32, 32, 128], [3], [None, 4], [], [])
        dataset = dataset.apply(
            tf.contrib.data.padded_batch_and_drop_remainder(batch_size, padded_shapes)
        )
        dataset = dataset.prefetch(buffer_size=1)

        self.iterator = dataset.make_initializable_iterator()

    def get_init_op_and_next_el(self):
        """
        :return:
            init_op: for initializer
            next_el: to get next data (feature, shape, boxes, num_boxes, filename)
        """
        init_op = self.iterator.initializer
        next_el = self.iterator.get_next()
        return init_op, next_el, self.num_shards

    def _parse_and_preprocess(self, example_proto):
        """What this function does:
        1. Parses one record from a tfrecords file and decodes it.
        2. (optionally) Augments it.
        Returns:
            image: a float tensor with shape [image_height, image_width, 3],
                an RGB image with pixel values in the range [0, 1].
            boxes: a float tensor with shape [num_boxes, 4].
            num_boxes: an int tensor with shape [].
            filename: a string tensor with shape [].
        """
        features = {
            'filename': tf.FixedLenFeature([], tf.string),
            'img_shape': tf.FixedLenFeature([3], tf.int64),
            'feature': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'ymin': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'xmin': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'ymax': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'xmax': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        }
        parsed_features = tf.parse_single_example(example_proto, features)

        # get feature
        feature = tf.to_float(parsed_features['feature'])
        feature = tf.reshape(feature, [32, 32, 128])
        feature = tf.clip_by_value(feature, clip_value_min=self.min_range, clip_value_max=self.max_range)
        # now pixel values are scaled to [min_range, max_range] range
        if self.max_range != 1.0:
            feature = (feature / (self.max_range / 2)) - 1.0
        feature = tf.clip_by_value(feature, clip_value_min=-1.0, clip_value_max=1.0)
        # now pixel values are scaled to [-1, 1] range

        # get ground truth boxes, they must be in from-zero-to-one format
        boxes = tf.stack([
            parsed_features['ymin'], parsed_features['xmin'],
            parsed_features['ymax'], parsed_features['xmax']
        ], axis=1)
        boxes = tf.to_float(boxes)
        # it is important to clip here!
        boxes = tf.clip_by_value(boxes, clip_value_min=0.0, clip_value_max=1.0)

        img_shape = tf.to_int32(parsed_features['img_shape'])
        num_boxes = tf.to_int32(tf.shape(boxes)[0])
        filename = parsed_features['filename']
        return feature, img_shape, boxes, num_boxes, filename
