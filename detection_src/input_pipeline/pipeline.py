import tensorflow as tf

from detection_src.constants import NUM_THREADS


class Pipeline:
    """Input pipeline for training or evaluating object detectors."""

    def __init__(self, filenames, is_training, batch_size,
                 load_size=286, fine_size=256, shuffle=False):
        self.is_training = is_training
        self.load_size = load_size
        self.fine_size = fine_size
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

        dataset = dataset.map(self._parse_and_preprocess, num_parallel_calls=NUM_THREADS)

        # we need batches of fixed size
        padded_shapes = ([self.fine_size, self.fine_size, 3], [])
        dataset = dataset.padded_batch(batch_size, padded_shapes, drop_remainder=True)
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
            'image': tf.FixedLenFeature([], tf.string)
        }
        parsed_features = tf.parse_single_example(example_proto, features)

        # get image
        image = tf.image.decode_jpeg(parsed_features['image'], channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # now pixel values are scaled to [0, 1] range

        if self.is_training:
            image = tf.image.resize_images(image, size=[self.load_size, self.load_size])
            image = tf.image.random_crop(image, size=[self.fine_size, self.fine_size, 3])
        else:
            image = tf.image.resize_images(image, size=[self.fine_size, self.fine_size])
        image = (image * 2.0) - 1.0
        # now pixel values are scaled to [-1, 1] range

        filename = parsed_features['filename']
        return image, filename
