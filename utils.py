import os

import numpy as np
import scipy.misc
from PIL import Image

try:
    _imread = scipy.misc.imread
except AttributeError:
    from imageio import imread as _imread


def load_data(file_path, image_size=256):
    img = Image.open(file_path).resize((image_size, image_size))
    image = np.asarray(img, dtype=np.float32)
    image = image / 127.5 - 1.
    # image pixel range (-1, 1)
    return image


def load_test_data(image_path, fine_size=256):
    img = imread(image_path)
    img = scipy.misc.imresize(img, [fine_size, fine_size])
    img = img / 127.5 - 1
    return img


def imread(path, is_grayscale=False):
    if is_grayscale:
        return _imread(path, flatten=True).astype(np.float)
    else:
        return _imread(path, mode='RGB').astype(np.float)


def inverse_transform(images):
    return (images + 1.) / 2.


def imsave(images, path):
    return scipy.misc.imsave(path, images)


def save_images(base_filenames, sample_images, dir_path, direction, index):
    for img_id in range(len(sample_images)):
        image = inverse_transform(sample_images[img_id])
        filename = os.path.join(dir_path, '{:>03d}_{:>03d}_{}_{}'.format(index, img_id, direction,
                                                                         base_filenames[img_id].split('/')[-1]))
        imsave(image, filename)
