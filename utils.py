import os

import numpy as np
import scipy.misc
from PIL import Image

try:
    _imread = scipy.misc.imread
except AttributeError:
    from imageio import imread as _imread


def makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_sample_npy(npy_array, max_range, output_dir):
    makedirs(output_dir)
    if len(npy_array) == 3:
        npy_array = np.expand_dims(npy_array, axis=0)
    for idx in range(len(npy_array)):
        save_arr = (npy_array[idx] + 1.0) * (max_range / 2.0)
        np.save(os.path.join(output_dir, "%04d" % idx), save_arr)


def save_sample_image(image_array, output_dir):
    image_array = (image_array + 1.0) * 127.5
    makedirs(output_dir)
    if len(image_array.shape) == 3:
        image_array = np.expand_dims(image_array, axis=0)
    for idx in range(len(image_array)):
        Image.fromarray(image_array[idx].astype(np.uint8)).save(os.path.join(output_dir, "%04d.jpg" % idx))


def load_data(file_path, image_size=256):
    img = Image.open(file_path).resize((image_size, image_size))
    image = np.asarray(img, dtype=np.float32)
    image = image / 127.5 - 1.
    # image pixel range (-1, 1)
    return image


def load_train_data(image_path, load_size=286, fine_size=256, is_testing=False):
    img_A = imread(image_path[0])
    img_B = imread(image_path[1])
    if not is_testing:
        img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        img_B = scipy.misc.imresize(img_B, [load_size, load_size])
        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
    else:
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    # img_AB = np.concatenate((img_A, img_B), axis=2)
    # # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_A, img_B


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
