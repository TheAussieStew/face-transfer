import cv2
import numpy as np
import os
import pathlib


def generate_dirs():
    pathlib.Path('data/raw_data').mkdir(parents=True, exist_ok=True)
    pathlib.Path('data/training_data').mkdir(parents=True, exist_ok=True)
    pathlib.Path('models').mkdir(parents=True, exist_ok=True)
    pathlib.Path('test/input/videos').mkdir(parents=True, exist_ok=True)
    pathlib.Path('test/input/images').mkdir(parents=True, exist_ok=True)
    pathlib.Path('test/output/videos').mkdir(parents=True, exist_ok=True)
    pathlib.Path('test/output/images').mkdir(parents=True, exist_ok=True)


def get_image_paths(directory):
    return [x.path for x in os.scandir(directory) if x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".jpeg") or x.name.endswith(".JPG")]


def load_images(image_paths, size=(256, 256)):
    x = []

    for image_filepath in image_paths:
        image = cv2.imread(image_filepath)
        if image is not None:
            image = cv2.resize(image, dsize=size)
            x.append(image)

    x = np.array(x)
    return x


def get_transpose_axes(n):
    if n % 2 == 0:
        y_axes = list(range(1, n - 1, 2))
        x_axes = list(range(0, n - 1, 2))
    else:
        y_axes = list(range(0, n - 1, 2))
        x_axes = list(range(1, n - 1, 2))
    return y_axes, x_axes, [n - 1]


def stack_images(images):
    images_shape = np.array(images.shape)
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [np.prod(images_shape[x]) for x in new_axes]
    return np.transpose(
        images,
        axes=np.concatenate(new_axes)
    ).reshape(new_shape)
