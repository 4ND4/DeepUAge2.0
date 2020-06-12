import os

import Augmentor
import random
from PIL import Image
import numpy as np
import cv2
from keras.utils import Sequence, to_categorical


def get_transform_func():
    p = Augmentor.Pipeline()
    p.flip_left_right(probability=0.5)
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    p.zoom_random(probability=0.5, percentage_area=0.95)
    p.random_distortion(probability=0.5, grid_width=2, grid_height=2, magnitude=8)
    p.random_color(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_contrast(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_brightness(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_erasing(probability=0.5, rectangle_area=0.2)

    def transform_image(image):
        image = [Image.fromarray(image)]
        for operation in p.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                image = operation.perform_operation(image)
        return image[0]
    return transform_image


class FaceGenerator(Sequence):
    def __init__(self, image_dir, batch_size=32, image_size=224, number_classes=101):
        self.image_path_and_age = []
        self._load_image(image_dir)
        self.image_num = len(self.image_path_and_age)
        self.batch_size = batch_size
        self.image_size = image_size
        self.indices = np.random.permutation(self.image_num)
        self.transform_image = get_transform_func()
        self.number_classes = number_classes

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, 1), dtype=np.int32)

        sample_indices = self.indices[idx * batch_size:(idx + 1) * batch_size]

        for i, sample_id in enumerate(sample_indices):
            image_path, age = self.image_path_and_age[sample_id]

            greyscale_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            image = np.repeat(greyscale_image[..., np.newaxis], 3, -1)

            x[i] = self.transform_image(cv2.resize(image, (image_size, image_size)))
            y[i] = age

        return x, to_categorical(y, self.number_classes)

    def on_epoch_end(self):
        self.indices = np.random.permutation(self.image_num)

    def _load_image(self, image_dir):
        image_root = image_dir

        train_image_dir = os.path.join(image_root, 'train')

        for root, dirs, _ in os.walk(train_image_dir):
            for d in dirs:
                for x in os.listdir(os.path.join(train_image_dir, d)):
                    if not x.startswith('.'):
                        age = d
                        image_name = x

                        file_path = os.path.join(root, age, image_name)

                        if os.path.isfile(file_path):
                            self.image_path_and_age.append([str(file_path), age])


class ValGenerator(Sequence):
    def __init__(self, image_dir, batch_size=32, image_size=224, number_classes=101):
        self.image_path_and_age = []
        self._load_image(image_dir)
        self.image_num = len(self.image_path_and_age)
        self.batch_size = batch_size
        self.image_size = image_size
        self.number_classes = number_classes

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, 1), dtype=np.int32)

        for i in range(batch_size):
            image_path, age = self.image_path_and_age[idx * batch_size + i]

            greyscale_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            image = np.repeat(greyscale_image[..., np.newaxis], 3, -1)

            x[i] = cv2.resize(image, (image_size, image_size))
            y[i] = age

        return x, to_categorical(y, self.number_classes)

    def _load_image(self, image_dir):
        image_root = image_dir

        test_image_dir = os.path.join(image_root, 'validation')

        for root, dirs, _ in os.walk(test_image_dir):
            for d in dirs:
                for x in os.listdir(os.path.join(test_image_dir, d)):
                    if not x.startswith('.'):
                        age = d
                        image_name = x

                        file_path = os.path.join(root, age, image_name)

                        if os.path.isfile(file_path):
                            self.image_path_and_age.append([str(file_path), age])
