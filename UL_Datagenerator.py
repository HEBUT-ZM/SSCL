# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
MEAN = tf.constant([160, 139, 122], dtype=tf.float32)
class UImageDataGenerator(object):
    def __init__(self, images, batch_size, num_classes, shuffle=True):
        self.img_paths = images
        self.num_classes = num_classes
        self.data_size = len(self.img_paths)
        self.pointer = 0
        if shuffle:
            self._shuffle_lists()
        self.img_paths = convert_to_tensor(self.img_paths,dtype=dtypes.string)
        data = tf.data.Dataset.from_tensor_slices((self.img_paths))
        data = data.map(self._parse_function_train)              
        data = data.batch(batch_size)       
        self.data = data
    def _shuffle_lists(self):
        path = self.img_paths
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        for i in permutation:
            self.img_paths.append(path[i])
    def _parse_function_train(self, filename):
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [256, 256],method=2)
        img_centered = tf.subtract(img_resized, MEAN)
        img_bgr = img_centered[:, :, ::-1]
        return img_bgr
