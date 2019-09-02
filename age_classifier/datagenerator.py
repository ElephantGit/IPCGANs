import tensorflow as tf
import numpy as np

from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

IMAGE_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.
    Requires Tensorflow >= version 1.12rc0
    """
    def __init__(self, txt_file, mode, batch_size, num_classes, shuffle=True, buffer_size=1000):
        """Create a new ImageDataGenerator
        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            txt_file: path to the txt file  
            mode: Either 'training' or 'validation'. Depending on this value,
                  different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of the class in the dataset.
            shuffle: Whether or not to shuffle the data in the dataset and 
                     the initial file list.
            buffer_size: Number of images used as buffer for TensorFlows shuffling of the dataset.
        """
        self.txt_file = txt_file
        self.num_classes = num_classes

        # retrieve the data from the text file
        self._read_txt_file()
        
        # number of samples in the dataset
        self.data_size = len(self.labels)
