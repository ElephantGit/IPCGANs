import tensorflow as tf
import numpy as np

class AlexNet(object):
    """Implementation of the AlexNet."""
    def __init__(self, x, keep_prob, num_class, skip_layer, weights_path='DEFAULT'):
        """Create the graph of the AlexNet model."""
         
