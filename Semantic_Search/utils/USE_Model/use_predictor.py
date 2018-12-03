import os

import tensorflow as tf
import numpy as np


# Reduce logging output.
tf.logging.set_verbosity(tf.logging.INFO)

class USEPredictor(object):

    def __init__(self, savedmodel):
        assert os.path.exists(savedmodel)
        self.predictor = tf.contrib.predictor.from_saved_model(savedmodel)

    def encode(self, lines):
        feed_dict = {'text': lines}
        return self.predictor(feed_dict)['embeddings'].astype(np.float64)