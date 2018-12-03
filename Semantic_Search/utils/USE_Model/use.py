import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
#import seaborn as sns

# Reduce logging output.
from utils.const import use_model_path

tf.logging.set_verbosity(tf.logging.ERROR)

model_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"  #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
model_dir = use_model_path #'/project/model/use_model/'
model_version = 'usev3'


def load_use_embed(model_path=model_dir):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(os.path.join(model_path, model_version))
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})


class USEEncoder(object):
    def __init__(self, model_dir, tf_sess=None):

        os.environ['TFHUB_CACHE_DIR'] = model_dir

        # predict_fn = tf.contrib.predictor.from_saved_model(os.path.join(model_dir, model_version))

        try:
            if len(os.listdir(model_dir)) != 0:
                # Import the Universal Sentence Encoder's TF Hub module
                self.use_model = hub.Module(os.path.join(model_dir, model_version))
            else:
                self.use_model = hub.Module(model_url)
        except:
            self.use_model = hub.Module(model_url)

        self.tf_sess = tf_sess

        if self.tf_sess is None:
            # config = tf.ConfigProto()
            # config.gpu_options.allow_growth = True
            # self.tf_sess = tf.Session(config=config)
            self.tf_sess = tf.Session()

        self.tf_sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def encode(self, lines):
        message_embeddings = self.tf_sess.run(self.use_model(lines))

        return message_embeddings

    def get_session(self):
        return self.tf_sess


if __name__ == '__main__':
    use_encoder = USEEncoder(model_dir)
    use_encoder.encode(['hello world'])
    use_encoder.encode(['hello everybody'])

