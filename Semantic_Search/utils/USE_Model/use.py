import tensorflow as tf
import tensorflow_hub as hub
import os

# Reduce logging output.
from Semantic_Search.utils.const import use_checkpoint_path


def load_use_embed(model_path):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(model_path)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})
