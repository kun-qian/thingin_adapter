import fastText
import logging
import os

from gensim.models.doc2vec import Doc2Vec

from .const import *

'''
The following functions are for training and loading Doc2Vec models
'''


def load_DBOW_model(model_path=d2v_model_path, model_file=dbow_model_file):
    filename = os.path.join(model_path, model_file)
    if not os.path.exists(filename):
        logging.error('no LDA model file in:' + filename)
        return None
    dbow = Doc2Vec.load(filename)
    return dbow


def load_DM_model(model_path=d2v_model_path, model_file=dm_model_file):
    filename = os.path.join(model_path, model_file)
    if not os.path.exists(filename):
        logging.error('no LDA model file in:' + filename)
        return None
    dbow = Doc2Vec.load(filename)
    return dbow


'''
The following functions are for loading FastText models
'''


def load_FastText_model():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = current_dir + fasttext_model_filepath
    if not os.path.exists(model_dir):
        logging.error('no fastText model file in:' + fasttext_model_filepath)
        return None
    fmodel = fastText.load_model(model_dir)
    logging.info('fastText model loaded!')
    return fmodel
