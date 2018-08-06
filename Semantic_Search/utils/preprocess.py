import fastText
import logging
import os

from gensim.models.doc2vec import Doc2Vec
from gensim.models import KeyedVectors

from .const import *
import config

'''
The following functions are for training and loading Word2Vec models
'''


def load_model(method):
    if method in [config.D2V_DM_NAMES_METHOD, config.D2V_DM_COMMENTS_METHOD]:
        return load_DM_model()
    if method in [config.D2V_DBOW_NAMES_METHOD, config.D2V_DBOW_COMMENTS_METHOD]:
        return load_DBOW_model()
    if method in [config.FASTTEXT_NAMES_METHOD, config.FASTTEXT_COMMENTS_METHOD]:
        return load_FastText_model()
    if method in [config.W2V_GOOGLE_NAMES_METHOD]:
        return load_w2v_model()
    if method in [config.W2V_GLOVE_NAMES_METHOD]:
        return load_w2v_model(model_choice="glove")


def load_w2v_model(model_path=w2v_model_path, model_choice='google'):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(model_path, w2v_model_files[0] if model_choice == 'google' else w2v_model_files[1])
    model_dir = current_dir + filename

    if not os.path.exists(model_dir):
        logging.error('no LDA model file in:' + model_dir)
        return None

    if model_choice == 'google':
        w2v_model = KeyedVectors.load_word2vec_format(model_dir, binary=True)
    else:
        w2v_model = KeyedVectors.load_word2vec_format(model_dir, binary=False)
    return w2v_model


'''
The following functions are for training and loading Doc2Vec models
'''


def load_DBOW_model(model_path=d2v_model_path, model_file=dbow_model_file):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(model_path, model_file)
    model_dir = current_dir + filename
    if not os.path.exists(model_dir):
        logging.error('no LDA model file in:' + model_dir)
        return None
    dbow = Doc2Vec.load(model_dir)
    return dbow


def load_DM_model(model_path=d2v_model_path, model_file=dm_model_file):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(model_path, model_file)
    model_dir = current_dir + filename
    if not os.path.exists(model_dir):
        logging.error('no LDA model file in:' + model_dir)
        return None
    dbow = Doc2Vec.load(model_dir)
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
