'''
solution to segfaul when import fastText and import torch
https://github.com/pytorch/pytorch/issues/9129

Reason:

    pytorch is compiled with gcc 4.9.2
    conda's default gcc is 4.8.5

Fix:

    install gcc-4.9 in conda (e.g. conda install -c serge-sans-paille gcc_49)
    install pytorch with conda install (in my case, conda install pytorch torchvision cuda90 -c pytorch)
    install fastText with gcc-4.9 compiler: CC=gcc-4.9 pip install . in the fastText git clone

other requirements:
Theano 1.0.2: pip install Theano
Ladage 0.2 dev : pip install https://github.com/Lasagne/Lasagne/archive/master.zip
'''
import logging

logging.basicConfig(level=logging.INFO)
import os
import pickle
import fastText
import theano
import lasagne
import torch
from gensim.models import KeyedVectors

from Semantic_Search.utils.GRAN_Model.GRAN import models
from Semantic_Search.utils.GRAN_Model.gran_utils import get_wordmap
from Semantic_Search.utils.InferSent_Model.InferSent import InferSent
from Semantic_Search.utils.const import *
import config


def load_model(method):
    if method in [config.FASTTEXT_NAMES_METHOD, config.WEIGHTED_W2V_FASTTEXT_NAMES_METHOD]:
        return load_FastText_model()
    if method in [config.W2V_GOOGLE_NAMES_METHOD, config.WEIGHTED_W2V_GOOGLE_NAMES_METHOD]:
        return load_w2v_model()
    if method in [config.INFERSENT_NAMES_METHOD]:
        return load_InferSent_model()
    if method in [config.GRAN_NAMES_METHOD]:
        return load_GRAN_model()


def load_w2v_model(model_path=w2v_model_path, model_choice='google'):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(model_path, w2v_model_files[0] if model_choice == 'google' else w2v_model_files[1])
    model_dir = current_dir + filename

    if not os.path.exists(model_dir):
        logging.error('no w2v model file in:' + model_dir)
        return None

    if model_choice == 'google':
        w2v_model = KeyedVectors.load_word2vec_format(model_dir, binary=True)
    else:
        w2v_model = KeyedVectors.load_word2vec_format(model_dir, binary=False)

    logging.info('finished loading model for w2v vector of : ' + model_choice)

    return w2v_model


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
    logging.info('finished loading fastText model!')
    return fmodel


'''
The following functions are for loading InferSent model
'''


def load_InferSent_model():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    MODEL_PATH = current_dir + infersent_model_filepath.format(infersent_version)
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': infersent_version,
                    'use_cuda': True}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))
    logging.info('finished loading InferSent model!')

    W2V_PATH = current_dir + infersent_wordvec_filepath[infersent_version - 1]
    infersent.set_w2v_path(W2V_PATH)

    logging.info('start building InferSent vocabulary !')

    infersent.build_vocab_k_words(K=100000)

    logging.info('finished building InferSent vocabulary !')

    return infersent


'''
The following functions are for loading GRAN model
'''


def load_GRAN_model():
    theano.config.dnn.enabled = 'False'

    current_dir = os.path.dirname(os.path.realpath(__file__))

    params = {'dropout': 0.0, 'word_dropout': 0.0, 'model': 'gran', 'outgate': True, 'gran_type': 1,
              'dim': 300, 'sumlayer': False}

    W2V_PATH = current_dir + gran_wordvec_filepath
    (words, We) = get_wordmap(W2V_PATH)
    model = models(We, params)
    MODEL_PATH = current_dir + gran_model_filepath
    base_params = pickle.load(open(MODEL_PATH, 'rb'), encoding='iso-8859-1')
    lasagne.layers.set_all_param_values(model.final_layer, base_params)

    logging.info('GRAN model loaded')

    return {'model': model, 'words': words}


if __name__ == '__main__':
    # load_InferSent_model()
    # load_GRAN_model()
    # print(dir(theano.config.dnn))
    pass
