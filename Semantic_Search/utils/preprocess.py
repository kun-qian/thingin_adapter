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



import os
import pickle
import theano
import lasagne
import torch

from Semantic_Search.utils.GRAN_Model.GRAN import models
from Semantic_Search.utils.GRAN_Model.gran_utils import get_wordmap
from Semantic_Search.utils.InferSent_Model.InferSent import InferSent
from Semantic_Search.utils.USE_Model.use_predictor import USEPredictor
from Semantic_Search.utils.const import *

import config

logging.basicConfig(level=logging.INFO)


def load_model(method):
    if method in [config.USE_NAMES_METHOD]:
        return load_USE_model()
    if method in [config.INFERSENT_NAMES_METHOD]:
        return load_InferSent_model()
    if method in [config.GRAN_NAMES_METHOD]:
        return load_GRAN_model()

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


'''
The following functions are for loading Google USE model
'''
def load_USE_model(session=None):

    #assert session is not None
    #use_encoder = load_use_embed(use_model_path)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    MODEL_PATH = current_dir + use_savedmodel_path

    use_predictor = USEPredictor(MODEL_PATH)
    logging.info('Google USE model loaded!')

    return use_predictor



if __name__ == '__main__':
    # load_InferSent_model()
    # load_GRAN_model()
    # print(dir(theano.config.dnn))
    pass
