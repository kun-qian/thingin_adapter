import numpy as np

from gensim import matutils

from Semantic_Search.utils.GRAN_Model.gran_utils import get_seq
from Semantic_Search.utils.const import *
from Semantic_Search.utils.tools import split_phase
from config import *

'''
# The following codes refer to InferSent model
'''


def infersent_sentence2vec(sentence, model, method=INFERSENT_NAMES_METHOD, tokenize=True):
    assert model is not None

    splits = split_phase(sentence)

    if len(splits) == 0:
        logging.info('no word after split_phase in sentence: {} in model: {}!'.format(sentence, methods[method]))
        return None

    sent = ' '.join(splits)
    vec = model.encode([sent], tokenize)
    vec = matutils.unitvec(vec[0])

    return vec


'''
# The following codes refer to GRAN model
'''


def gran_sentence2vec(sentence, model, method=GRAN_NAMES_METHOD):
    assert model is not None

    splits = split_phase(sentence)

    if len(splits) == 0:
        logging.info('no word after split_phase in sentence: {} in model: {}!'.format(sentence, methods[method]))
        return None

    sen = ' '.join(splits)
    seq = []
    X1 = get_seq(sen, model['words'], testmode=DEBUG_ALGORITHM_TESTING)
    if X1 is None:
        return None
    if len(X1) == 0:
        return None
    seq.append(X1)
    x1, m1 = model['model'].prepare_data(seq)
    vec = model['model'].encoding_function(x1, m1)[0]
    vec = matutils.unitvec(vec)
    return vec


'''
# The following codes refer to Google USE model
'''


def use_sentence2vec(sentence, model, method=USE_NAMES_METHOD):
    assert model is not None

    splits = split_phase(sentence)

    if len(splits) == 0:
        logging.info('no word after split_phase in sentence: {} in model: {}!'.format(sentence, methods[method]))
        return None

    sent = ' '.join(splits)

    # by USEEncoder
    vec = model([sent]).astype(np.float64)

    # by USEPredictor
    # vec = model.encode([sent])

    vec = matutils.unitvec(vec[0])

    return vec
