import numpy as np
from gensim import matutils

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
