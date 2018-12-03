import numpy as np

import config
from Semantic_Search.utils.Sentence2Vec import *

from config import *

def get_sentence_vector(sentence, method=USE_NAMES_METHOD):
    '''

    :param sentence:
    :param method:
    :return: a vector for this sentence, if failed, throw KeyError exception which need to be handled
    '''

    # assert isinstance(sentence, str) and 0 < method <= len(models)

    model = config.models[method]

    if 'name_vector_infersent' in methods[method]:
        vec = infersent_sentence2vec(sentence, model, method=INFERSENT_NAMES_METHOD, tokenize=True)
    elif 'name_vector_gran' in methods[method]:
        vec = gran_sentence2vec(sentence, model)
    elif 'name_vector_use' in methods[method]:
        vec = use_sentence2vec(sentence, model)

    return vec


def vecsim(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0
    return np.dot(vec1, vec2)