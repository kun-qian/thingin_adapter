import logging
import numpy as np
from gensim import matutils


from Semantic_Search.utils.GRAN_Model.gran_utils import get_seq
from Semantic_Search.utils.SIF_Model.SIF import get_word_frequency
from Semantic_Search.utils.const import *
from Semantic_Search.utils.tools import split_phase
from config import *


def average_sentence2vec(sentence, model, method=FASTTEXT_NAMES_METHOD):
    assert model is not None

    splits = split_phase(sentence)

    if len(splits) == 0:
        logging.info('no word after split_phase in sentence: {} in model: {}!'.format(sentence, methods[method]))
        return None

    vec = []
    for word in splits:
        if 'fasttext' in methods[method]:
            vec.append(model.get_word_vector(word))
        else:
            try:
                vec.append(model[word])
            except:
                logging.info('no existed word {} in sentence: {} in model: {}!'.format(word, sentence, methods[method]))
                if DEBUG_ALGORITHM_TESTING:
                    continue
                else:
                    return None

    if len(vec) == 0:
        return None

    vec = np.array(vec).mean(axis=0)
    vec = matutils.unitvec(vec)
    return vec



'''
# The following codes refer to paper "A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SENTENCE EMBEDDINGS" 
# by Sanjeev Arora, Yingyu Liang, Tengyu Ma, and their implementation in https://github.com/peter3125/sentence2vec,
# following Apache License, Version 2.0
'''


def weighted_sentence2vec(sentence, model, vec_u, method=WEIGHTED_W2V_GOOGLE_NAMES_METHOD, coef_a=coefficient_a):
    assert model is not None and vec_u is not None

    splits = split_phase(sentence)

    if len(splits) == 0:
        return None

    vec = []

    if 'fasttext' in methods[method]:
        for word in splits:
            a_value = coef_a / (coef_a + get_word_frequency(word))  # smooth inverse frequency, SIF
            vec.append(np.multiply(a_value, model.get_word_vector(word))) # vs += sif * word_vector
    else:
        for word in splits:
            a_value = coef_a / (coef_a + get_word_frequency(word))  # smooth inverse frequency, SIF
            try:
                vec.append(np.multiply(a_value, model[word]))  # vs += sif * word_vector
            except:
                logging.info('no existed word {} in sentence: {} in model: {}!'
                             .format(word, sentence, methods[method]))
                # raise KeyError('no vector for a word in sentence {}!'.format(sentence))
                if DEBUG_ALGORITHM_TESTING:
                    continue
                else:
                    return None

    if len(vec) == 0:
        return None

    vec = np.array(vec).mean(axis=0)  # weighted average
    sub = np.multiply(vec_u, vec)
    vec = np.subtract(vec, sub)
    vec = matutils.unitvec(vec)

    return vec


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
    X1 = get_seq(sen, model['words'])
    if X1 is None:
        return None
    if len(X1) == 0:
        return None
    seq.append(X1)
    x1, m1 = model['model'].prepare_data(seq)
    vec = model['model'].encoding_function(x1, m1)[0]
    vec = matutils.unitvec(vec)
    return vec
