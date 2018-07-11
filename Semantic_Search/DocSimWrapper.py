import logging
import numpy as np
from gensim import matutils

from .utils.const import d2v_model2_path
from .utils.tools import split_phase
from .utils.preprocess import load_DM_model, load_FastText_model

FORDEV = False
VECDIM = 10

methods = {1: 'd2v_dm_words', 2: 'fasttext_words', 3: 'fasttext_sentence'}

if not FORDEV:
    models = [load_DM_model(model_path=d2v_model2_path), load_FastText_model()]


def sensim(source, targets, method=1, threshhold=0):
    sims = []

    if FORDEV:
        rsims = list(np.random.random(size=VECDIM))
        for index, score in enumerate(rsims):
            sims.append({'index': index, 'score': score})
        sims.sort(key=lambda k: k['score'], reverse=True)
        return sims

    if isinstance(targets, str):
        targets = [targets]
    else:
        targets = targets

    try:
        source_vec = get_sentence_vector(source, method)
    except KeyError:
        return sims
    for index, sentence in enumerate(targets):
        try:
            target_vec = get_sentence_vector(sentence, method)
        except KeyError:
            continue
        sim_score = vecsim(source_vec, target_vec)
        if sim_score < threshhold:
            logging.info('similarity between source: {} and target: {} is lower than threshold {}, ignored!'.
                         format(source, sentence, threshhold))
            continue
        sims.append({'index': index, 'score': sim_score, })

    # Sort results by score in desc order
    sims.sort(key=lambda k: k['score'], reverse=True)
    return sims


def get_sentence_vector(sentence, method=1):
    '''

    :param sentence:
    :param method:
    :return: a vector for this sentence, if failed, throw KeyError exception which need to be handle
    '''

    # assert isinstance(sentence, str) and 0 < method <= len(models)

    if FORDEV:
        vec = np.random.random(size=VECDIM)
        vec = matutils.unitvec(vec)
        return vec


    model = models[0] if method == 1 else models[1]

    if method == 1:
        try:
            vec = np.array([model[word] for word in split_phase(sentence)]).mean(axis=0)
            vec = matutils.unitvec(vec)
        except :
            logging.info('no existed word in sentence: {} in model: {}!'
                         .format(sentence, methods[method]))
            # raise KeyError('no vector for a word in sentence {}!'.format(sentence))
            return None
    elif method == 2:
        print(sentence)
        splits = split_phase(sentence)
        if len(splits) == 0:
            return None
        else:
            vec = np.array([model.get_word_vector(word) for word in splits]).mean(axis=0)
            vec = matutils.unitvec(vec)
    elif method == 3:
        vec = matutils.unitvec(model.get_sentence_vector(sentence))

    return vec


def vecsim(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0
    return np.dot(vec1, vec2)
