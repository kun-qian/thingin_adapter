import logging
import numpy as np
from gensim import matutils

from .utils.const import d2v_model2_path, d2v_model_path
from .utils.tools import split_phase
from .utils.preprocess import load_DM_model, load_FastText_model, load_DBOW_model

FORDEV = False
VECDIM = 10

methods = {1: 'd2v_dm_names', 2: 'd2v_dm_comments', 3: 'd2v_dbow_names', 4:'d2v_dbow_comments', 5: 'fasttext_names', 6: 'fasttext_comments'}


def dbow_model2_path(args):
    pass


if not FORDEV:

    models = [load_DM_model(model_path=d2v_model2_path), load_DBOW_model(model_path=d2v_model2_path), load_FastText_model()]


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


    model = models[int((method-1)/2)]

    if method in [1, 3]:
        try:
            vec = np.array([model[word] for word in split_phase(sentence)]).mean(axis=0)
            vec = matutils.unitvec(vec)
        except :
            logging.info('no existed word in sentence: {} in model: {}!'
                         .format(sentence, methods[method]))
            # raise KeyError('no vector for a word in sentence {}!'.format(sentence))
            return None
    elif method == 5:
        print(sentence)
        splits = split_phase(sentence)
        if len(splits) == 0:
            return None
        else:
            vec = np.array([model.get_word_vector(word) for word in splits]).mean(axis=0)
            vec = matutils.unitvec(vec)
    elif method == 6:
        vec = matutils.unitvec(model.get_sentence_vector(sentence))
    elif method in [2, 4]:
        try:
            vec = model.infer_vector(doc_words=split_phase(sentence), alpha=0.1, min_alpha=0.0001, steps=20)
            vec = matutils.unitvec(vec)
        except:
            logging.info('fail to infer vector of sentence: {}!'.format(sentence))
            # raise KeyError('no vector for a word in sentence {}!'.format(sentence))
            return None

    return vec


def vecsim(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0
    return np.dot(vec1, vec2)
