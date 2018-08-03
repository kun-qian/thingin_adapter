import logging
import numpy as np
from gensim import matutils
import config
from .utils.tools import split_phase

from config import D2V_DM_NAMES_METHOD, D2V_DM_COMMENTS_METHOD, \
    D2V_DBOW_NAMES_METHOD, D2V_DBOW_COMMENTS_METHOD, \
    FASTTEXT_NAMES_METHOD, FASTTEXT_COMMENTS_METHOD, methods, W2V_GOOGLE_NAMES_METHOD, \
    W2V_GLOVE_NAMES_METHOD

FORDEV = False
VECDIM = 10


if not FORDEV:
    dm_model = config.d2v_dm_model
    dbow_model = config.d2v_dbow_model  # load_DBOW_model(model_path=d2v_model2_path)
    fasttext_model = config.fasttext_model
    w2v_google_model = config.w2v_google_model  # load_w2v_model(model_choice='google')
    w2v_glove_model = config.w2v_glove_model  # load_w2v_model(model_choice='glove')

    models = {D2V_DM_NAMES_METHOD: dm_model,
              D2V_DM_COMMENTS_METHOD: dm_model,
              D2V_DBOW_NAMES_METHOD: dbow_model,
              D2V_DBOW_COMMENTS_METHOD: dbow_model,
              FASTTEXT_NAMES_METHOD: fasttext_model,
              FASTTEXT_COMMENTS_METHOD: fasttext_model,
              W2V_GOOGLE_NAMES_METHOD: w2v_google_model,
              W2V_GLOVE_NAMES_METHOD: w2v_glove_model}


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


def get_sentence_vector(sentence, method=D2V_DBOW_NAMES_METHOD):
    '''

    :param sentence:
    :param method:
    :return: a vector for this sentence, if failed, throw KeyError exception which need to be handled
    '''

    # assert isinstance(sentence, str) and 0 < method <= len(models)

    if FORDEV:
        vec = np.random.random(size=VECDIM)
        vec = matutils.unitvec(vec)
        return vec

    model = models[method]

    if model is None:
        return None

    if method in [D2V_DM_NAMES_METHOD, D2V_DBOW_NAMES_METHOD, W2V_GOOGLE_NAMES_METHOD, W2V_GLOVE_NAMES_METHOD]:
        vec = []
        for word in split_phase(sentence):
            try:
                vec.append(model[word])
            except:
                logging.info('no existed word {} in sentence: {} in model: {}!'.format(word, sentence, methods[method]))
                continue
        if len(vec) == 0:
            return None
        vec = np.array(vec).mean(axis=0)
        vec = matutils.unitvec(vec)
        return vec
    elif method == FASTTEXT_NAMES_METHOD:
        print(sentence)
        splits = split_phase(sentence)
        if len(splits) == 0:
            return None
        else:
            vec = np.array([model.get_word_vector(word) for word in splits]).mean(axis=0)
            vec = matutils.unitvec(vec)
    elif method == FASTTEXT_COMMENTS_METHOD:
        vec = matutils.unitvec(model.get_sentence_vector(sentence))
    elif method in [D2V_DM_COMMENTS_METHOD, D2V_DBOW_COMMENTS_METHOD]:
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


