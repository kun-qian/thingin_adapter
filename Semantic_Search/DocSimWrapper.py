import config
from Semantic_Search.utils.Sentence2Vec import *

from config import *


# FORDEV = False
# VECDIM = 10
#
# if not FORDEV:
#     fasttext_model = config.models[FASTTEXT_NAMES_METHOD]
#     w2v_google_model = config.models[W2V_GOOGLE_NAMES_METHOD]
#     inferset_model = config.models[INFERSENT_NAMES_METHOD]
#     gran_model = config.models[GRAN_NAMES_METHOD]
#
#     models = {FASTTEXT_NAMES_METHOD: fasttext_model,
#               W2V_GOOGLE_NAMES_METHOD: w2v_google_model,
#               WEIGHTED_W2V_FASTTEXT_NAMES_METHOD: fasttext_model,
#               WEIGHTED_W2V_GOOGLE_NAMES_METHOD: w2v_google_model,
#               INFERSENT_NAMES_METHOD: inferset_model,
#               GRAN_NAMES_METHOD: gran_model
#               }
#
#     vector_u = None  # vector u for weighted avaerage word2vec


def get_sentence_vector(sentence, method=W2V_GOOGLE_NAMES_METHOD):
    '''

    :param sentence:
    :param method:
    :return: a vector for this sentence, if failed, throw KeyError exception which need to be handled
    '''

    # assert isinstance(sentence, str) and 0 < method <= len(models)

    model = config.models[method]

    if 'average_name_vector_' in methods[method]:
        vec = average_sentence2vec(sentence, model, method)
    elif 'weighted_name_vector_' in methods[method]:
        vector_u = config.vectors_u[method]
        vec = weighted_sentence2vec(sentence, model, vector_u, method)
    elif 'name_vector_infersent' in methods[method]:
        vec = infersent_sentence2vec(sentence, model, method=INFERSENT_NAMES_METHOD, tokenize=True)
    elif 'name_vector_gran' in methods[method]:
        vec = gran_sentence2vec(sentence, model)
    return vec


def vecsim(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0
    return np.dot(vec1, vec2)
