import pickle
from sklearn.decomposition import PCA
import numpy as np
import logging

from Semantic_Search.utils.const import *
from Semantic_Search.utils.tools import split_phase
from config import *


def cal_vector_u(sentence_list, model, method=WEIGHTED_W2V_GOOGLE_NAMES_METHOD, coef_a=coefficient_a):
    sentence_set = []
    embedding_size = model.get_dimension() if 'fasttext' in methods[method] else model.vector_size

    for sentence in sentence_list:
        vec = []  # add all word2vec values into one vector for the sentence

        splits = split_phase(sentence)

        if len(splits) == 0:
            continue
        else:
            if 'fasttext' in methods[method]:
                for word in splits:
                    a_value = coef_a / (coef_a + get_word_frequency(word))  # smooth inverse frequency, SIF
                    vec.append(np.multiply(a_value, model.get_word_vector(word)))  # vs += sif * word_vector
            else:
                for word in splits:
                    a_value = coef_a / (coef_a + get_word_frequency(word))  # smooth inverse frequency, SIF
                    try:
                        vec.append(np.multiply(a_value, model[word]))  # vs += sif * word_vector
                    except:
                        logging.info('no existed word {} in sentence: {} in model: {}!'
                                     .format(word, sentence, methods[method]))
                        if DEBUG_ALGORITHM_TESTING:
                            continue
                        else:
                            break

            if len(vec) == 0:
                continue

            vec = np.array(vec).mean(axis=0)  # weighted average
            sentence_set.append(vec)  # add to our existing re-calculated set of sentences

    # calculate PCA of this sentence set
    pca = PCA(n_components=embedding_size)
    pca.fit(np.array(sentence_set))
    u = pca.components_[0]  # the PCA vector
    u = np.multiply(u, np.transpose(u))  # u x uT

    # pad the vector?  (occurs if we have less sentences than embeddings_size)
    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            u = np.append(u, 0)  # add needed extension for multiplication below

    # dump vector u
    filename = vector_u_cache_file.format(methods[method])
    with open(filename, 'wb') as file:
        pickle.dump(u, file)

    return u


# todo: get a proper word frequency for a word in a document set
# or perhaps just a typical frequency for a word from Google's n-grams
def get_word_frequency(word_text):
    return 0.0001  # set to a low occurring frequency - probably not unrealistic for most words, improves vector values

def load_vector_u(method):
    filename = vector_u_cache_file.format(methods[method])
    try:
        with open(filename, 'rb') as file:
            vec_u = pickle.load(file)
        return vec_u
    except:
        return None