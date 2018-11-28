import logging
import os
import pickle
import re
import requests

from Semantic_Search.utils.SIF_Model.SIF import load_vector_u, cal_vector_u
from Semantic_Search.utils.preprocess import load_model
import config
from config import *

logging.basicConfig(level=logging.INFO)

# os.environ['http_proxy'] = 'http://10.193.250.16:8080/'
# os.environ['https_proxy'] = 'http://10.193.250.16:8080/'

# only run onece to download the NLTK data for tokenizer

# only  run onece to download the NLTK data for tokenizer#  import nltk
# nltk.download('punkt')


USE_CACHED_VECTOR = False
CACHE_FILE_BASIC_NAME = 'cached_classes_name_IRI_vector_method_{}.pkl'

logging.info("loading models...")
for method in config.enabled_methods:
    if config.models[method] is None:
        config.models[method] = load_model(method)
    if method == config.FASTTEXT_NAMES_METHOD and config.WEIGHTED_W2V_FASTTEXT_NAMES_METHOD in config.enabled_methods:
        config.models[config.WEIGHTED_W2V_FASTTEXT_NAMES_METHOD] = config.models[method]
    if method == config.W2V_GOOGLE_NAMES_METHOD and config.WEIGHTED_W2V_GOOGLE_NAMES_METHOD in config.enabled_methods:
        config.models[config.WEIGHTED_W2V_GOOGLE_NAMES_METHOD] = config.models[method]
logging.info("finish loading models...")

from Semantic_Search.DocSimWrapper import get_sentence_vector


def is_similarity_by_name(method=None):
    # if method in [1, 3, 5, 7, 9]:
    #     return True
    # else:
    #     return False
    return True


for method in config.enabled_methods:
    cache_file = CACHE_FILE_BASIC_NAME.format(methods[method])

    if USE_CACHED_VECTOR and os.path.exists(cache_file):
        logging.info("loading cache file to get vectors of classes")
        with open(cache_file, 'rb') as file:
            classes = pickle.load(file)
            print(classes[:3])
            for i in range(len(classes)):
                if classes[i]['name'] == 'Tricam':
                    print(i)
    else:
        try:
            # retrieve classes with comments
            if not is_similarity_by_name(method):
                r = requests.get(
                    "http://ziggy-dev-ols.nprpaas.ddns.integ.dns-orange.fr/api/v1/ontologies/classes?includeFields="
                    "iri%2Cname%2Ccomment&limit=10000")
            else:  # retrieve classes just with name
                r = requests.get(
                    "http://ziggy-dev-ols.nprpaas.ddns.integ.dns-orange.fr/api/v1/ontologies/classes?includeFields="
                    "iri%2Cname&limit=10000")
            logging.info("received %s classes from thingin!" % len(r.json()))
        except requests.RequestException:
            logging.info("can't get info from thingin!")
            exit()

        classes = r.json()
        filtered_keywords = 'command|notification|functionality|function|unit|system|status|state|scheme|fragment|package' \
                            '|specification'
        classes = [c for c in classes if len(re.findall(filtered_keywords, c['data']['name'].lower())) == 0]

        # if consider the comments similarity, remove classes without comments
        if not is_similarity_by_name(method):
            classes = [c for c in classes if 'comment' in c['data'].keys()]
            for c in classes:
                c['data']['comment'] = re.sub('\n|\r', ' ', c['data']['comment'][0])
                # print(classes[0])

        # remove duplicate iri class
        classes = [dict(element) for element in set([tuple(c['data'].items()) for c in classes])]

    data_key = 'name' if is_similarity_by_name() else 'comment'

    # build the cache of vector_u for weighted average vector methods
    if method in [config.WEIGHTED_W2V_FASTTEXT_NAMES_METHOD, config.WEIGHTED_W2V_GOOGLE_NAMES_METHOD]:
        vec_u = load_vector_u(method)
        if vec_u is None:
            all_sentences = (c[data_key] for c in classes)
            all_sentences = list(set(all_sentences))
            vec_u = cal_vector_u(all_sentences, models[method], method)
        vectors_u[method] = vec_u

    for c in classes:
        if data_key not in c.keys():
            c['vec'] = None
        else:
            vector = get_sentence_vector(c[data_key], method)
            c['vec'] = vector
    # print(classes[:3])

    # write classes into local file for cache
    print('write total {} classes into file: {}'.format(len(classes), cache_file))
    with open(cache_file, 'wb') as file:
        pickle.dump(classes, file)
