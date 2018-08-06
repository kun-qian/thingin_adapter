import logging
import os
import pickle
import re
import requests
from Semantic_Search.utils.preprocess import load_model
import config
from config import methods, FASTTEXT_COMMENTS_METHOD

logging.basicConfig(level=logging.INFO)

USE_CACHED_VECTOR = False
CACHE_FILE_BASIC_NAME = 'cached_classes_name_IRI_vector_method_{}.pkl'

# use COMMENTS if method is even, use NAMES if odd
method = FASTTEXT_COMMENTS_METHOD

cache_file = CACHE_FILE_BASIC_NAME.format(methods[method])
logging.info("loading models...")
config.models[method] = load_model(method)
logging.info("finish loading models...")

from Semantic_Search.DocSimWrapper import get_sentence_vector


def is_similarity_by_name(method):
    if method in [1, 3, 5, 7, 9]:
        return True
    else:
        return False


if USE_CACHED_VECTOR and os.path.exists(cache_file):
    logging.info("loading cache file to get vectors of classes")
    with open(cache_file, 'rb') as file:
        classes = pickle.load(file)
        print(classes[:3])
else:
    try:
        # os.environ['http_proxy'] = 'http://10.193.250.16:8080/'
        # os.environ['https_proxy'] = 'http://10.193.250.16:8080/'

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

    data_key = 'name' if is_similarity_by_name(method) else 'comment'

    for c in classes:
        if data_key not in c.keys():
            c['vec'] = None
        else:
            vector = get_sentence_vector(c[data_key], method)
            c['vec'] = vector
    # print(classes[:3])

    # write classes into local file for cache
    with open(cache_file, 'wb') as file:
        pickle.dump(classes, file)
