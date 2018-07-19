import logging

import requests
import re
import os
import pickle
from const import SEPARATOR, KEY_VALUE_SEPARATOR
from Semantic_Search.DocSimWrapper import get_sentence_vector, vecsim
from const import methods, D2V_DM_NAMES_METHOD, D2V_DM_COMMENTS_METHOD, D2V_DBOW_NAMES_METHOD, D2V_DBOW_COMMENTS_METHOD, \
    FASTTEXT_NAMES_METHOD, FASTTEXT_COMMENTS_METHOD

logging.basicConfig(level=logging.INFO)

USE_CACHED_VECTOR = True
CACHE_FILE_BASIC_NAME = 'cached_classes_name_IRI_vector_method_{}.pkl'

# use COMMENTS if method is even, use NAMES if odd
method = FASTTEXT_NAMES_METHOD

cache_file = CACHE_FILE_BASIC_NAME.format(methods[method])

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
        if method % 2 == 0:
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
    if method % 2 == 0:
        classes = [c for c in classes if 'comment' in c['data'].keys()]
        for c in classes:
            c['data']['comment'] = re.sub('\n|\r', ' ', c['data']['comment'][0])
        # print(classes[0])

    # remove duplicate iri class
    classes = [dict(element) for element in set([tuple(c['data'].items()) for c in classes])]

    data_key = 'name' if method % 2 == 1 else 'comment'

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

keywords = ['a place to have dinner', 'cool down the temperature', 'cooling', 'dinner', 'lunch dinner', 'drink',
            'water', 'heater', 'air conditioner', 'sunny', 'cafe', 'temperature controller', 'tea', 'hungry',
            'printer', 'cleaner', 'power charge', 'car wash', 'flower store', 'restaurant', 'theater', 'bicycle',
            'park', 'playground',
            'children playground', 'entertainment', 'bicycle station', 'bus station']


def get_top_n_similar_classes(vec, classes, n=30, threshold=0):
    res = []
    for c in classes:
        class_vec = c['vec']
        sim = vecsim(vec, class_vec)
        if sim > threshold:
            res.append({"class": c['iri'],
                        "name": c['name'],
                        "similarity": sim})

    res.sort(key=lambda x: x['similarity'], reverse=True)
    return res[:n]


with open("mapping_{}.txt".format(methods[method]), "w+") as f:
    for keyword in keywords:
        keyword_vec = get_sentence_vector(keyword, method)
        # keyword_vec = get_sentence_vector(keyword, method)
        similar_classes = get_top_n_similar_classes(keyword_vec, classes)
        print(similar_classes)
        similar_classes_str = ";".join(map(lambda x: x["name"] + SEPARATOR + x["class"] + SEPARATOR + str(x["similarity"]), similar_classes))
        f.write(keyword + KEY_VALUE_SEPARATOR + similar_classes_str + "\r\n")
