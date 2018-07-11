import logging

import requests
import time
import re, os, pickle
from Semantic_Search.DocSimWrapper import get_sentence_vector, vecsim

logging.basicConfig(level=logging.INFO)

USECACHEDVECTOR = True
cache_file_basic_name = 'cached_classes_name_IRI_vector_method{}.pkl'

# method could be 1, 2, 3
method = 2

cache_file = cache_file_basic_name.format(method)

if USECACHEDVECTOR and os.path.exists(cache_file):
    with open(cache_file, 'rb') as file:
        start = time.time()
        classes = pickle.load(file)
        print(classes[:3])
else:
    try:
        os.environ['http_proxy'] = 'http://10.193.250.16:8080/'
        os.environ['https_proxy'] = 'http://10.193.250.16:8080/'
        r = requests.get(
            "http://ziggy-dev-ols.nprpaas.ddns.integ.dns-orange.fr/api/v1/ontologies/classes?includeFields=iri%2Cname&limit=10000")
        logging.info("received %s classes from thingin!" % len(r.json()))

    except requests.RequestException:
        logging.info("can't get info from thingin!")
        exit()

    classes = r.json()
    filter_keywords = 'command|notification|functionality|function|unit|system|status|state|scheme|fragment|package' \
                      '|specification'
    classes = [c for c in classes if len(re.findall(filter_keywords, c['data']['name'].lower())) == 0]
    # remove duplicate iri class
    classes = [dict(element) for element in set([tuple(c['data'].items()) for c in classes])]



    start = time.time()
    for c in classes:
        vector = get_sentence_vector(c['name'], method)
        c['vec'] = vector
    print(classes[:3])

    # write classes into local file for cache
    with open(cache_file, 'wb') as file:
        pickle.dump(classes, file)

keywords = ['lunch', 'dinner', 'lunch dinner', 'drink', 'water', 'heater', 'air conditioner', 'sunny', 'cafe', 'temperature controller', 'tea', 'hungry',
            'printer', 'cleaner', 'power charge', 'car wash', 'flower store', 'restaurant', 'theater', 'bicycle', 'park', 'playground',
            'children playground', 'entertainment', 'bicycle station', 'bus station']


def get_top_similar_classes(vec, classes, n, threshold=0):
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


print("the time used for computing vector and sim is: " + str(time.time() - start))

with open("mapping.txt", "w+") as f:
    for keyword in keywords:
        keyword_vec = get_sentence_vector(keyword, method)
        similar_classes = get_top_similar_classes(keyword_vec, classes, 5)
        print(similar_classes)
        similar_classes_str = ";".join(map(lambda x: x["class"], similar_classes))
        f.write(keyword + ": " + similar_classes_str + "\r\n")
