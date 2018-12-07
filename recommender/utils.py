#!usr/bin/env python3
# -*- coding: utf-8 -*-
from thingin_recommender import settings
from config import SEPARATOR, KEY_VALUE_SEPARATOR
import os
import sys
import pickle
import logging

logging.basicConfig(level=logging.INFO)
from config import methods
from Semantic_Search.DocSimWrapper import vecsim, get_sentence_vector

CACHE_FILE_BASIC_NAME = 'cached_classes_name_IRI_vector_method_{}.pkl'


def get_recommendations_from_keywords2(keywords, top_n, threshold, method):
    results = {}
    mapping_file_path = os.path.join(settings.BASE_DIR, "mapping_{}.txt".format(methods[method]))
    with open(mapping_file_path, 'r') as f:
        lines = f.readlines()
        for word in keywords:
            structured_recommendations = list()
            for line in lines:
                if line.split(KEY_VALUE_SEPARATOR)[0] == word:
                    recommendations = line.split(KEY_VALUE_SEPARATOR)[1].split(';')
                    for recommendation in recommendations:
                        items = recommendation.split(SEPARATOR)
                        if float(items[2]) >= float(threshold) and len(structured_recommendations) < int(top_n):
                            d = {
                                'name': items[0],
                                'class': items[1],
                                'similarity': items[2]
                            }
                            structured_recommendations.append(d)
                    results[word] = structured_recommendations
                    break
    return results


def get_recommendations_from_keywords(keywords, top_n, threshold, method):
    d = dict()
    cache_file = CACHE_FILE_BASIC_NAME.format(methods[method])
    if os.path.exists(cache_file):
        logging.info("opening the cache file")
        try:
            with open(cache_file, 'rb') as file:
                classes = pickle.load(file)
            for keyword in keywords:
                keyword_vec = get_sentence_vector(keyword, method)
                d[keyword] = get_top_n_similar_classes(keyword_vec, classes, top_n, threshold)
            return d
        except IOError:
            pass
        except:
            # should not happen, just in case
            logging.info("unexpected error when reading cache file: {}".format(sys.exc_info()[0]))
    else:
        logging.info("the cache file does not exist")
        return d


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


if __name__ == "__main__":
    print(get_recommendations_from_keywords(['lunch', 'dinner']))
