import logging
import os
import pickle
import re

import requests

import config
from Semantic_Search.utils.preprocess import load_model
from config import *

logging.basicConfig(level=logging.INFO)

# os.environ['http_proxy'] = 'http://openwatt-proxy-np.itn.ftgroup:8080'
# os.environ['https_proxy'] = 'http://10.193.250.16:8080/'

# only run onece to download the NLTK data for tokenizer

# only  run onece to download the NLTK data for tokenizer#  import nltk
# nltk.download('punkt')


USE_CACHED_VECTOR = False
CACHE_FILE_BASIC_NAME = 'recommender/cache_files/cached_classes_name_IRI_vector_method_{}.pkl'

logging.info("loading models...")
for method in config.enabled_methods:
    if config.models[method] is None:
        config.models[method] = load_model(method)
logging.info("finish loading models...")

from Semantic_Search.DocSimWrapper import get_sentence_vector


def is_similarity_by_name(method=None):
    return True


def retrieve_all_classes(include_fields):
    if not include_fields or len(include_fields) == 0:
        return None
    # this request can get almost 10000 classes
    try:
        path = "http://ziggy-dev-ols.nprpaas.ddns.integ.dns-orange.fr/api/v1/ontologies/classes?scrollPagination&includeFields="
        for field in include_fields:
            path += (field + "%2C")
        path += "&limit=10000"
        logging.info("the path getting first 10000 classes is: {}".format(path))
        r = requests.get(path, timeout=20)
    except requests.exceptions.RequestException:
        logging.info("IO error when retrieving classes from thingin")
        return None
    except requests.exceptions.ConnectTimeout:
        logging.info("timeout when retrieving classes from thingin")
        return None
    except:
        logging.info("unknown error when retrieving classes from thingin")
        return None
    finally:
        pass

    thingin_classes = r.json()

    # get rest classes
    headers = r.headers
    next_offset = headers.get("x-next-offset", "")
    logging.info("the next offset is: {}".format(next_offset))
    if next_offset != "":

        base = "http://ziggy-dev-ols.nprpaas.ddns.integ.dns-orange.fr/api/v1/ontologies?scrollPagination&offset="
        base += next_offset
        while 1:
            try:
                r = requests.get(base, timeout=20)
            except requests.exceptions.RequestException:
                logging.info("request exception when keep retrieving rest classes")
                break
            except requests.exceptions.ConnectTimeout:
                logging.info("timeout when keep retrieving rest classes")
                break
            except:
                break
            finally:
                pass
            if r.status_code == 200:
                # logging.info("the response is: {}, and type is: {}".format(str(r.json()), type(r.json())))
                if isinstance(r.json(), list) and len(r.json()) > 0:
                    thingin_classes += r.json()
                else:
                    break
            else:
                break
    logging.info("the final thingin classes length is: {}".format(len(thingin_classes)))
    if len(thingin_classes) > 0:
        filtered_keywords = 'command|notification|functionality|function|unit|system|status|state|scheme|fragment|package' \
                            '|specification'
        thingin_classes = [c for c in thingin_classes if
                           len(re.findall(filtered_keywords, c['data']['name'].lower())) == 0]
        return thingin_classes
    return None


if not USE_CACHED_VECTOR:
    classes = retrieve_all_classes(['iri', 'name', 'comment'])
    if not classes or len(classes) == 0:
        logging.info("no classes getted from thingin")
        exit()
    logging.info("the first classes: {}".format(str(classes[0])))
    logging.info("the last classes: {}".format(str(classes[-1])))

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
        # if consider the comments similarity, remove classes without comments
        if not is_similarity_by_name(method):
            classes = [c for c in classes if 'comment' in c['data'].keys()]
            for c in classes:
                c['data']['comment'] = re.sub('\n|\r', ' ', c['data']['comment'][0])
                # print(classes[0])

        else:
            # remove 'comment' field
            for c in classes:
                c['data'].pop('comment', None)

        # remove duplicate iri class
        # print(classes)
        final_classes = [dict(element) for element in tuple({tuple(c['data'].items()) for c in classes})]
        # print(classes)

        data_key = 'name' if is_similarity_by_name() else 'comment'

        for c in final_classes:
            if data_key not in c.keys():
                c['vec'] = None
            else:
                vector = get_sentence_vector(c[data_key], method)
                c['vec'] = vector
        # logging.info("the first 3 classes: {}".format(classes[:3]))
        # write classes into local file for cache
        print('write total {} classes into file: {}'.format(len(final_classes), cache_file))
        with open(cache_file, 'wb') as file:
            pickle.dump(final_classes, file)
