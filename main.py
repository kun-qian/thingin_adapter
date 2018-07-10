import logging

import requests
import time
from Semantic_Search.DocSimWrapper import get_sentence_vector, vecsim

logging.basicConfig(level=logging.INFO)

try:
    r = requests.get(
        "http://ziggy-dev-ols.nprpaas.ddns.integ.dns-orange.fr/api/v1/ontologies/classes?includeFields=iri%2Cname&limit=10000")
    logging.info("received %s classes from thingin!" % len(r.json()))

except requests.RequestException:
    logging.info("can't get info from thingin!")
    exit()

classes = r.json()

# method could be 1, 2, 3
method = 2
start = time.time()
for c in classes:
    vector = get_sentence_vector(c['data']['name'], method)
    c['data']['vec'] = vector

# print(classes)

keywords = ['lunch', 'dinner', 'break', 'cooling', 'heating', 'sunny']


def get_top_similar_classes(vec, classes, n, threshold=0):
    res = []
    for c in classes:
        class_vec = c['data']['vec']
        sim = vecsim(vec, class_vec)
        if sim > threshold:
            res.append({"class": c['data']['iri'],
                        "name": c['data']['name'],
                        "similarity": sim})

    res.sort(key=lambda x: x['similarity'], reverse=True)
    return res[:n]


print("the time used for computing vector and sim is: " + str(time.time() - start))

with open("mapping.txt", "w+") as f:
    for keyword in keywords:
        keyword_vec = get_sentence_vector(keyword, method)
        similar_classes = get_top_similar_classes(keyword_vec, classes, 5)
        print(similar_classes)
        similar_classes_str = ";".join(map(lambda x: x["name"], similar_classes))
        f.write(keyword + " " + similar_classes_str + "\r\n")
