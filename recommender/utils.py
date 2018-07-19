#!usr/bin/env python3
# -*- coding: utf-8 -*-
from thingin_recommender import settings
from const import SEPARATOR, KEY_VALUE_SEPARATOR
import os


def get_recommendations_from_keywords(keywords):
    results = {}
    mapping_file_path = os.path.join(settings.BASE_DIR, 'mapping_fasttext_comments.txt')
    with open(mapping_file_path, 'r') as f:
        lines = f.readlines()
        for word in keywords:
            structured_recommendations = list()
            for line in lines:
                if line.split(KEY_VALUE_SEPARATOR)[0] == word:
                    # print("line: {}".format(line.split(KEY_VALUE_SEPARATOR)[1]))
                    recommendations = line.split(KEY_VALUE_SEPARATOR)[1].split(';')
                    # print("recommendations: {}".format(recommendations))
                    for recommendation in recommendations:
                        items = recommendation.split(SEPARATOR)
                        # print(recommendation)
                        d = {
                            'name': items[0],
                            'class': items[1],
                            'similarity': items[2]
                        }
                        structured_recommendations.append(d)
                    results[word] = structured_recommendations
                    break

    return results


if __name__ == "__main__":
    print(get_recommendations_from_keywords(['lunch', 'dinner']))
