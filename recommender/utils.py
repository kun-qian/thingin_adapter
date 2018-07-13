#!usr/bin/env python3
# -*- coding: utf-8 -*-
from thingin_recommender import settings
import os


def get_recommendations_from_keywords(keywords):
    results = {}
    mapping_file_path = os.path.join(settings.BASE_DIR, 'mapping.txt')
    with open(mapping_file_path, 'r') as f:
        lines = f.readlines()
        for word in keywords:
            for line in lines:
                if line.split()[0] == word:
                    recommendations = line.split()[1].split(';')
                    results[word] = recommendations
                    break

    return results


if __name__ == "__main__":
    print(get_recommendations_from_keywords(['lunch', 'dinner']))
