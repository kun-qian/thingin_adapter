#!usr/bin/env python3
# -*- coding: utf-8 -*-

SEPARATOR = "***"
KEY_VALUE_SEPARATOR = "$$$"

'''
#The old methods 1,2,3, 4, 6, 9 were depreciated
# D2V_DM_NAMES_METHOD = 1
# D2V_DM_COMMENTS_METHOD = 2
# D2V_DBOW_NAMES_METHOD = 3
# D2V_DBOW_COMMENTS_METHOD = 4
# FASTTEXT_NAMES_METHOD = 5
# FASTTEXT_COMMENTS_METHOD = 6
# W2V_GOOGLE_NAMES_METHOD = 7
# W2V_GLOVE_NAMES_METHOD = 9

#WEIGHTED_W2V_FASTTEXT_NAMES_METHOD = 1
#WEIGHTED_W2V_GOOGLE_NAMES_METHOD = 2

'''

# new methods are reordered

USE_NAMES_METHOD = 1
INFERSENT_NAMES_METHOD = 2
GRAN_NAMES_METHOD = 3

inferset_model = None
gran_model = None
use_model = None

methods = {INFERSENT_NAMES_METHOD: 'sent2vec_name_vector_infersent',
           GRAN_NAMES_METHOD: 'sent2vec_name_vector_gran',
           USE_NAMES_METHOD: 'sent2vec_name_vector_use'}

models = {INFERSENT_NAMES_METHOD: inferset_model,
          GRAN_NAMES_METHOD: gran_model,
          USE_NAMES_METHOD: use_model
          }

enabled_methods = [USE_NAMES_METHOD, INFERSENT_NAMES_METHOD, GRAN_NAMES_METHOD]

DEBUG_ALGORITHM_TESTING = True  # code switch for algorithm accuracy testing or for thing'in

if __name__ == "__main__":
    print(models[USE_NAMES_METHOD])
