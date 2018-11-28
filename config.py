#!usr/bin/env python3
# -*- coding: utf-8 -*-

SEPARATOR = "***"
KEY_VALUE_SEPARATOR = "$$$"

#The old methods 1,2,3, 4, 6, 9 were depreciated
# D2V_DM_NAMES_METHOD = 1
# D2V_DM_COMMENTS_METHOD = 2
# D2V_DBOW_NAMES_METHOD = 3
# D2V_DBOW_COMMENTS_METHOD = 4
# FASTTEXT_NAMES_METHOD = 5
# FASTTEXT_COMMENTS_METHOD = 6
# W2V_GOOGLE_NAMES_METHOD = 7
# W2V_GLOVE_NAMES_METHOD = 9

#new methods are reordered
WEIGHTED_W2V_FASTTEXT_NAMES_METHOD = 1
WEIGHTED_W2V_GOOGLE_NAMES_METHOD = 2
INFERSENT_NAMES_METHOD = 3
GRAN_NAMES_METHOD = 4
FASTTEXT_NAMES_METHOD = 5
W2V_GOOGLE_NAMES_METHOD = 6
USE_NAMES_METHOD = 7

fasttext_model = None
w2v_google_model = None
inferset_model = None
gran_model = None
use_model = None


vector_u_fasttext = None
vector_u_google = None


methods = {WEIGHTED_W2V_FASTTEXT_NAMES_METHOD: 'weighted_name_vector_w2v_fasttext',
           WEIGHTED_W2V_GOOGLE_NAMES_METHOD: 'weighted_name_vector_w2v_google',
           INFERSENT_NAMES_METHOD: 'sent2vec_name_vector_infersent',
           GRAN_NAMES_METHOD: 'sent2vec_name_vector_gran',
           FASTTEXT_NAMES_METHOD: 'average_name_vector_w2v_fasttext',
           W2V_GOOGLE_NAMES_METHOD: 'average_name_vector_w2v_google',
           USE_NAMES_METHOD: 'sent2vec_name_vector_use'}

models = {WEIGHTED_W2V_FASTTEXT_NAMES_METHOD: fasttext_model,
          WEIGHTED_W2V_GOOGLE_NAMES_METHOD: w2v_google_model,
          INFERSENT_NAMES_METHOD: inferset_model,
          GRAN_NAMES_METHOD: gran_model,
          FASTTEXT_NAMES_METHOD: fasttext_model,
          W2V_GOOGLE_NAMES_METHOD: w2v_google_model,
          USE_NAMES_METHOD: use_model
          }

vectors_u = {WEIGHTED_W2V_FASTTEXT_NAMES_METHOD: vector_u_fasttext,
             WEIGHTED_W2V_GOOGLE_NAMES_METHOD: vector_u_google
            }

# enabled_methods = [WEIGHTED_W2V_FASTTEXT_NAMES_METHOD, WEIGHTED_W2V_GOOGLE_NAMES_METHOD,
#                    INFERSENT_NAMES_METHOD, GRAN_NAMES_METHOD, USE_NAMES_METHOD]
enabled_methods = [INFERSENT_NAMES_METHOD, GRAN_NAMES_METHOD, USE_NAMES_METHOD]



DEBUG_ALGORITHM_TESTING = True# code switch for algorithm accuracy testing or for thing'in

if __name__ == "__main__":
    print(models[FASTTEXT_NAMES_METHOD])
