#!usr/bin/env python3
# -*- coding: utf-8 -*-

SEPARATOR = "***"
KEY_VALUE_SEPARATOR = "$$$"

D2V_DM_NAMES_METHOD = 1
D2V_DM_COMMENTS_METHOD = 2
D2V_DBOW_NAMES_METHOD = 3
D2V_DBOW_COMMENTS_METHOD = 4
FASTTEXT_NAMES_METHOD = 5
FASTTEXT_COMMENTS_METHOD = 6
W2V_GOOGLE_NAMES_METHOD = 7
W2V_GLOVE_NAMES_METHOD = 9

global d2v_dm_model
global d2v_dbow_model
global fasttext_model
global w2v_google_model
global w2v_glove_model

methods = {D2V_DM_NAMES_METHOD: 'd2v_dm_names',
           D2V_DM_COMMENTS_METHOD: 'd2v_dm_comments',
           D2V_DBOW_NAMES_METHOD: 'd2v_dbow_names',
           D2V_DBOW_COMMENTS_METHOD: 'd2v_dbow_comments',
           FASTTEXT_NAMES_METHOD: 'fasttext_names',
           FASTTEXT_COMMENTS_METHOD: 'fasttext_comments',
           W2V_GOOGLE_NAMES_METHOD: 'w2v_google_names',
           W2V_GLOVE_NAMES_METHOD: 'w2v_glove_names'}

models = {D2V_DM_NAMES_METHOD: d2v_dm_model,
          D2V_DM_COMMENTS_METHOD: d2v_dm_model,
          D2V_DBOW_NAMES_METHOD: d2v_dbow_model,
          D2V_DBOW_COMMENTS_METHOD: d2v_dbow_model,
          FASTTEXT_NAMES_METHOD: fasttext_model,
          FASTTEXT_COMMENTS_METHOD: fasttext_model,
          W2V_GOOGLE_NAMES_METHOD: w2v_google_model,
          W2V_GLOVE_NAMES_METHOD: w2v_glove_model
          }

enabled_methods = [FASTTEXT_NAMES_METHOD, W2V_GOOGLE_NAMES_METHOD]
