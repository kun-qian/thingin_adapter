"""
WSGI config for thingin_recommender project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.0/howto/deployment/wsgi/
"""

import os
from Semantic_Search.utils.preprocess import load_FastText_model, load_DBOW_model, load_DM_model, load_w2v_model
import logging
import const

from django.core.wsgi import get_wsgi_application

logging.info("loading models...")

const.d2v_dm_model = load_DM_model()
const.d2v_dbow_model = load_DBOW_model()
const.fasttext_model = load_FastText_model()
const.w2v_google_model = load_w2v_model()
const.w2v_glove_model = load_w2v_model(model_choice="glove")
logging.info("finish loading models...")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "thingin_recommender.settings")

application = get_wsgi_application()
