"""
WSGI config for thingin_recommender project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.0/howto/deployment/wsgi/
"""

import os
from Semantic_Search.utils.preprocess import load_model
import logging
import config

from django.core.wsgi import get_wsgi_application

logging.info("loading models...")

for enabled_method in config.enabled_methods:
    config.models[enabled_method] = load_model(enabled_method)

logging.info("finish loading models...")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "thingin_recommender.settings")

application = get_wsgi_application()
