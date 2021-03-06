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

logging.info("start to load all enabled models... from wsgi.py")

for enabled_method in config.enabled_methods:
    if config.models[enabled_method] is None:
        config.models[enabled_method] = load_model(enabled_method)

logging.info("finish loading all models... in wsgi.py")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "thingin_recommender.settings")

application = get_wsgi_application()
