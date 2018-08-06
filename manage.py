#!/usr/bin/env python
import os
import sys
import logging
import config
from Semantic_Search.utils.preprocess import load_model

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logging.info("loading models...")

    for enabled_method in config.enabled_methods:
        config.models[enabled_method] = load_model(enabled_method)

    logging.info("finish loading models...")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "thingin_recommender.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)
