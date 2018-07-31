#!/usr/bin/env python
import os
import sys
from Semantic_Search.utils.preprocess import load_FastText_model, load_DBOW_model, load_DM_model, load_w2v_model

if __name__ == "__main__":
    global d2v_dm_model
    global d2v_dbow_model
    global fasttext_model
    global w2v_google_model
    global w2v_glove_model

    d2v_dm_model = load_DM_model()
    d2v_dbow_model = load_DBOW_model()
    fasttext_model = load_FastText_model()
    w2v_google_model = load_w2v_model()
    w2v_glove_model = load_w2v_model(model_choice="glove")

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
