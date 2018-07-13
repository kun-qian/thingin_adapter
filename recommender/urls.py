#!usr/bin/env python3
# -*- coding: utf-8 -*-

from django.urls import path

from . import views

urlpatterns = [
    path('', views.get_recommendations, name='get_recommendations'),
]
