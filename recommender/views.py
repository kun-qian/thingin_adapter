from django.shortcuts import render

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
from json.decoder import JSONDecodeError
from .utils import get_recommendations_from_keywords


@require_POST
@csrf_exempt
def get_recommendations(request):
    if request.method == 'POST':
        res = {}
        body = request.body
        try:
            params = json.loads(body)
        except JSONDecodeError:
            res = {'code': 400,
                   'res': 'illegal request body'}
            return JsonResponse(res)

        keywords = params.get('keywords')
        if keywords is not None and len(keywords) > 0:
            mapping = get_recommendations_from_keywords(keywords)
            res['code'] = 200
            res['res'] = mapping
        else:
            res['code'] = '400'
            res['res'] = 'no keyword'

        return JsonResponse(res)
