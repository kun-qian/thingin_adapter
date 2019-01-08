import json
from json.decoder import JSONDecodeError

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from config import enabled_methods
from .utils import get_recommendations_from_keywords


@require_POST
@csrf_exempt
def get_recommendations(request):
    if request.method == 'POST':
        res = {}
        body = request.body.decode('utf-8')
        try:
            params = json.loads(body)
        except JSONDecodeError:
            res = {'code': 400,
                   'res': 'illegal request body'}
            return JsonResponse(res)

        keywords = params.get('keywords')
        top_n = params.get('top_n', 3)
        threshold = params.get('threshold', 0)
        method = params.get('method', 5)
        if method not in enabled_methods:
            res['code'] = 401
            res['res'] = 'method {} is not available any more'.format(method)
            return JsonResponse(res)
        if keywords is not None and len(keywords) > 0:
            mapping = get_recommendations_from_keywords(keywords, top_n, threshold, method)
            res['code'] = 200
            res['res'] = mapping
        else:
            res['code'] = 400
            res['res'] = 'no keyword'

        return JsonResponse(res)
