# -*- coding: utf-8 -*-
# Author: Lx
# Date: 2021/3/16 18:20

from django.urls import path
from django.http import HttpResponse
from api.views import informer_predict


def index(request):
    return HttpResponse("Hello, world. You're at Peak Leak by Informer.")


urlpatterns = [
    path('', index, name='index'),
    path('optimization-model/push-optimize-init-data', informer_predict.push_init_data),
]
