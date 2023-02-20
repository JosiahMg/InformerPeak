# -*- coding: utf-8 -*-
# Author: Mh
# Date: 2022/10/11 14:43
# Function: 推送调峰优化算法需要初始值

import json
import traceback
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from api.views.common.date_encoder import DateEncoder
from api.views.common.status import Status, INIT_STATUS, STATUS
from api.service.service_informer_predict import ServiceInformerPredict
from common.log_utils import get_logger

logger = get_logger(__name__)


def post_method_proc(request, context):
    try:
        logger.debug('receive informer post method request')
        params_dict = request.POST
        if request.content_type == 'application/json':
            params_dict = json.loads(request.body if request.body else '{}')
        data = ServiceInformerPredict(params_dict).execute()
        context.update(data)
    except Exception as e:
        message = f'informer预测失败, 错误原因{e}'
        logger.error(traceback.format_exc())
        STATUS[Status.OTHER_ERROR.name]['message'] = message
        context.update(STATUS[Status.OTHER_ERROR.name])

    return HttpResponse(json.dumps(context, cls=DateEncoder, ensure_ascii=False),
                        content_type="application/json; charset=utf-8")


def other_method_proc(request, context):
    logger.debug('optimize push approval index get method request')
    context.update(STATUS[Status.BAD_GET.name])
    return HttpResponse(json.dumps(context, cls=DateEncoder, ensure_ascii=False),
                        content_type="application/json; charset=utf-8")


@csrf_exempt
def push_init_data(request):
    """ 从前推送同济计算使用的边界值 """
    context = INIT_STATUS.copy()
    if request.method == 'POST':
        return post_method_proc(request, context)
    else:
        return other_method_proc(request, context)
