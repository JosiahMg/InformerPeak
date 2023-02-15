# -*- coding: utf-8 -*-
# Author: Mh
# Date: 2022/11/16 11:11
# Function: SQL语句
from common.log_utils import get_logger
from conf.constant import RESERVE_NODE_TYPE

logger = get_logger(__name__)


class SqlSentencePG:
    @staticmethod
    def get_node_by_proj_id(proj_id, third_proj_id):
        sql = """select gid as device_id, dno from dt_node where projectId='{}' and third_project_id = '{}'
        and dno in {}""".format(proj_id, third_proj_id, RESERVE_NODE_TYPE)
        logger.info('get node type, pg sql: {}'.format(sql))
        return sql


class SqlSentenceTD:
    @staticmethod
    def get_enn_iot_by_id(proj_id, ids, start_dt, end_dt, interval_min):
        """
        :param start_dt: 开始时间 格式 "%Y-%m-%d %H:%M:%S"
        :param during_day: 持续时间 hour
        :param interval: 时间间隔 min
        :param proj_id:
        :return:
        """
        sql = """SELECT
        LAST( pressure ) pressure,
        LAST ( flow ) flow
        FROM
        slsl_dt.enn_iot
        WHERE
        project_id = '{}'
        AND ts BETWEEN '{}' and '{}'
        AND device_id in {}
        INTERVAL ( {}m )
        GROUP BY project_id,device_id""".format(proj_id, start_dt, end_dt, ids, interval_min)
        return sql

    @staticmethod
    def get_all_enn_iot_data_all():
        sql = """SELECT
        ts, pressure, flow, device_id
        FROM
        slsl_dt.enn_iot
        WHERE
        project_id = 'qd_high'
        AND ts BETWEEN '2022-08-23 00:00:00' and '2023-02-10 00:00:00'
        """
        return sql

    @staticmethod
    def get_all_enn_iot_data_lng():
        sql = """SELECT
        ts, pressure, flow, device_id
        FROM
        slsl_dt.enn_iot
        WHERE
        project_id = 'qd_high'
        AND ts BETWEEN '2022-08-23 00:00:00' and '2023-02-10 00:00:00'
        AND device_id in ('100100022', '10012167')
        """
        return sql
