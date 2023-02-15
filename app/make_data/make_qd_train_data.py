# -*- coding: utf-8 -*-
# Author: Mh
# Date: 2023/2/9 16:31
# Function: 从青岛enn_iot表读取数据
import logging
import os.path
from datetime import datetime

import numpy as np
import pandas as pd

from app.make_data.sql_sentence import SqlSentencePG, SqlSentenceTD
from common.log_utils import get_logger
from conf.constant import *
from conf.path_config import resource_dir
from database.operate_tdengine import OperateTD
from database.postgresql import PostgreOp

logger = get_logger(__name__)


class CreateTrainData:
    def __init__(self, start_dt=None, end_dt=None, proj_id='qd_high', interval_ft=60,
                 interval_target=5, third_proj_id='3d0840da3d0d450e9b54d14be8e50055'):
        """
        interval_ft: 提取特征使用的时间间隔
        interval_qt： 提取每小时供气和用量量 单位万方
        """
        self.enn_iot_ft = pd.DataFrame([])
        self.enn_iot_target = pd.DataFrame([])
        self.node_type_df = pd.DataFrame([])
        self.td_op = OperateTD()
        self.pg_op = PostgreOp()
        if start_dt is None:
            self.start_dt = self.create_start_dt()
        else:
            self.start_dt = start_dt
        if end_dt is None:
            self.end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            self.end_dt = end_dt
        self.proj_id = proj_id
        self.third_proj_id = third_proj_id
        self.interval_ft_min = interval_ft
        self.interval_target_min = interval_target

    @staticmethod
    def create_start_dt():
        train_file = os.path.join(resource_dir, 'feature.csv')
        if os.path.exists(train_file):
            train_df = pd.read_csv(train_file, usecols=['ts'])
            start_dt = train_df[-1, 'ts'].values
        else:
            logging.error("you must give a start time")
            raise Exception("必须指定一个起始时间")
        return start_dt

    def create_enn_iot_data(self):
        if self.interval_ft_min == self.interval_target_min:
            self.enn_iot_ft = self.load_enn_iot_data_td(self.interval_ft_min)
            self.enn_iot_target = self.enn_iot_ft
        else:
            self.enn_iot_ft = self.load_enn_iot_data_td(self.interval_ft_min)
            self.enn_iot_target = self.load_enn_iot_data_td(self.interval_target_min)

        self.enn_iot_ft = self.enn_iot_ft.groupby('device_id').apply(self.post_process_enn_iot_data,
                                                                     (self.interval_ft_min))
        self.enn_iot_target = self.enn_iot_target.groupby('device_id').apply(self.post_process_enn_iot_data,
                                                                      (self.interval_target_min))
        logger.info("create enn iot data success")

    def post_process_enn_iot_data(self, sub_df, interval_min):
        """ 填充为空的压力或者流量数据 """
        ts = pd.date_range(self.start_dt, self.end_dt, freq=f"{interval_min}min", inclusive='left')
        df_ts = pd.DataFrame({'ts': ts})
        total_count = df_ts.shape[0]
        sub_df = pd.merge(df_ts, sub_df, on='ts', how='left')
        device_id = sub_df['device_id'].dropna().unique()[0]
        p_nan_count = sub_df['pressure'].isna().sum()
        q_nan_count = sub_df['flow'].isna().sum()
        if p_nan_count > 0:
            logger.warning(f"device id: {device_id} exist null pressure count: {p_nan_count}/{total_count}")
        if q_nan_count > 0:
            logger.warning(f"device id: {device_id} exist null flow count: {q_nan_count}/{total_count}")
        sub_df.fillna(method='ffill', inplace=True)
        sub_df.fillna(method='bfill', inplace=True)
        return sub_df

    def load_enn_iot_data_td(self, interval_min):
        """ 从td库中读取self.ids的所有从时间self.start_dt到self.end_dt指定间隔的数据
        """
        ids = tuple(self.node_type_df['device_id'].tolist())
        sql = SqlSentenceTD.get_enn_iot_by_id(self.proj_id, ids, self.start_dt, self.end_dt, interval_min)
        enn_iot_df = self.td_op.query_df(sql)
        enn_iot_df = pd.merge(enn_iot_df, self.node_type_df, on='device_id', how='left')
        return enn_iot_df

    def load_node_type(self):
        sql = SqlSentencePG.get_node_by_proj_id(self.proj_id, self.third_proj_id)
        data = self.pg_op.query(sql)
        data_df = pd.DataFrame(data)
        if data_df.empty:
            logger.error(f"get project: {self.proj_id} node type failed from postgresql")
            raise Exception(f"从PG库获取项目: {self.proj_id} 点表数据类型失败")
        data_df['device_id'] = data_df['device_id'].astype(str)
        self.node_type_df = data_df
        logger.info("load node type success")

    def make_target_data_hour(self, df_hour):
        # 计算每小时的总供气和总用气
        feature_dt = {}
        ct = 60 / self.interval_target_min  # 每小时多少个点
        gas_condition = (df_hour['dno'] == 7) & (~df_hour['device_id'].isin(LNG_ID_LIST))
        user_condition = df_hour['dno'] == 11
        gas_quantity = df_hour[gas_condition]['flow'].sum() / ct / 10000  # 气源供气 万方
        user_quantity = df_hour[user_condition]['flow'].sum() / ct / 10000  # 用户用气 万方

        feature_dt.update({TARGET_NAMES[0]: gas_quantity, TARGET_NAMES[1]: user_quantity})
        # 计算LNG补气信息
        lng_df = df_hour[df_hour['device_id'].isin(LNG_ID_LIST)].copy()
        lng_df.loc[lng_df['flow'] <= LNG_MNI_FLOW_HOUR, 'flow'] = 0
        for i, lng_id in enumerate(LNG_ID_LIST, OFFSET_IDX):
            lng_df = df_hour[df_hour['device_id'] == lng_id]
            apply_count = (lng_df['flow'] != 0).sum()
            if apply_count > ct / 3:  # 如果1小时内存在1/3以上的补气则认为这一个小时是在补气
                flow_hour = lng_df[lng_df['flow'] != 0]['flow'].mean()
                feature_dt.update({TARGET_NAMES[i]: flow_hour})
            else:
                feature_dt.update({TARGET_NAMES[i]: 0})

        feature_df = pd.DataFrame([feature_dt])
        return feature_df

    @staticmethod
    def make_feature_hour(df_hour):
        feature_dt = {}
        for ft_name in FEATURE_NAMES:
            gid, type = ft_name.split('_')
            value = df_hour.loc[df_hour['device_id'] == gid, type]
            if value.empty:
                feature_dt.update({ft_name: np.nan})
            else:
                feature_dt.update({ft_name: value.values[0]})
        feature_df = pd.DataFrame([feature_dt])
        return feature_df

    def make_x_feature(self):
        ft_df = self.enn_iot_ft.groupby('ts').apply(self.make_feature_hour).droplevel(1).reset_index()
        logger.info("make x feature success")
        return ft_df

    def make_target_data(self):
        target_df = self.enn_iot_target.groupby(pd.Grouper(key='ts', freq='1h')).apply(self.make_target_data_hour)
        target_df = target_df.droplevel(1).reset_index()
        logger.info("make target feature success")
        return target_df

    def make_feature(self):
        ft_df = self.make_x_feature()
        target_df = self.make_target_data()
        feature_df = pd.merge(ft_df, target_df, on='ts', how='inner')
        feature_df = feature_df[['ts'] + FEATURE_NAMES + TARGET_NAMES]
        feature_df.to_csv(os.path.join(resource_dir, f'feature_{self.start_dt[:10]}.csv'), index=False)
        logger.info(f"save feature dataframe into {resource_dir}")
        return feature_df

    def execute(self):
        self.load_node_type()
        self.create_enn_iot_data()
        self.make_feature()


if __name__ == "__main__":
    CreateTrainData(start_dt="2023-02-01 00:00:00", end_dt="2023-02-13 00:00:00").execute()
