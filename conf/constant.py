# -*- coding: utf-8 -*-
# Author: Mh
# Date: 2022/9/15 19:42
# Function:

# v1版本的dno映射表
DNO_TYPE = {
    1: '管段',
    2: '调压站',  # 调压设备
    3: '阀门',
    4: '凝水缸',
    5: '带气点',
    6: '空节点',
    7: '气源',  # 气源点
    8: '流量计',
    9: '检测点',
    10: '民用户',
    11: '工商户',
    12: '节点',
    13: '阴极保护设备'
}

RESERVE_NODE_TYPE = ('2', '7', '11')

GAS_ID_DICT = {'100100024': '相州门站', '100100020': '古庙门站(城阳)', '100100017': '马店中石化',
               '79': '马店调压站', '100100022': '临港门站LNG', '10012167': '团结路门站LNG'}

USER_ID_DICT = {'12167': '团结路门站出站', '100004': '海尔大道调压站', '1002196': '灵山卫调压站',
                '100100009': '辛屯调压站', '100100028': '临港门站出站至市网', '100100006': '黄山调压计量站',
                '100100019': '红石崖门站', '100100005': '徐村调压站', '100100008': '王台调压站',
                '100100014': '九龙高中压调压站', '100100015': '崔家夼调压站'}

REG_ID_DICT = {'11142': "九龙分输站", '12166': '临港门站进站'}

FEATURE_NAMES = ['100100024_pressure', '100100024_flow', '100100020_pressure', '100100020_flow', '100100017_pressure',
                 '100100017_flow', '79_pressure', '79_flow', '12167_pressure', '12167_flow', '100004_pressure',
                 '100004_flow', '1002196_pressure', '1002196_flow', '100100028_pressure', '100100028_flow',
                 '100100006_pressure', '100100019_pressure', '100100019_flow', '100100005_pressure', '100100005_flow',
                 '100100008_pressure', '100100008_flow', '100100014_pressure', '100100014_flow', '11142_pressure',
                 '11142_flow', '12166_pressure', '12166_flow']
TARGET_NAMES = ['hour_in_wm3', 'hour_out_wm3', '100100022_flow', '10012167_flow']
Y_OFFSET_IDX = 1
LNG_ID_LIST = [lng_id.split('_')[0] for lng_id in TARGET_NAMES[Y_OFFSET_IDX:]]
LNG_MNI_FLOW_HOUR = 500  # lng补气超过500 m3/h则认为是在补气

MODE_TYPE_DT = {
    # 训练每小时供气量预测模型
    "input": {"filename": 'qd_peak_input.csv', "target": ['hour_in_wm3']},
    "output": {"filename": 'qd_peak_output.csv', "target": ['hour_out_wm3']},         # 训练每小时用气量预测模型
    "tuanjielu": {"filename": "qd_peak_tuanjielu.csv", "target": ['10012167_flow']},   # 训练团结路LNG预测模型
    "lingang": {"filename": "qd_peak_lingang.csv", "target": ['100100022_flow']}     # 训练临港LNG预测模型
}

MODE_TYPE = 'input'
MODE_FILENAME = MODE_TYPE_DT[MODE_TYPE]['filename']
MODE_TARGET = MODE_TYPE_DT[MODE_TYPE]['target']
SCALE_FLAG = True   # 是否需要进行标准化

SEQ_LEN = 96
LABEL_LEN = 24
PREDICT_LEN = 48
