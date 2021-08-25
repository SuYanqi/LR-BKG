"""json文件输入输出"""
import json
import pickle
import re

from datetime import datetime


# datetime无法写入json文件，用这个处理一下
class CJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        # convert the ISO8601 string to a datetime object
        converted = datetime.datetime.strptime(obj.value, "%Y%m%dT%H:%M:%S")
        if isinstance(converted, datetime.datetime):
            return converted.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(converted, datetime.date):
            return converted.strftime("%Y-%m-%d")
        else:
            return json.JSONEncoder.default(self, converted)


class FileUtil:
    @staticmethod
    # 从文件中取数据
    def load_json(filepath):
        with open(filepath, 'r') as load_f:
            data_list = json.load(load_f)
        return data_list

    @staticmethod
    def dump_json(filepath, data_list):
        with open(filepath, 'w') as f:
            # json.dump(data_list, f, cls=CJsonEncoder)
            json.dump(data_list, f)

    @staticmethod
    def load_pickle(filepath):
        # load功能
        # load 从数据文件中读取数据，并转换为python的数据结构
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def dump_pickle(filepath, data):
        # dump功能
        # dump 将数据通过特殊的形式转换为只有python语言认识的字符串，并写入文件
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

# 文件名不能包含一些特殊字符，因此需要特殊处理
def filename(class_name, pro_name, com_name):
    classification = re.sub(r'[<>/\\|:\"*?]+', '-', class_name)
    product = re.sub(r'[<>/\\|:\"*?]+', '-', pro_name)
    component = re.sub(r'[<>/\\|:\"*?]+', '-', com_name)
    return classification, product, component
