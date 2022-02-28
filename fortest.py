import itertools
import json
import os

L = 4
theta = 45

def get_target_value(key, dic, tmp_list):
    if not isinstance(dic, dict) or not isinstance(tmp_list, list):  # 对传入数据进行格式校验
        return 'argv[1] not an dict or argv[-1] not an list '

    if key in dic.keys():
        tmp_list.append(dic[key])  #传入数据存在则存入tmp_list

    for value in dic.values():  #传入数据不符合则对其value值进行遍历
        if isinstance(value, dict):
            get_target_value(key, value, tmp_list)  #传入数据的value值是字典，则直接调用自身
        elif isinstance(value, (list, tuple)):
            _get_value(key, value, tmp_list)  #传入数据的value值是列表或者元组，则调用_get_value

    return tmp_list

def _get_value(key, val, tmp_list):
    for val_ in val:
        if isinstance(val_, dict):
            get_target_value(key, val_, tmp_list)  #传入数据的value值是字典，则调用get_target_value
        elif isinstance(val_, (list, tuple)):
            _get_value(key, val_, tmp_list)   #传入数据的value值是列表或者元组，则调用自身

if __name__ == '__main__':
    jsonpath = './data/SOC/train/2S1/aspect.json'
    savepath = './data/SOC/train/2S1/sequence.json'

    with open(jsonpath, 'r', encoding='utf-8') as f:
        aspect_data = json.load(f)
    aspect_list = get_target_value('aspect', aspect_data, [])
    path_list = get_target_value('path', aspect_data, [])

    path_aspect = zip(path_list,aspect_list)
    sorted_path_aspect = sorted(path_aspect, key=lambda x: x[1])
    result = zip(*sorted_path_aspect)
    sorted_path, sorted_aspect = [list(x) for x in result]

    print(sorted_path)
    print(sorted_aspect)