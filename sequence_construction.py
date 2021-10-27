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

def sequenceConstruction(path_list,aspect_list):
    file_number = len(path_list)

    j=0
    sequence=[]
    tmp=[0]*L
    for i in range(0, file_number):
        if i < file_number-1-L:
            delta1 = abs(aspect_list[i]-aspect_list[i+L-1])
            delta2 = abs(aspect_list[i]-aspect_list[i+L])
            if delta2 < theta:
                tmp2=[0]*(L+1)
                for k in range(0, L+1):
                    tmp2[k]=path_list[i+k]
                group=list(itertools.combinations(tmp2, L))
                for k in range(0,len(group)-1):
                    sequence.append(group[k])
            elif delta1 < theta:
                for k in range(0, L):
                    tmp[k]=path_list[i+k]
                sequence.append(tmp)
            else:
                continue
        else:
            delta3 = abs(aspect_list[file_number-1-L]-aspect_list[file_number-1])
            if delta3 < theta:
                for k in range(0, L):
                    tmp[k]=path_list[i+k+1]
                sequence.append(tmp)
            else:
                break
            break
    return sequence


if __name__ == '__main__':
    dataTypes = ['train', 'test']
    labels = os.listdir('./data/EOC-D/train/')
    for dataType in dataTypes:
        for label in labels:
            jsonpath='./data/EOC-D/{}/{}/aspect.json'.format(dataType, label)
            savepath='./data/EOC-D/{}/{}/sequence.json'.format(dataType, label)

            with open(jsonpath, 'r', encoding = 'utf-8') as f:
                aspect_data = json.load(f)
            aspect_list = get_target_value('aspect', aspect_data, [])
            path_list = get_target_value('path', aspect_data, [])

            sequence=sequenceConstruction(path_list,aspect_list)
            print(len(sequence))

            with open(savepath, "w") as f:
                json.dump(sequence,f)
                print("write success")