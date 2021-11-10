import json
import os
import re

import numpy as np
import struct
from PIL import Image

def save_image_from_array(save_path, ary, width, height):
    ndary = (np.array(ary)*255).astype(np.uint8)
    image = Image.fromarray(ndary.reshape(width, height))
    image.convert('RGB').save(save_path)

def readDataFromFile(path,save_path):
    with open(path, 'rb') as f:
        dataBytes = f.read()
        # get width and height and aspect
        width_re = rb'NumberOfColumns= (\d+)'
        height_re = rb'NumberOfRows= (\d+)'
        aspect_re = rb'TargetAz= (\d+\.\d+)'
        width_mt = re.search(width_re, dataBytes) 
        height_mt = re.search(height_re, dataBytes)
        target_as = re.search(aspect_re, dataBytes)
        assert width_mt is not None
        assert height_mt is not None
        assert target_as is not None
        width = int(width_mt.group(1))
        height = int(height_mt.group(1))
        aspect = float(target_as.group(1))
        # print(dataBytes)
        # print("image:{}, width:{}, height:{}, aspect:{}".format(path, width, height,aspect))

        # get image array
        image_pre = rb'\[EndofPhoenixHeader\]\n'
        image_pre_mt = re.search(image_pre, dataBytes)
        assert image_pre_mt is not None
        image_begin = image_pre_mt.end()
        image_bytes = dataBytes[image_begin:image_begin+height*width*4]
        image_array = [struct.unpack('>f', image_bytes[i*4:(i+1)*4]) for i in range(height*width)]

        # save image
        savepath = save_path+'.jpg'
        # save_image_from_array(savepath, image_array, height, width)

        #return aspect
        return {
            "path": savepath,
            "aspect": aspect,
        }

if __name__ == '__main__':
    # dataTypes = ['train', 'test']
    # labels = os.listdir('./rawdata/EOC-C/train/')
    # for dataType in dataTypes:
    #     for label in labels:
    #         path = './rawdata/EOC-C/{}/{}'.format(dataType, label)
    #         savepath = './data/EOC-C/{}/{}'.format(dataType, label)
    #         if not os.path.exists(savepath):
    #             os.makedirs(savepath)
    #         jsonpath = './data/EOC-C/{}/{}/aspect.json'.format(dataType, label)
    #
    #         aspect = {}
    #         flie_dir = os.listdir(path)
    #         for i in range(0, len(flie_dir)):
    #             file = flie_dir[i]
    #             if not os.path.isdir(file):
    #                 file_path = os.path.join(path, file)
    #                 file_name = os.path.splitext(file)
    #                 save_path = os.path.join(savepath, file_name[0])
    #                 aspect[file_name[0]] = readDataFromFile(file_path, save_path)
    #             else:
    #                 print('Path error!')
    #
    #         aspect = json.dumps(aspect)
    #         with open(jsonpath, "w", encoding='utf-8') as f:
    #             f.write(aspect)
    #             print("write success")

    labels = os.listdir('./rawdata/EOC-C/test/')
    for label in labels:
        path = './rawdata/EOC-C/test/{}'.format(label)
        savepath = './data/EOC-C/test/{}'.format(label)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        jsonpath = './data/EOC-C/test/{}/aspect.json'.format(label)
        aspect = {}
        flie_dir = os.listdir(path)
        for i in range(0, len(flie_dir)):
            file = flie_dir[i]
            if not os.path.isdir(file):
                file_path = os.path.join(path, file)
                file_name = os.path.splitext(file)
                save_path = os.path.join(savepath, file_name[0])
                aspect[file_name[0]] = readDataFromFile(file_path, save_path)
            else:
                print('Path error!')

        aspect = json.dumps(aspect)
        with open(jsonpath, "w", encoding='utf-8') as f:
            f.write(aspect)
            print("write success")