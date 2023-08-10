"""
    @Time ： 2022/5/13 15:11
    @Auth ： 吴俊君
    @File ：coco_to_yolo_format.py
    @Describe：将coco标注json格式文件转化为YOLO标注txt格式文件
"""

import json
import argparse
import numpy as np


# json中bbox的格式是top_x, top_y, w, h
# 转成Yolo的格式是cen_x, cen_y, w, h的相对位置
def convert(img_size, box):
    dw = 1. / (img_size[0])
    dh = 1. / (img_size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def main(args):
    # 读取 json 文件数据
    with open(args.json_file, 'r') as load_f:
        content = json.load(load_f)

    img_hw = dict()
    for k in content['images']:
        file_name = k['id']
        # file_name = k['file_name'].rsplit('.',1)[0]
        img_hw[file_name] = [k["width"], k["height"], k["file_name"].rsplit('.', 1)[0]]
    # 循环处理
    for t in content['annotations']:
        tmp = t['image_id']
        filename = args.output + img_hw[tmp][2] + '.txt'

        width = img_hw[tmp][0]
        height = img_hw[tmp][1]

        # 计算 yolo 数据格式所需要的中心点的 相对 x, y 坐标, w,h 的值
        bbox = convert((width, height), t['bbox'])
        box = np.array(bbox, dtype=np.float64)

        # categroy_id = t['category_id'] - 1
        categroy_id = t['category_id']

        if box[2] > 0 and box[3] > 0:
            line = tuple([categroy_id] + box.tolist())

            with open(filename, mode='a') as fp:
                fp.write(('%g ' * len(line)).rstrip() % line + '\n')

        else:
            print('bbox error')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start convert.')
    parser.add_argument('--json_file',
                        default="D:/Desktop/det_v0.02/train/labels.json", type=str,
                        help='json file path')  # json文件路径
    parser.add_argument('--output', default="D:/Desktop/det_yolo_v0.02/labels/train/", type=str,
                        help='output path')  # 输出的 txt 文件路径
    args = parser.parse_args()
    main(args)
