"""
    @Time ： 2022/5/10 15:33
    @Auth ： 吴俊君
    @File ：yolo_to_coco_format.py
    @Describe：将YOLO标注txt格式文件转化为coco标注json格式文件
"""
import os
import json
from PIL import Image


def yolo_to_coco(save_path="../data/coco_dataset/json_labels",  # json格式文件保存位置
                 image_path="../data/coco_dataset/images",  # 图片位置
                 classes_path="military_object.names",  # coco类别文件位置
                 annotation_path="../data/coco_dataset/yolo_labels"):  # YOLO格式文件位置

    with open(classes_path, 'r') as fr:  # 打开并读取类别文件
        classes = fr.readlines()

    categories = []  # 存储类别的列表
    for j, label in enumerate(classes):
        label = label.strip()
        categories.append({'id': j + 1, 'name': label, 'supercategory': 'None'})  # 将类别信息添加到categories中

    write_json_context = dict()  # 写入.json文件的大字典
    write_json_context['info'] = {'description': '', 'url': '', 'version': '', 'year': 2022, 'contributor': '',
                                  'date_created': '2022-05-10'}
    write_json_context['licenses'] = [{'id': 1, 'name': None, 'url': None}]
    write_json_context['categories'] = categories
    write_json_context['images'] = []
    write_json_context['annotations'] = []

    imageFileList = []
    annotationsFileList = os.listdir(annotation_path)
    for s in annotationsFileList:
        imageFileList.append(s.replace("txt", "jpg"))

    for i, imageFile in enumerate(imageFileList):
        imagePath = os.path.join(image_path, imageFile)  # 获取图片的绝对路径
        image = Image.open(imagePath)  # 读取图片，然后获取图片的宽和高
        W, H = image.size

        img_context = {}  # 使用一个字典存储该图片信息
        img_context['file_name'] = imageFile
        img_context['height'] = H
        img_context['width'] = W
        img_context['date_captured'] = '2022-05-10'
        img_context['id'] = i
        img_context['license'] = 1
        img_context['color_url'] = ''
        img_context['flickr_url'] = ''
        write_json_context['images'].append(img_context)  # 将该图片信息添加到'image'列表中

        txtFile = annotationsFileList[i]  # 获取该图片获取的txt文件
        # print(imageFile, txtFile, "\n")
        with open(os.path.join(annotation_path, txtFile), 'r') as fr:
            lines = fr.readlines()  # 读取txt文件的每一行数据，lines是一个列表，包含了一个图片的所有标注信息
        for j, line in enumerate(lines):
            bbox_dict = {}  # 将每一个bounding box信息存储在该字典中
            # line = line.strip().split()
            # print(line.strip().split(' '))

            class_id, x, y, w, h = line.strip().split(' ')  # 获取每一个标注框的详细信息
            class_id, x, y, w, h = int(class_id), float(x), float(y), float(w), float(h)  # 将字符串类型转为可计算的int和float类型

            xmin = (x - w / 2) * W  # 坐标转换
            ymin = (y - h / 2) * H
            xmax = (x + w / 2) * W
            ymax = (y + h / 2) * H
            w = w * W
            h = h * H

            bbox_dict['id'] = i * 10000 + j  # bounding box的坐标信息
            bbox_dict['image_id'] = i
            bbox_dict['category_id'] = class_id + 1  # 注意目标类别要加一
            bbox_dict['iscrowd'] = 0
            height, width = abs(ymax - ymin), abs(xmax - xmin)
            bbox_dict['area'] = height * width
            bbox_dict['bbox'] = [xmin, ymin, w, h]
            bbox_dict['segmentation'] = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]
            write_json_context['annotations'].append(bbox_dict)  # 将每一个由字典存储的bounding box信息添加到'annotations'列表中

    name = os.path.join(save_path, "train" + '.json')
    with open(name, 'w') as fw:  # 将字典信息写入.json文件中
        json.dump(write_json_context, fw, indent=2)

if __name__ == "__main__":
    yolo_to_coco()