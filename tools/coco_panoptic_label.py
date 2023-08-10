import json
import os

"""
     @Date : 2022/5/12 17:57
     @File : coco_panoptic_label.py
     @author : Wu Junjun
     @description : 从coco分割分割annotation中提取部分指定图片的annotation
"""


def get_panoptic_labels(image_path="../data/coco_dataset/images",  #
                        annotation_path="../data/coco/annotations/",
                        save_path="../data/coco_dataset/labels/",
                        label_type="train"):
    name_list = os.listdir(image_path)  # image id list of our images

    anno_file = open(annotation_path + "panoptic_" + label_type + "2017.json")
    panoptic = json.load(anno_file)  # dict

    dic = {}
    dic["info"] = panoptic["info"]
    dic["licenses"] = panoptic["licenses"]
    dic["categories"] = panoptic["categories"]

    img_list = []
    id_list = []
    for img in panoptic["images"]:
        if img["file_name"] in name_list:
            id_list.append(img["id"])
            img_list.append(img)

    anno_list = []
    for anno in panoptic["annotations"]:
        if anno["image_id"] in id_list:
            anno_list.append(anno)

    dic["images"] = img_list
    dic["annotations"] = anno_list

    with open(save_path + "panoptic_" + label_type + "2017.json", "w") as f:
        json.dump(dic, f)


if __name__ == "__main__":
    get_panoptic_labels()
