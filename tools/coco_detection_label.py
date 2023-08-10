import json
import os

"""
     @Date : 2022/5/12 17:35
     @File : coco_detection_label.py
     @author : Wu Junjun
     @description : 从coco标准检测annotation中提取部分指定图片的annotation
"""


def get_detection_labels(image_path="../../data/coco_dataset/dataset_v0.0.3/val2017",
                         annotation_path="../../data/coco/annotations/",
                         save_path="../../data/coco_dataset/dataset_v0.0.3/annotations/",
                         label_type="val"):  # train or val
    name_list = os.listdir(image_path)  # image id list of our images

    instances_file = open(annotation_path + "instances_" + label_type + "2017.json")  # coco annotation file
    instances = json.load(instances_file)  # instances label

    keypoints_file = open(annotation_path + "person_keypoints_" + label_type + "2017.json")  # coco annotation file
    keypoints = json.load(keypoints_file)  # keypoints label

    captions_file = open(annotation_path + "captions_" + label_type + "2017.json")  # coco annotation file
    captions = json.load(captions_file)  # captions label

    instances_dict = {}  # instances information
    keypoints_dict = {}  # keypoints information
    captions_dict = {}  # captions information

    # The "info","licenses", and "images" labels are the same in different json
    instances_dict["info"] = instances["info"]
    instances_dict["licenses"] = instances["licenses"]

    keypoints_dict["info"] = keypoints["info"]
    keypoints_dict["licenses"] = keypoints["licenses"]

    captions_dict["info"] = captions["info"]
    captions_dict["licenses"] = captions["licenses"]

    img_list = []  # images of our data
    id_list = []  # id of images

    print("Reading Images...")
    for img in instances["images"]:
        if img["file_name"] in name_list:
            id_list.append(img["id"])
            img_list.append(img)

    instances_dict["images"] = img_list
    keypoints_dict["images"] = img_list
    captions_dict["images"] = img_list

    print("Processing instances...")
    # instances_train2017
    categories = []

    for category in instances["categories"]:  # Map 80 classes to 16 classes
        if category["name"] == "person":
            category["id"] = 1
            categories.append(category)
        elif category["name"] == "bicycle":
            category["id"] = 2
            categories.append(category)
        elif category["name"] == "motorcycle":
            category["id"] = 3
            categories.append(category)
        elif category["name"] == "fire hydrant":
            category["id"] = 4
            categories.append(category)
        elif category["name"] == "bench":
            category["id"] = 5
            categories.append(category)
        elif category["name"] == "cat":
            category["id"] = 6
            categories.append(category)
        elif category["name"] == "dog":
            category["id"] = 7
            categories.append(category)
        elif category["name"] == "backpack":
            category["id"] = 8
            categories.append(category)
        elif category["name"] == "umbrella":
            category["id"] = 9
            categories.append(category)
        elif category["name"] == "frisbee":
            category["id"] = 10
            categories.append(category)
        elif category["name"] == "kite":
            category["id"] = 11
            categories.append(category)
        elif category["name"] == "skateboard":
            category["id"] = 12
            categories.append(category)
        elif category["name"] == "chair":
            category["id"] = 13
            categories.append(category)
        elif category["name"] == "potted plant":
            category["id"] = 14
            categories.append(category)
        elif category["name"] == "sink":
            category["id"] = 15
            categories.append(category)

    category["id"] = 16
    category["supercategory"] = "others"
    category["name"] = "others"
    categories.append(category)



    ins_anno_list = []
    print("Category Mapping...")
    id = 1
    for anno in instances["annotations"]:  # Only 15 classes of annotations are retained and mapped
        if anno["image_id"] in id_list:
            anno["id"] = id
            id += 1
            print("Processing the annotation for %d " % id)
            if anno["category_id"] == 1:
                anno["category_id"] = 1
                ins_anno_list.append(anno)
            elif anno["category_id"] == 2:
                anno["category_id"] = 2
                ins_anno_list.append(anno)
            elif anno["category_id"] == 4:
                anno["category_id"] = 3
                ins_anno_list.append(anno)
            elif anno["category_id"] == 11:
                anno["category_id"] = 4
                ins_anno_list.append(anno)
            elif anno["category_id"] == 15:
                anno["category_id"] = 5
                ins_anno_list.append(anno)
            elif anno["category_id"] == 17:
                anno["category_id"] = 6
                ins_anno_list.append(anno)
            elif anno["category_id"] == 18:
                anno["category_id"] = 7
                ins_anno_list.append(anno)
            elif anno["category_id"] == 27:
                anno["category_id"] = 8
                ins_anno_list.append(anno)
            elif anno["category_id"] == 28:
                anno["category_id"] = 9
                ins_anno_list.append(anno)
            elif anno["category_id"] == 34:
                anno["category_id"] = 10
                ins_anno_list.append(anno)
            elif anno["category_id"] == 38:
                anno["category_id"] = 11
                ins_anno_list.append(anno)
            elif anno["category_id"] == 41:
                anno["category_id"] = 12
                ins_anno_list.append(anno)
            elif anno["category_id"] == 62:
                anno["category_id"] = 13
                ins_anno_list.append(anno)
            elif anno["category_id"] == 64:
                anno["category_id"] = 14
                ins_anno_list.append(anno)
            elif anno["category_id"] == 81:
                anno["category_id"] = 15
                ins_anno_list.append(anno)
            else:
                anno["category_id"] = 16
                ins_anno_list.append(anno)

    instances_dict["annotations"] = ins_anno_list

    print("Processing person_keypoints...")
    # person_keypoints_train2017
    key_anno_list = []
    for anno in keypoints["annotations"]:
        if anno["image_id"] in id_list:
            key_anno_list.append(anno)
    keypoints_dict["annotations"] = key_anno_list
    keypoints_dict["categories"] = keypoints["categories"]  # only person

    print("Processing captions...")
    # captions_train2017
    cap_anno_list = []
    for anno in captions["annotations"]:
        if anno["image_id"] in id_list:
            cap_anno_list.append(anno)
    captions_dict["annotations"] = cap_anno_list

    print("Saving annotation...")
    # Save the extracted JSON file
    with open(save_path + "instances_" + label_type + "2017.json", "w") as f:
        json.dump(instances_dict, f)
    with open(save_path + "person_keypoints_" + label_type + "2017.json", "w") as f:
        json.dump(keypoints_dict, f)
    with open(save_path + "captions_" + label_type + "2017.json", "w") as f:
        json.dump(captions_dict, f)


if __name__ == "__main__":
    get_detection_labels()
