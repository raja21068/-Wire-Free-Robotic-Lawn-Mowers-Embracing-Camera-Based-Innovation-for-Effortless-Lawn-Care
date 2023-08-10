"""
    @Time ： 2022/5/23 9:53
    @Auth ： 吴俊君
    @File ：sample_statistics.py
    @Describe：统计数据集中各样本的数量
"""
import json
import pandas as pd


# def getSampleNumber(path="../../data/coco-subset-15/annotations/instances_train2017.json",
#                     save_path="../../data/coco-subset-15/annotations/sample_train.csv"):
def getSampleNumber(path="../../data/coco_dataset/dataset_v0.0.3/annotations/instances_train2017.json",
                    save_path="../../data/coco_dataset/dataset_v0.0.3/annotations/sample_train.csv"):
    instances_file = open(path)
    instances = json.load(instances_file)
    classnumber = [0 for x in range(0, 16)]
    name = ["person", "bicycle", "motorcycle", "fire_hydrant", "bench",
            "cat", "dog", "backpack", "umbrella", "frisbee",
            "kite", "skateboard", "chair", "potted_plant", "sink", "others"]

    num = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    data = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

    for cls in instances["annotations"]:
        if cls["category_id"] == 1:
            classnumber[0] += 1
            num[0].append(cls["image_id"])
        elif cls["category_id"] == 2:
            classnumber[1] += 1
            num[1].append(cls["image_id"])
        elif cls["category_id"] == 3:
            classnumber[2] += 1
            num[2].append(cls["image_id"])
        elif cls["category_id"] == 4:
            classnumber[3] += 1
            num[3].append(cls["image_id"])
        elif cls["category_id"] == 5:
            classnumber[4] += 1
            num[4].append(cls["image_id"])
        elif cls["category_id"] == 6:
            classnumber[5] += 1
            num[5].append(cls["image_id"])
        elif cls["category_id"] == 7:
            classnumber[6] += 1
            num[6].append(cls["image_id"])
        elif cls["category_id"] == 8:
            classnumber[7] += 1
            num[7].append(cls["image_id"])
        elif cls["category_id"] == 9:
            classnumber[8] += 1
            num[8].append(cls["image_id"])
        elif cls["category_id"] == 10:
            classnumber[9] += 1
            num[9].append(cls["image_id"])
        elif cls["category_id"] == 11:
            classnumber[10] += 1
            num[10].append(cls["image_id"])
        elif cls["category_id"] == 12:
            classnumber[11] += 1
            num[11].append(cls["image_id"])
        elif cls["category_id"] == 13:
            classnumber[12] += 1
            num[12].append(cls["image_id"])
        elif cls["category_id"] == 14:
            classnumber[13] += 1
            num[13].append(cls["image_id"])
        elif cls["category_id"] == 15:
            classnumber[14] += 1
            num[14].append(cls["image_id"])
        else:
            classnumber[15] += 1
            num[15].append(cls["image_id"])

    # for cls in instances["annotations"]:
    #     if cls["category_id"] == 1:
    #         classnumber[0] += 1
    #         num[0].append(cls["image_id"])
    #     elif cls["category_id"] == 2:
    #         classnumber[1] += 1
    #         num[1].append(cls["image_id"])
    #     elif cls["category_id"] == 4:
    #         classnumber[2] += 1
    #         num[2].append(cls["image_id"])
    #     elif cls["category_id"] == 11:
    #         classnumber[3] += 1
    #         num[3].append(cls["image_id"])
    #     elif cls["category_id"] == 15:
    #         classnumber[4] += 1
    #         num[4].append(cls["image_id"])
    #     elif cls["category_id"] == 17:
    #         classnumber[5] += 1
    #         num[5].append(cls["image_id"])
    #     elif cls["category_id"] == 18:
    #         classnumber[6] += 1
    #         num[6].append(cls["image_id"])
    #     elif cls["category_id"] == 27:
    #         classnumber[7] += 1
    #         num[7].append(cls["image_id"])
    #     elif cls["category_id"] == 28:
    #         classnumber[8] += 1
    #         num[8].append(cls["image_id"])
    #     elif cls["category_id"] == 34:
    #         classnumber[9] += 1
    #         num[9].append(cls["image_id"])
    #     elif cls["category_id"] == 38:
    #         classnumber[10] += 1
    #         num[10].append(cls["image_id"])
    #     elif cls["category_id"] == 41:
    #         classnumber[11] += 1
    #         num[11].append(cls["image_id"])
    #     elif cls["category_id"] == 62:
    #         classnumber[12] += 1
    #         num[12].append(cls["image_id"])
    #     elif cls["category_id"] == 64:
    #         classnumber[13] += 1
    #         num[13].append(cls["image_id"])
    #     elif cls["category_id"] == 81:
    #         classnumber[14] += 1
    #         num[14].append(cls["image_id"])
    #     else:
    #         classnumber[15] += 1
    #         num[15].append(cls["image_id"])

    for i in range(0, 16):
        data[i].append(name[i])
        data[i].append(len(num[i]))
        classes = set(num[i])
        data[i].append(len(classes))
        print(data[i])

    columns = ("Class", "Bbox", "Samples")
    save = pd.DataFrame(columns=columns, data=data)
    save.to_csv(save_path, encoding='utf_8_sig', mode="w+")


if __name__ == "__main__":
    getSampleNumber()
