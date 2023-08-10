"""
    @Time ： 2022/6/27 9:48
    @Auth ： 吴俊君
    @File ：greyImageSort.py
    @Describe：从eiseg保存的三种标注图中提取出灰度图
"""

import os
import shutil

labelpath = "D:/Desktop/gray/"
imagepath = "D:/Desktop/pennovation_compression/"
dstpath = "D:/Desktop/images/"
imageList = os.listdir(labelpath)

for x in imageList:
    shutil.copyfile(imagepath + x.replace(".png", ".jpg"), dstpath + x.replace(".png", ".jpg"))

# classes = ["leaf debris", "sprinkler", "pipe", "faeces", "trashcan lid"]
# for cls in classes:
#     path = "D:/Desktop/" + cls + "/data/"
#     labelpath = "D:/Desktop/" + cls + "/label/"
#     dstpath = "D:/Desktop/" + cls + "_grey/"
#     imageList = os.listdir(path)
#     grayLsit = []
#
#     for x in imageList:
#         shutil.copyfile(labelpath + x.replace(".jpg", ".png"), dstpath + x.replace(".jpg", ".png"))
