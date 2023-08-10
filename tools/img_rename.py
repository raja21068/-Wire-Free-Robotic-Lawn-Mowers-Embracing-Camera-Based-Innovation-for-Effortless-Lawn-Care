"""
    @Time ： 2022/7/13 16:19
    @Auth ： 吴俊君
    @File ：img_rename.py
    @Describe：整合多个文件夹的图片并合并
"""
import os
import shutil

root = "D:/Desktop/pick_video/"
dst = "D:/Desktop/UESTC_huzhou"
dir_list = os.listdir(root)

is_exits = os.path.exists(dst)
if not is_exits:
    os.makedirs(dst)
index = 1
for dir in dir_list:
    path = os.path.join(root, dir)
    file_list = os.listdir(path)
    for file in file_list:
        filename = "UESTC_huzhou_" + str(index) + ".jpg"
        index += 1
        file_path = os.path.join(path, file)
        # print(file_path + "  " + os.path.join(dst, filename))
        shutil.copyfile(file_path, os.path.join(dst, filename))
