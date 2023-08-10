# -*- coding:utf8 -*-
import cv2
import os
import shutil


def get_frame_from_video(video_dir, video_name, save_dir, interval):
    """
    Args:
        video_name:输入视频名字
        interval: 保存图片的帧率间隔
    Returns:
    """

    # 保存图片的路径
    save_path = video_name.split('.mp4')[0]

    # is_exists = os.path.exists(save_path)
    # if not is_exists:
    #     os.makedirs(save_path)
    #     print('path of %s is build' % save_path)
    # else:
    #     shutil.rmtree(save_path)
    #     os.makedirs(save_path)
    #     print('path of %s already exist and rebuild' % save_path)

    # 开始读视频
    # video_capture = cv2.VideoCapture(os.path.join(os.getcwd(), video_dir, video_name))
    video_capture = cv2.VideoCapture(os.path.join(video_dir, video_name))
    i = 0
    j = 0

    while True:
        success, frame = video_capture.read()
        i += 1
        if success:
            if i % interval == 0:
                # 保存图片
                j += 1
                save_name = save_dir + save_path + str(j) + '_' + str(i) + '.jpg'
                print("save_name", save_name)
                cv2.imwrite(save_name, frame)
                print('image of %s is saved' % save_name)
        if not success:
            print('video is all read')
            break


if __name__ == '__main__':
    # 视频文件名字
    video_dir = 'E:/Data/video_frame/'
    # file_list = os.listdir(os.path.join(os.getcwd(), video_dir))
    file_list = os.listdir(video_dir)

    save_path = "E:/Data/video_frame_result/"
    is_exists = os.path.exists(save_path)
    if not is_exists:
        os.makedirs(save_path)
        print('path of %s is build' % save_path)

    video_list = [file for file in file_list if file.endswith('mp4')]
    for video_name in video_list:
        interval = 90
        get_frame_from_video(video_dir, video_name, save_path, interval)

# -*- coding: utf-8 -*-
# import cv2
# import os
# import numpy as np


# videos_src_path = './video'  # 提取图片的视频文件夹

# # 筛选文件夹下MP4格式的文件
# videos = os.listdir(videos_src_path)  # 用于返回指定的文件夹包含的文件或文件夹的名字的列表。
# videos = filter(lambda x: x.endswith('mp4'), videos)
# dirs = os.listdir(videos_src_path)  # 获取指定路径下的文件


# # 根据名称创建对应的文件夹
# def mkdir(path):
#     folder = os.path.exists(path)
#     if not folder:
#         os.makedirs(path)
#         print(path+"---Done---")
#     else:
#         print(path+"---This is the folder---")


# count = 0

# # 数总帧数
# total_frame = 0

# # 写入txt
# f = "name.txt"
# with open(f, "w+") as file:
#     file.write("-----start-----\n")

# # 循环读取路径下的文件并操作
# for video_name in dirs:
#     # 生成文件名对应的文件夹，并去掉文件格式后缀
#     name = video_name.split('.')
#     name = name[0]
#     outputPath = name
#     print(outputPath)
#     mkdir(outputPath)

#     print("start\n")
#     print(videos_src_path + video_name)
#     vc = cv2.VideoCapture(videos_src_path + video_name)

#     # 初始化,并读取第一帧
#     # rval表示是否成功获取帧
#     # frame是捕获到的图像
#     rval, frame = vc.read()

#     # 获取视频fps
#     fps = vc.get(cv2.CAP_PROP_FPS)
#     # 获取每个视频帧数
#     frame_all = vc.get(cv2.CAP_PROP_FRAME_COUNT)
#     # 获取所有视频总帧数
#     # total_frame+=frame_all

#     print("[INFO] 视频FPS: {}".format(fps))
#     print("[INFO] 视频总帧数: {}".format(frame_all))
#     # print("[INFO] 所有视频总帧: ",total_frame)
#     # print("[INFO] 视频时长: {}s".format(frame_all/fps))

#     # if os.path.exists(outputPath) is False:
#     #     print("[INFO] 创建文件夹,用于保存提取的帧")
#     #     os.mkdir(outputPath)

#     # 每隔n帧保存一张图片
#     frame_interval = 4
#     # 统计当前帧
#     frame_count = 1
#     # count=0

#     while rval:
#         rval, frame = vc.read()
#         # 隔n帧保存一张图片
#         if frame_count % frame_interval == 0:
#             # 当前帧不为None，能读取到图片时
#             if frame is not None:
#                 filename = outputPath + "{}.jpg".format(count)

#                 # 水平、垂直翻转
#                 frame = cv2.flip(frame, 0)
#                 frame = cv2.flip(frame, 1)

#                 # 旋转90°
#                 frame = np.rot90(frame)
#                 cv2.imwrite(filename, frame)
#                 count += 1
#                 print("保存图片:{}".format(filename))
#         frame_count += 1

#     # 将成功抽帧的视频名称写入txt文件，方便检查
#     file = open(f, "a")
#     file.write(video_name+"\n")

#     # 关闭视频文件
#     vc.release()
#     print("[INFO] 总共保存：{}张图片\n".format(count))
