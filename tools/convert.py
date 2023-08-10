import os

ori_dir = 'E:/Data/vedio_us'
saved_dir = 'E:/Data/vedio_frame'

video_list = os.listdir(os.path.join(os.getcwd(), ori_dir))
# print(video_list)
for video_dir in video_list:
    videos = os.listdir(os.path.join(os.getcwd(), ori_dir, video_dir))
    # print(videos)
    for video in videos:
        print(os.path.join(os.getcwd(), ori_dir, video_dir, video))
        print(os.path.join(os.getcwd(), saved_dir, video_dir, video[:-3] + 'mp4'))
        os.system('ffmpeg -i ' + os.path.join(os.getcwd(), ori_dir, video_dir, video) + ' -b:v 9600k -s 1920x1080 ' + os.path.join(os.getcwd(), saved_dir, video_dir, video[:-3] + 'mp4'))

