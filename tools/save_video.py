"""
    @Time ： 2022/7/5 21:17
    @Auth ： 吴俊君
    @File ：save_video.py
    @Describe：保存MX307的视频流
"""

import ffmpeg

# camera = 'rtsp://admin:1234abcd@192.168.1.64/h264/ch1/main/av_stream'
# probe = ffmpeg.probe(camera)

index=4

ffmpeg. \
    input("rtsp://192.168.1.10/stream_chn0.h264").output("D:/Desktop/video/video"+str(index)+".mp4").overwrite_output().run(capture_stdout=True)
    # input("rtsp://192.168.1.10/stream_chn0.h264").output("D:/Desktop/video/video2.mp4").overwrite_output().run(capture_stdout=True)
