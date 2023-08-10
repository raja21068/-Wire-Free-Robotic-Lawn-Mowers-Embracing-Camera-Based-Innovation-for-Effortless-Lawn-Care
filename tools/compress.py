# %% This script compresses the video using ffmpeg and leaves a empty customize.json
# Author: Jiancong Wang, Laifei Intelligence
# Date: 6.30.2022.
import os
import os.path as osp
from multiprocessing import Pool
import argparse
import glob
import ffmpeg
import json


# This function compresses the video and output the compressed video to a same relative path of the input directory
# Say if there is a video 
# /input_dir/sub1/sub2/video.mp4, this will output the video at 
# /output_dir/sub1/sub2/video.mp4
def compress_video(param):
    video_dir, input_dir, output_dir, frame_rate, resolution, input_ext, output_ext = param
    print(f"Converting video {video_dir}")
    out_video_dir = video_dir.replace(input_dir, output_dir).replace(input_ext, output_ext)
    out_dir = osp.dirname(out_video_dir)
    os.makedirs(out_dir, exist_ok=True)
    command = f"ffmpeg -i '{video_dir}'  -vf \"fps={frame_rate},scale={resolution[0]}:{resolution[1]}\" -vcodec libx265 '{out_video_dir}' "
    os.system(command)


# This function dumps a json file made from parsing the input videos to the output directory
def create_json(in_dir, input_dir, output_dir, input_ext, output_ext):
    if in_dir.endswith("/"):
        in_dir = args.output_dir[:-1]

    videos = list(glob.glob(f"{in_dir}/*.{input_ext}"))
    video = videos[0]

    videos = sorted([osp.basename(v).replace(input_ext, output_ext) for v in videos])
    # The video name is sorted by lexigraphical order. This is easier for the checker to fill one-by-one

    capture_tool, latlng, create_time = get_meta_iphone(video)
    jdict = define_json(capture_tool, latlng, create_time, videos)

    out_dir = in_dir.replace(input_dir, output_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/customize.json", 'w') as f:
        f.write(json.dumps(jdict, indent=4))


# This function extracts meta info from the mov file made from iphone13. I guess it could work for other iphone as well
def get_meta_iphone(video):
    f = ffmpeg.probe(video)

    # this is the date of when the video getting made
    from dateutil import parser as dparser
    time = f['format']['tags']['com.apple.quicktime.creationdate']
    d = dparser.parse(time)
    create_time = d.strftime('%Y-%m-%d %H:%M:%S')

    # this is the location of the video being made in latitude, longitude, altitude
    latlng = f['format']['tags']['com.apple.quicktime.location.ISO6709']  # +37.3632-122.0982+052.682
    latlng = latlng.replace("+", ",+").replace("-", ",-")[1:]  # this will separate the latitude, longitude with ,

    capture_tool = f['format']['tags']['com.apple.quicktime.make'] + ", " + f['format']['tags'][
        'com.apple.quicktime.model']  # iPhone 13

    return capture_tool, latlng, create_time


# This function returns a template json dictionary with some meta information filled in
def define_json(capture_tool, latlng, create_time, videos):
    jdict = {"date": "",
             "author": "",
             "last_modified_by": "",
             "last_modified_time": "",
             "dataset_name": "",
             "version": "0.01",
             "description": "",
             "info": {
                 "capture_tool": capture_tool,
                 "bot_id": "",
                 "expand_params": {},
                 "create_time": create_time,
                 "dataset_type": "mixed",
                 "country": "",
                 "province": "",
                 "latlng": latlng,
                 "season": "",
                 "weather": "",
                 "scene": "",
                 "timeofday": ""
             },
             "images": [
                 {
                     "image_name": ".png",
                     "description": ""
                 }
             ],
             "videos": [{"video_name": v, "description": ""} for v in videos]
             }

    return jdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, help='The directory of overall input videos.',
                        default="E:/Data/vedio_us",
                        # required = True
                        )

    parser.add_argument('-o', '--output_dir', type=str, help='The directory of overall output videos. Say if there is a video at /input_dir/sub1/sub2/video.mp4. \
                        this will output the video at /output_dir/sub1/sub2/video.mp4',
                        default="E:/Data/vedio_frame",
                        # required = True
                        )

    parser.add_argument('--frame_rate', type=int, help='The frame rate of the output video, default 60', default=60)
    parser.add_argument('--resolution', type=tuple, help='The resolution of the output video, default 1920x1080',
                        default=(1280, 720))

    parser.add_argument('--input_ext', type=str, help='The file extension of the input video. Default to MOV',
                        default="MOV")
    parser.add_argument('--output_ext', type=str, help='The file extension of the output video. Default to mp4',
                        default="mp4")

    parser.add_argument('--num_threads', type=int,
                        help='The number of thread for running conversion in parallel. Default 4', default=4)
    args = parser.parse_args()


    if args.input_dir.endswith("/"):
        args.input_dir = args.input_dir[:-1]

    if args.output_dir.endswith("/"):
        args.output_dir = args.output_dir[:-1]

    if args.input_ext.startswith("."):
        args.input_ext = args.input_ext[1:]

    if args.output_ext.startswith("."):
        args.output_ext = args.output_ext[1:]

    all_videos = list(glob.glob(f"{args.input_dir}/**/*.{args.input_ext}", recursive=True))

    # Create json
    all_video_dirs = list(set([osp.dirname(v) for v in all_videos]))
    for v_dir in all_video_dirs:
        create_json(v_dir, args.input_dir, args.output_dir, args.input_ext, args.output_ext)


    # Compress video
    params = [(v, args.input_dir, args.output_dir, args.frame_rate, args.resolution, args.input_ext, args.output_ext)
              for v in all_videos]

    for x in params:
        print(x)
    p = Pool(4)
    p.map(compress_video, params)
    p.close()
    p.join()

