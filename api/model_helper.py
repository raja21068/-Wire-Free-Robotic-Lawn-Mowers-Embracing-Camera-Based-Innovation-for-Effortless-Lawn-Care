"""
Helper to call the trained MMDet and MMSeg models.
"""
import os

import torch
# MMDetection - https://github.com/open-mmlab/mmdetection/blob/master/demo/inference_demo.ipynb
from mmdet.apis import init_detector, inference_detector
# MMSegmetation - https://github.com/open-mmlab/mmsegmentation/blob/master/demo/inference_demo.ipynb
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import mmcv

import model_configs


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)


def detect_yolo(image, model_path):
    """
    Call the YOLO trained model for object detection.
    :param image: one loaded image.
    :param model: the YOLO trained model path.
    :return: bbox.
    sample:
    [{"xmin":228.25,"ymin":120.125,"xmax":357.75,"ymax":423.5,"confidence":0.9370117188,"class":14,"name":"person"},
    {"xmin":157.625,"ymin":17.375,"xmax":271.75,"ymax":311.0,"confidence":0.9233398438,"class":14,"name":"person"},
    {"xmin":504.25,"ymin":114.375,"xmax":619.5,"ymax":377.5,"confidence":0.9038085938,"class":14,"name":"person"},
    {"xmin":148.5,"ymin":0.28125,"xmax":358.75,"ymax":30.46875,"confidence":0.6474609375,"class":6,"name":"construction"},
    {"xmin":535.0,"ymin":210.0,"xmax":583.0,"ymax":249.75,"confidence":0.5380859375,"class":21,"name":"toy"}]
    """
    model = torch.hub.load('../tools', 'custom', path=model_path, source="local")
    result = model(image)  # tensor
    return result.pandas().xyxy[0].to_json(orient="records")


def detect_mmdet(image):
    """
    Call the MMDetection trained model for object detection.
    :param: one loaded image.
    :return: bbox.
    """
    det_configs = model_configs.detection()
    # Specify the path to model config and checkpoint file
    config_file = det_configs["config"]
    checkpoint_file = det_configs["checkpoint"]

    # 调用 MMDet 进行目标检测
    if os.path.exists(config_file):
        # build the model from a config file and a checkpoint file
        model = init_detector(config_file, checkpoint_file, device=get_device())
        # test a single image and show the results
        # img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
        result = inference_detector(model, image)
        # test a video and show the results
        # video = mmcv.VideoReader('video.mp4')
        # for frame in video:
        #     result = inference_detector(model, frame)
        #     model.show_result(frame, result, wait_time=1)
        return result

    return None


def detect_mmdet_show(model, image, result, save_path=None):
    """
    Call MMDet to show and/save the detection result.
    :param model:
    :param image:
    :param result:
    :param save_path:
    :return:
    """
    # visualize the results in a new window
    model.show_result(image, result)
    if os.path.exists(save_path):
        # or save the visualization results to image files
        model.show_result(image, result, out_file=save_path)


def segment_mmseg(image):
    """
    Call the MMSegmentation trained model for image segmentation.
    See https://github.com/open-mmlab/mmsegmentation/blob/master/demo/inference_demo.ipynb
    :param image:
    :return:
    """
    seg_configs = model_configs.segmentation()
    # Specify the path to model config and checkpoint file
    config_file = seg_configs["config"]
    checkpoint_file = seg_configs["checkpoint"]

    # 调用 MMSeg 进行目标检测
    if os.path.exists(config_file):
        model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
        result = inference_segmentor(model, image)
        return result

    return None


def segment_mmseg_show(model, image, result, save_path=None):
    """

    :param model:
    :param image:
    :param result:
    :param save_path:
    :return:
    """
    print("TODO")
    # show_result_pyplot(model, img, result, get_palette('cityscapes'))


def main():
    image = "../examples/test/test/test_2.jpg"
    model = "../tools/models/yolov5s.pt"
    res = detect_yolo(image, model)
    print(res)

if __name__ == '__main__':
    main()
