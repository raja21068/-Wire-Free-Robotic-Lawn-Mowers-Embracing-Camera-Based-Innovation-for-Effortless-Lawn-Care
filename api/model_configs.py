"""
Configurations of the trained models.
"""


def detection():
    return {
        "config": "configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py",
        "checkpoint": "checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
    }


def segmentation():
    return {
        "config": "configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py",
        "checkpoint": "checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth"
    }


def main():
    print("")


if __name__ == '__main__':
    main()
