import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
import os
from pybboxes import BoundingBox
import matplotlib.patches as patches

DET_CLASSES = [
    'potted plant', 'person', 'animal', 'bench', 'table', 'kite', 'vehicle', 'toy', 'frisbee', 'motorcycle',
    'bicycle', 'chair', 'bottle', 'fire hydrant', 'rock', 'construction', 'trashcan', 'plant', 'tire',
    'others', 'leaf debris', 'hedgehog', 'faeces', 'trashcan lid', 'sprinkler', 'branch'
]
SEG_CLASSES = [
    'lawn', 'road', 'terrain', 'sky', 'person', 'animal', 'toy', 'leaf_debris', 'plant', 'vehicle',
    'construction', 'fire_hydrant', 'sprinkler', 'bench', 'table', 'chair', 'pipe', 'faeces', 'rock',
    'bottle', 'trashcan_lid', 'trashcan', 'others'
]
SEG_MASK = {
    1: 'lawn', 2: 'road', 3: 'terrain', 4: 'sky', 5: 'person', 6: 'animal', 7: 'toy', 8: 'leaf_debris',
    9: 'plant', 10: 'vehicle', 11: 'construction', 12: 'fire_hydrant', 13: 'sprinkler', 14: 'bench',
    15: 'table', 16: 'chair', 17: 'pipe', 18: 'faeces', 19: 'rock', 20: 'bottle', 21: 'trashcan_lid',
    22: 'trashcan',
    255: 'others'
}

RELABEL = {0: [4, 0], 1: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 255], 2: [1, 2, 3]}
DET_LABEL = {0: [], 1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], 2: []}

ORIGINAL_PALETTE = [[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
                    [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
                    [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
                    [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
                    [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
                    [128, 128, 0], [128, 192, 0]]

RELABEL_PALETTE = [[64, 96, 0], [0, 128, 160], [0, 32, 64]]


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', type=str, default="segmentation", choices=["segmentation", "detection", "both"],
                        help="Type of operation to perform")
    parser.add_argument('image_path', type=str, help="Image path")
    parser.add_argument('--output_path', type=str, default='', help="Path to save results")
    parser.add_argument('--mask_path', type=str, help="Infered mask text mask path")
    parser.add_argument('--bbox_path', type=str, help="A text file path with bounding boxes")
    parser.add_argument('--display', action='store_false', help="True or False option for display")
    args = parser.parse_args()

    return args


def save_output(output_path, results, file_name):
    """Converts yolo bounding box format to coco format with absolute values.
               Parameters
               ----------
               output_path: str, directory path to save output
               results: Numpy Array, results of relabeling detection and segmentation operation
               file_name: str, Filename to save results
               Returns
               -------
               tuple with absolute coco bounding box
    """
    if output_path:
        np.savetxt(os.path.join(output_path, file_name), results.astype(int), fmt='%i')
    else:
        np.savetxt(file_name, results.astype(int), fmt='%i')


def get_abosolute_position(shape_img, bbox):
    """Converts yolo bounding box format to coco format with absolute values.
                        Parameters
                        ----------
                        shape_img: Tuple, the image input path
                        bbox: List, the bounding  boxes
                        Returns
                        -------
                        tuple with absolute coco bounding box
    """
    cc = BoundingBox.from_yolo(bbox[0], bbox[1], bbox[2], bbox[3], image_size=(shape_img[0], shape_img[1]))
    abs_x, abs_y, abs_w, abs_h = cc.to_coco().values
    return abs_x, abs_y, abs_w, abs_h


def draw_bounding(image, boxes, detection_mask):
    """Display the bounding boxes and the detection mask.
                     Parameters
                     ----------
                     image: PIL Image Object, the image input path
                     boxes: List, the bounding  boxes
                     detection_mask: Numpy Array, the new detection relabeled output
                     Returns
                     -------
                     None
    """

    fig, ax = plt.subplots(2)
    for box in boxes:
        bb = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2, edgecolor="blue", facecolor="none")
        ax[0].add_patch(bb)

    ax[0].set_title("Detection")
    ax[1].set_title("Detection mask")
    ax[0].imshow(image)
    ax[1].imshow(detection_mask)
    ax[0].axis('off')
    ax[1].axis('off')
    plt.savefig("relabel_detection.png")
    plt.show()



def detection_extraction(image_path, bboxes, output_path,display):
    """relabels the detection results labels into background, soft and hard.
                 Parameters
                 ----------
                 image_path: str, the image input path
                 bboxes: str, the bounding  box  text file path
                 output_path: str, the directory path to save output
                 display: bool, whether to display detection results
                 Returns
                 -------
                 the new detection label results
    """
    image = Image.open(image_path)
    img_shape = image.size
    detection_boxes = np.loadtxt(bboxes)
    new_image = np.zeros((img_shape[1], img_shape[0]))
    classes = detection_boxes[:, 0].astype(int)
    relabeled_bounding = []
    for i, det_class in enumerate(classes):
        for reID, trID in DET_LABEL.items():
            if int(det_class) in trID:
                newID = reID
                x, y, w, h = get_abosolute_position(img_shape, detection_boxes[i, 1:])
                new_image[y:y+h, x:x+w] = newID
                relabeled_bounding.append([x, y, w, h])
                break

    if display:
       draw_bounding(image, relabeled_bounding, new_image)
    save_output(output_path, new_image, "detection_relabeled.txt")

    return new_image

def segment_extraction(image_path, segmentation_path, output_path, display=True):
    """relabels the segmentation mask labels into background, soft and hard.
              Parameters
              ----------
              image_path: str, the image input path
              segmentation_path: str, the segmentation text file path
              output_path: str, the directory path to save output
              display: bool, whether to display segmentation results
              Returns
              -------
              the new segmentation mask label
    """
    image = Image.open(image_path)
    segmentation = np.loadtxt(segmentation_path, delimiter=',', dtype=int)

    result_mask = segmentation
    get_unique_class = np.unique(segmentation)
    copy_mask = result_mask.copy()
    for class_id in get_unique_class:
        class_id = int(class_id)
        for reID, trID in RELABEL.items():
            if int(class_id) in trID:
                newID = reID
                copy_mask[result_mask == class_id] = newID
                break

    if display:
        plt.figure(figsize=(15, 5))
        grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

        plt.subplot(grid_spec[0])
        plt.imshow(image)
        plt.axis('off')
        plt.title('input image')

        plt.subplot(grid_spec[1])
        seg_image = label_to_color_image(segmentation, False).astype(np.uint8)
        plt.imshow(seg_image)
        plt.axis('off')
        plt.title('segmentation mask')

        plt.subplot(grid_spec[2])
        seg_image_re = label_to_color_image(copy_mask, True).astype(np.uint8)
        plt.imshow(seg_image_re)
        plt.axis('off')
        plt.title('segmentation relabeled')
        plt.savefig("seg_relabeled.png")
        plt.show()

    save_output(output_path, copy_mask, "segment_relabeled.txt")

    return copy_mask


def create_label_colormap(relabel=False):
    """Creates a label colormap for  segmentation.
    Returns:
    A colormap for visualizing segmentation results.
   """
    colormap = np.zeros((256, 3), dtype=np.uint8)

    if not relabel:
        colormap[0] = ORIGINAL_PALETTE[0]
        colormap[1] = ORIGINAL_PALETTE[1]
        colormap[2] = ORIGINAL_PALETTE[2]
        colormap[3] = ORIGINAL_PALETTE[3]
        colormap[4] = ORIGINAL_PALETTE[4]
        colormap[5] = ORIGINAL_PALETTE[5]
        colormap[6] = ORIGINAL_PALETTE[6]
        colormap[7] = ORIGINAL_PALETTE[7]
        colormap[8] = ORIGINAL_PALETTE[8]
        colormap[9] = ORIGINAL_PALETTE[9]
        colormap[10] = ORIGINAL_PALETTE[10]
        colormap[11] = ORIGINAL_PALETTE[11]
        colormap[12] = ORIGINAL_PALETTE[12]
        colormap[13] = ORIGINAL_PALETTE[13]
        colormap[14] = ORIGINAL_PALETTE[14]
        colormap[15] = ORIGINAL_PALETTE[15]
        colormap[16] = ORIGINAL_PALETTE[16]
        colormap[17] = ORIGINAL_PALETTE[17]
        colormap[18] = ORIGINAL_PALETTE[18]
        colormap[19] = ORIGINAL_PALETTE[19]
        colormap[20] = ORIGINAL_PALETTE[20]
        colormap[21] = ORIGINAL_PALETTE[21]
        colormap[255] = ORIGINAL_PALETTE[20]
    else:
        colormap[0] = RELABEL_PALETTE[0]
        colormap[1] = RELABEL_PALETTE[1]
        colormap[2] = RELABEL_PALETTE[2]

    return colormap


def label_to_color_image(label, relabel_state):
    """Converts segementation output into color palettes.
                   Parameters
                   ----------
                   label: int, segmentation pixel label
                   relabel_state: bool, checks whether to display original segmentation or transformed segmentation labels
                   Returns
                   -------
                   Numpy Array, new array transformed into color palettes
    """
    colormap = create_label_colormap(relabel_state)
    return colormap[label]


if __name__ == "__main__":
    args = args_parser()
    operation_type = args.operation
    operation_type = str(operation_type).lower()
    image_path = args.image_path
    display = args.display
    output_path = args.output_path

    if operation_type == "segmentation":
        segmentation_path = args.mask_path
        relabeled_maks = segment_extraction(image_path, segmentation_path, output_path, display)
    elif operation_type == "detection":
        bboxes_path = args.bbox_path
        relabeled_detect = detection_extraction(image_path, bboxes_path, output_path, display)
    elif operation_type == "both":
        segmentation_path = args.mask_path
        bboxes_path = args.bbox_path
        relabeled_maks = segment_extraction(image_path, segmentation_path, output_path, display)
        relabeled_detect = detection_extraction(image_path, bboxes_path, output_path, display)

