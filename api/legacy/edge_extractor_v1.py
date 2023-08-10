""""""

import os
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as patches

background_hard_soft = [0, 1, 2]

det_classes = [
    'potted plant', 'person', 'animal', 'bench', 'table', 'kite', 'vehicle', 'toy', 'frisbee', 'motorcycle',
    'bicycle', 'chair', 'bottle', 'fire hydrant', 'rock', 'construction', 'trashcan', 'plant', 'tire',
    'others', 'leaf debris', 'hedgehog', 'faeces', 'trashcan lid', 'sprinkler', 'branch'
]
seg_classes = [
    'lawn', 'road', 'terrain', 'sky', 'person', 'animal', 'toy', 'leaf_debris', 'plant', 'vehicle',
    'construction', 'fire_hydrant', 'sprinkler', 'bench', 'table', 'chair', 'pipe', 'faeces', 'rock',
    'bottle', 'trashcan_lid', 'others'
]
seg_soft_classes = ['lawn', 'road']
seg_class_ids = {
    'lawn': 1,          'road': 2,          'terrain': 3,       'sky': 4,       'person': 5,
    'animal': 6,        'toy': 7,           'leaf_debris': 8,   'plant': 9,     'vehicle': 10,
    'construction': 11, 'fire_hydrant': 12, 'sprinkler': 13,    'bench': 14,    'table': 15,
    'chair': 16,        'pipe': 17,         'faeces': 18,       'rock': 19,     'bottle': 20,
    'trashcan_lid': 21,
    'others': 255
}
seg_soft_class_ids = [1, 2]
seg_class_ids1 = {
    1: 'lawn', 2: 'road', 3: 'terrain', 4: 'sky', 5: 'person', 6: 'animal', 7: 'toy', 8: 'leaf_debris',
    9: 'plant', 10: 'vehicle', 11: 'construction', 12: 'fire_hydrant', 13: 'sprinkler', 14: 'bench',
    15: 'table', 16: 'chair', 17: 'pipe', 18: 'faeces', 19: 'rock', 20: 'bottle', 21: 'trashcan_lid',
    255: 'others'
}

OPT_TYPES = ["segmentation", "seg", "detection", "det", "both"]
ORIGINAL_PALETTE = [[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
                    [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
                    [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
                    [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
                    [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
                    [128, 128, 0]]
RELABEL_PALETTE = [[64, 96, 0], [0, 128, 160], [0, 32, 64]]


def read_det_results(filename):
    """Read detection results (bbox) to a numpy array."""
    text_file = open(filename, "r")
    lines = text_file.readlines()
    text_file.close()
    bbox_arr = []
    for line_str in lines: # one box in a line
        str_arr = line_str.split(" ")
        num_arr = list(map(float, str_arr))
        bbox_arr.append(num_arr)
    return np.array(bbox_arr)


def read_seg_results(filename):
    """Read segmentation results (mask) to a numpy array."""
    text_file = open(filename, "r")
    lines = text_file.readlines()
    text_file.close()
    seg_mask = []
    for line_str in lines: # one box in a line
        str_arr = line_str.split(",")
        num_arr = list(map(float, str_arr))
        seg_mask.append(num_arr)
    return np.array(seg_mask)


def create_label_colormap(relabel=False):
    """Create a label colormap for result visualization."""
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
        colormap[255] = ORIGINAL_PALETTE[20]
    else:
        colormap[0] = RELABEL_PALETTE[0]
        colormap[1] = RELABEL_PALETTE[1]
        colormap[2] = RELABEL_PALETTE[2]
    return colormap


def label_to_color_image(label, relabel_state):
    """Converts the output into color palettes."""
    colormap = create_label_colormap(relabel_state)
    label = label.astype(np.uint8)
    return colormap[label]


def save_image(np_image, image_file):
    Image.fromarray(np_image).save(image_file)


def draw_results(image, image_bhs, seg_mask, bbox_list, output_filename):
    """Draw the image, as well as the output BHS."""
    full_fig = plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 6])
    # Draw the original image
    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('Input Image')
    # Draw the BHS image
    plt.subplot(grid_spec[1])
    bhs_image = label_to_color_image(image_bhs, True).astype(np.uint8)
    plt.imshow(bhs_image)
    plt.axis('off')
    plt.title('BHS Edges')
    save_image(bhs_image, output_filename + "_bhs.png")
    # Draw the segmentation image (if any)
    sub_grid_id = 2
    if seg_mask is not None:
        plt.subplot(grid_spec[sub_grid_id])
        seg_image = label_to_color_image(seg_mask, False).astype(np.uint8)
        plt.imshow(seg_image)
        plt.axis('off')
        plt.title('Segmentation Mask')
        save_image(seg_image, output_filename + "_seg.png")
        sub_grid_id = 3  # use next grid
    # Draw the detection bbox image (if any)
    if bbox_list is not None:
        ax = plt.subplot(grid_spec[sub_grid_id])
        ax.set_title("Detection BBox")
        ax.imshow(image)
        w, h, c = image.shape
        for bbox in bbox_list:
            if len(bbox) == 5:
                bbox = bbox[1:]
            bbox_new = center_xy_to_left_xy([w, h], bbox)
            bb = patches.Rectangle(
                (bbox_new[0], bbox_new[1]), bbox_new[2], bbox_new[3],
                linewidth=2, edgecolor="blue", facecolor="none")
            ax.add_patch(bb)
        plt.axis('off')
        extent = ax.get_window_extent().transformed(
            full_fig.dpi_scale_trans.inverted())
        full_fig.savefig(output_filename + "_det.png", bbox_inches=extent)

    # Save the image
    output_image = output_filename + "_all.png"
    plt.savefig(output_image)
    # plt.show()
    return plt


def overlay_mask(img_bhs, seg_mask):
    # TODO assert sizes are equal
    # h, w = img_bhs.shape
    for i in range(len(img_bhs)):
        for j in range(len(img_bhs[i])):
            value = seg_mask[i][j]
            if value in seg_soft_class_ids:
                img_bhs[i][j] = background_hard_soft[2]
            elif value > 0:
                img_bhs[i][j] = background_hard_soft[1]
            else:
                img_bhs[i][j] = background_hard_soft[0]
    return img_bhs


def center_xy_to_left_xy(w_h, bbox):
    if len(bbox) == 5:
        bbox = bbox[1:]
    w, h = w_h  # real size
    left_x = int(bbox[0]*h)
    left_y = int(bbox[1]*w)
    box_w = int(bbox[2]*w)
    box_h = int(bbox[3]*h)
    left_x = int(left_x - box_w / 2)
    left_y = int(left_y - box_h / 2)
    return [left_x, left_y, box_w, box_h]


def overlay_box(img_bhs, bbox, bhs=1):
    h, w = img_bhs.shape
    box_x = int(bbox[0]*w)
    box_y = int(bbox[1]*h)
    box_w = int(bbox[2]*w)
    box_h = int(bbox[3]*h)
    # assign the specific elements to bhs
    x1 = int(box_x-box_w/2)
    x2 = int(box_x+box_w/2)
    y1 = int(box_y-box_h/2)
    y2 = int(box_y+box_h/2)
    img_bhs[y1:y2, x1:x2] = bhs
    # sub_img = img_bhs[y1:y2, x1:x2]
    # print(sub_img)
    return img_bhs


def parse(image, seg_mask=None, bbox_list=None, img_depth=None):
    """Extract the edge based on predefined background, hard and soft (BHS) classes."""
    if image is None:
        print("Input image is None!")
        return None
    if bbox_list is None and seg_mask is None:
        print("Both bbox_list and seg_mask are None!")
        return None

    h, w, c = image.shape
    img_bhs = np.zeros((h, w))
    # Edge extraction using segmentation results.
    if seg_mask is not None:
        # print()
        img_bhs = overlay_mask(img_bhs, seg_mask)
    # Edge extraction using detection results.
    if bbox_list is not None:
        # All detected objects are hard.
        bhs = background_hard_soft[1]
        for bbox in bbox_list:
            class_id = bbox[0]
            print("Get box of class_id=%d" % class_id)
            img_bhs = overlay_box(img_bhs, bbox[1:], bhs)

    return img_bhs


def save_results(image_bhs, output_path, filename="image_bhs"):
    if image_bhs is None:
        print("Image BHS is None.")
        return None
    if output_path is None:
        output_path = os.getcwd()
    else:  # Prepare the output dir
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    # Save the results (values)
    result_file = os.path.join(output_path, filename+"_bhs.txt")
    np.savetxt(result_file, image_bhs.astype(int), fmt='%i')


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', type=str, default="segmentation",
                        # choices=OPT_TYPES,  # ["segmentation", "seg", "detection", "det", "both"]
                        help="Base operation(s), segmentation, detection or both, for edge extraction.")
    parser.add_argument('image_path', type=str, help="Path of the target image.")
    parser.add_argument('--output_path', type=str, default='', help="Path to save the results.")
    parser.add_argument('--mask_path', type=str, help="Path of the segmentation mask.")
    parser.add_argument('--bbox_path', type=str, help="Path of the bounding boxes by detection.")
    parser.add_argument('--display', action='store_false', help="True or False option for display")
    args = parser.parse_args()
    return args


def main(args):
    operation_type = args.operation
    operation_type = str(operation_type).lower()
    if operation_type == OPT_TYPES[1]:  # "seg":
        operation_type = OPT_TYPES[0]  # "segmentation"
    if operation_type == OPT_TYPES[3]:  # "det":
        operation_type = OPT_TYPES[2]  # "detection"

    # Read data: image, seg_results, det_results and img_depth (TBD)
    image_path = args.image_path
    image_filename = os.path.basename(image_path)
    filename_tmp = image_filename.rsplit(".")
    if len(filename_tmp) > 0:
        image_filename = filename_tmp[0]
    print("Load an image from " + image_path)
    target_image = Image.open(image_path)
    target_image = np.asarray(target_image)

    seg_mask = None
    bbox_list = None
    # Extract the background, hard and soft (BHS) edges
    if operation_type == OPT_TYPES[0] \
            or operation_type == OPT_TYPES[4]:  # "segmentation"
        segmentation_path = args.mask_path
        seg_mask = read_seg_results(segmentation_path)
    if operation_type == OPT_TYPES[2] \
            or operation_type == OPT_TYPES[4]:  # "detection"
        detection_path = args.bbox_path
        bbox_list = read_det_results(detection_path)
    # TODO: img_depth
    image_bhs = parse(target_image, seg_mask, bbox_list)

    # Output the results
    output_path = args.output_path
    save_results(image_bhs, output_path, image_filename)

    # Save the results (image)
    result_filename = os.path.join(output_path, image_filename)
    plt_res = draw_results(target_image, image_bhs, seg_mask, bbox_list,
                           result_filename)

    # Display and/or save the images
    if args.display:
        plt_res.show()


if __name__ == '__main__':
    args = args_parser()
    main(args)



