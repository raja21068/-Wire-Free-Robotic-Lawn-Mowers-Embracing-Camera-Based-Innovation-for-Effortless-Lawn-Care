import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from pybboxes import BoundingBox
import argparse
import os

RELABEL = {2: [], 1: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 255], 0: [1, 2, 3, 0]}
TEST_LABEL = {0:[1],1:[2]}
ALLOW_GAP = 5
EDGE_PIXEL = {"soft_edge":0,"border_edge":1,"within_edge":2, "unknown": 3}
EDGES_PALETTE = [[0, 192, 96], [0, 128, 160], [128, 0, 96], [0, 32, 192]]

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help="Path of the target image.")
    parser.add_argument('--output_path', type=str, default='', help="Path to save the results.")
    parser.add_argument('--mask_path', type=str, help="Path of the segmentation mask.")
    parser.add_argument('--bbox_path', type=str, help="Path of the bounding boxes by detection.")
    parser.add_argument('--display', action='store_false', help="True or False option for display")
    args = parser.parse_args()
    return args


#Check whether object class is hard or soft
def check_label_type(class_id):
    edge_type = 1
    if class_id<4:
        edge_type =0
    return  edge_type


def check_unknown_intersection(edge_cordinates, detection_cordinates):
  intersect =False
  list_inter = []
  for dtc in detection_cordinates:
      if dtc in edge_cordinates:
          return True, dtc

  return  intersect, list_inter

def assign_unknown_color(start_c, end_cordinate, color, edge_pix,  image_display, image_results):
    image_display_c = image_display
    image_results_c = image_results
    for img_c in range(start_c, end_cordinate[1]+1):
          image_display_c[img_c ,end_cordinate[0],:]= color
          image_results_c[img_c ,end_cordinate[0]] = edge_pix
    return image_display_c, image_results_c

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


def save_results(image_bhs, results_img, output_path, filename="image_edges"):

    if output_path=='':
        output_path = os.getcwd()
    else:  # Prepare the output dir
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    # Save the results (values)

    cv2.imwrite("display_results.jpg", image_bhs)

    result_file = os.path.join(output_path, filename+"_bhs.txt")
    np.savetxt(result_file, results_img.astype(int), fmt='%i')


def discard_edges(hard_edges, main_image , countours, saved_cordinates, relabed_image):

    cnts= countours
    nump_img= main_image
    relabel_edge = relabed_image
    keep_hard_edges =hard_edges
    history_cord = saved_cordinates

    if len(keep_hard_edges)>1:
        keep_hard_edges.sort(key=lambda x: x[1])
        stepest_hard_edge = keep_hard_edges[-1]
        stepest_hard_key = stepest_hard_edge[2]
        stepest_hard_cor = [stepest_hard_edge[1],stepest_hard_edge[0]]
        stepest_hard_color =  stepest_hard_edge[3]
        cv2.drawContours(nump_img, cnts, int(stepest_hard_key.split("_")[1]), stepest_hard_color, thickness=2)
        for ori_key, ori_values in history_cord.items():

            if ori_key == stepest_hard_key:
                continue

            ori_lowest = ori_values['lowest']
            ori_hard_state = ori_values['hard_state']
            ori_originals =   ori_values['original']
            ct_con = ori_values['ct']
            color_edge = ori_values['color']
            x_cord = ori_lowest[1]

            if x_cord < stepest_hard_cor[0]:
                for indx, cp_ct in enumerate(ct_con):
                    cs = cp_ct[0]
                    nump_img[cs[1], cs[0], :] = list(ori_originals[indx])
                    relabel_edge[cs[1], cs[0], :] = list(ori_originals[indx])
            else:
                cv2.drawContours(nump_img, cnts, int(ori_key.split("_")[1]), color_edge, thickness=2)

    else:

        for ori_key, ori_values in history_cord.items():
            color_edge = ori_values['color']
            cv2.drawContours(nump_img, cnts, int(ori_key.split("_")[1]), color_edge, thickness=2)

    return nump_img, relabel_edge

def depth(image_path, bboxes_path, segmentation_path, depth_path, output_path, display):
    image = cv2.imread(image_path)
    segmentation = np.loadtxt(segmentation_path, delimiter=',', dtype=int)
    depth = np.loadtxt(depth_path, delimiter=',', dtype=int)
    detection_boxes = np.loadtxt(bboxes_path)
    d_ndim = detection_boxes.ndim
    if d_ndim == 1:
        detection_classes = [int(detection_boxes[0])]
    else:
        detection_classes = detection_boxes[:, 0].astype(int)



def main(image_path, bboxes_path, segmentation_path, output_path, display):

    image = cv2.imread(image_path)
    segmentation = np.loadtxt(segmentation_path, delimiter=',', dtype=int)
    detection_boxes = np.loadtxt(bboxes_path)
    d_ndim = detection_boxes.ndim

    if d_ndim == 1:
        detection_classes = [int(detection_boxes[0])]
    else:
        detection_classes = detection_boxes[:, 0].astype(int)

    result_mask = segmentation
    plt.imshow(result_mask)
    plt.show()

    copy_mask = result_mask.copy()
    mask = copy_mask.astype(np.uint8)
    history_cord = {}

    # perform edge extraction with canny and find contours
    mask = cv2.Canny(mask, 0, -1)
    cnts, heirachy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # Find  all possibile Edges in mask

    #anchor_image = np.asarray(copy.deepcopy(image))
    relabel_edge = np.asarray(copy.deepcopy(image))
    nump_img = np.asarray(image)

    width, height = nump_img.shape[1], nump_img.shape[0]
    img_shape = [width, height]

    keep_hard_edges = []

    for index, cp in enumerate(cnts):
        if len(cp) > 100:  # consider only edges with more than 100 cordinates, remove small edges

            unknown_edge_status = False  # considers the state of an unknown object
            hard_edge_state = False

            # sort countours cordinates to get steepest x cordinate
            exame = cp.copy().squeeze().tolist()
            exame.sort(key=lambda x: x[1])
            select_index = len(exame) - 1
            selected_cordinate = exame[select_index]
            current_class_anchor = result_mask[selected_cordinate[1], selected_cordinate[0]]

            # get coordinates of lower and above of the  edges
            above_point = [selected_cordinate[1] - 8, selected_cordinate[0]]
            below_point = [selected_cordinate[1] + 8, selected_cordinate[0]]
            check_hard_above_edge = result_mask[1:above_point[0], 1: width - 1]

            # get possible class of above surrounding regions of current edge
            above_class = -1
            above_class_1 = result_mask[above_point[0] - 3:above_point[0], above_point[1]:above_point[1] + 3]
            above_class_2 = result_mask[above_point[0] - 3:above_point[0], above_point[1] - 3:above_point[1]]
            if len(above_class_1) != 0 or len(above_class_2) != 0:
                above_class = np.bincount(np.concatenate((above_class_1, above_class_2), axis=0).flatten()).argmax()

            # get possible class of below surrounding regions of current edge
            below_class = -1
            below_class_1 = result_mask[below_point[0] - 3:below_point[0], below_point[1]:below_point[1] + 3]
            below_class_2 = result_mask[below_point[0] - 3:below_point[0], below_point[1]:below_point[1] + 3]
            if len(below_class_1) != 0 or len(below_class_2) != 0:
                below_class = np.bincount(np.concatenate((below_class_1, below_class_2), axis=0).flatten()).argmax()

            # get possible class of current edge using suround regions
            current_class = -1
            current_class_t1 = result_mask[selected_cordinate[1] - 3:selected_cordinate[1],
                               selected_cordinate[0]:selected_cordinate[0] + 3]
            current_class_t2 = result_mask[selected_cordinate[1] - 3:selected_cordinate[1],
                               selected_cordinate[0] - 3:selected_cordinate[0]]
            concate_nump = np.concatenate((current_class_t1, current_class_t2), axis=0).flatten()

            if len(current_class_t1) != 0 or len(current_class_t2) != 0:
                max_class = np.bincount(concate_nump).argmax()
                current_class = max_class

            # get whether current object class is hard or soft
            above_type = check_label_type(above_class)
            below_type = check_label_type(below_class)
            current_class_type = check_label_type(current_class)

            # Check whether we have a bigger class region
            allow_state = False
            allowable_surrounding = result_mask[selected_cordinate[1] - ALLOW_GAP:selected_cordinate[1] + ALLOW_GAP,
                                    selected_cordinate[0] - ALLOW_GAP:selected_cordinate[0] + ALLOW_GAP]
            if allowable_surrounding.shape[1] != 0:
                pix_surround = allowable_surrounding.flatten()
                count_surround = np.bincount(pix_surround)
                max_surround = count_surround.argmax()
                count_max = count_surround[max_surround]
                reg_conf = count_max / len(pix_surround)
                if reg_conf > 0.50:
                    allow_state = True

            # check all conditions for soft and hard edges
            
            if selected_cordinate[1]/height<0.2 :
             continue
            if current_class == 5 and above_type == 1 and below_type == 0:
                color_choice = EDGES_PALETTE[2]
                edge_pixel = EDGE_PIXEL['within_edge']
            elif above_class == 4 or current_class_anchor == 4:
                continue
            elif current_class_type == 1 and below_type == 0 and current_class == 5:
                color_choice = EDGES_PALETTE[2]
                edge_pixel = EDGE_PIXEL['within_edge']
            elif current_class == 9 and (below_class == 1 or below_class == 2) and allow_state and len(cp) > (
                    0.5 * width) and (19 in check_hard_above_edge) == False:
                color_choice = EDGES_PALETTE[1]
                edge_pixel = EDGE_PIXEL['border_edge']
                unknown_edge_status = True
                hard_edge_state = True
            elif current_class == 9 and (below_class == 1 or below_class == 2) and allow_state and len(cp) < (
                    0.5 * width):
                color_choice = EDGES_PALETTE[2]
                edge_pixel = EDGE_PIXEL['within_edge']
            elif (above_type == 0 and below_type == 0):
                color_choice = EDGES_PALETTE[0]
                edge_pixel = EDGE_PIXEL['soft_edge']

            elif current_class == 7 and allow_state and below_class == 2:
                color_choice = EDGES_PALETTE[2]
                edge_pixel = EDGE_PIXEL['within_edge']
            elif current_class == 19 and below_type == 0 and allow_state:
                color_choice = EDGES_PALETTE[2]
                edge_pixel = EDGE_PIXEL['within_edge']
            elif current_class == 11 and below_type == 0 and  len(cp) < (0.5 * width) :
                color_choice = EDGES_PALETTE[2]
                edge_pixel = EDGE_PIXEL['within_edge']
            elif above_class == 19 and allow_state:
                color_choice = EDGES_PALETTE[2]
                edge_pixel = EDGE_PIXEL['within_edge']
            else:
                color_choice = EDGES_PALETTE[1]
                hard_edge_state =True
                edge_pixel = EDGE_PIXEL['border_edge']

                if (19 in check_hard_above_edge) == False:
                    unknown_edge_status = True

            # if hard_edge_state ==True:
            #      keep_hard_edges.append(selected_cordinate+["state_"+str(index),color_choice])

            original_pixels= []
            # loop through edges and assign new edge values
            for ct in cp:
                cs = ct[0]
                #original_pixels.append(anchor_image[cs[1], cs[0]])
                nump_img[cs[1], cs[0], :] = color_choice

                relabel_edge[cs[1], cs[0]] = edge_pixel

            #comment this statement if discard_edges function is enabled
            cv2.drawContours(nump_img, cnts, index, tuple(color_choice), thickness=2)

            #history_cord["state_"+str(index)]= {"ct":cp,"hard_state":hard_edge_state,"lowest":selected_cordinate,"original":original_pixels,"color":color_choice}
            # condition to specifically draw unknown boundaries
            if unknown_edge_status == True:
                for i, det_class in enumerate(detection_classes):
                    if d_ndim ==1:
                        x, y, w, h = get_abosolute_position(img_shape, detection_boxes[1:])
                    else:
                        x, y, w, h = get_abosolute_position(img_shape, detection_boxes[i, 1:])

                    #generate corner points of  detection mask (might be needed)
                    top_left = (x, y)
                    top_right = (x + w, y)
                    bottom_left = (x, y + h)
                    bottom_right = (x + w, y + h)

                    nump_img[y, x:x + w, :] = EDGES_PALETTE[3]  # top line

                    # get coordinates of left and right side lines of detection mask
                    left_side_coords = [[x, c] for c in range(y, y + h + 1)]
                    right_side_coords = [[x + w - 1, c] for c in range(y, y + h + 1)]
                    left_state, left_cordinate = check_unknown_intersection(exame, left_side_coords)
                    right_state, right_cordinate = check_unknown_intersection(exame, right_side_coords)

                    if left_cordinate:
                        nump_img, relabel_edge = assign_unknown_color(y, left_cordinate, EDGES_PALETTE[3],
                                                                      EDGE_PIXEL['unknown'], copy.deepcopy(nump_img),
                                                                      copy.deepcopy(relabel_edge))

                    if right_state:
                        nump_img, relabel_edge = assign_unknown_color(y, right_cordinate, EDGES_PALETTE[3],
                                                                      EDGE_PIXEL['unknown'], copy.deepcopy(nump_img),
                                                                      copy.deepcopy(relabel_edge))


    #uncomment this code if egdes above hard egdes needs to be discarded
    #nump_img, relabel_edge= discard_edges(keep_hard_edges, copy.deepcopy(nump_img) , cnts, history_cord, copy.deepcopy(relabel_edge))

    if display:
       cv2.imshow('mask', nump_img)
       cv2.waitKey()

    save_results(nump_img, relabel_edge[:, :, 0].astype(int), output_path, filename="image_edges")




if __name__ == "__main__":

    args = args_parser()
    image_path = args.image_path
    output_path = args.output_path
    segmentation_path = args.mask_path
    bboxes_path = args.bbox_path
    display = args.display

    main(image_path, bboxes_path, segmentation_path, output_path ,display)



