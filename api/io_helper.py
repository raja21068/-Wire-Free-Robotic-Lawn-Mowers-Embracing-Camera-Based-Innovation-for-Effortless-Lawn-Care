"""
输入输出方法，支持视频和图像。
"""
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

LABEL_PALETTE = [[32, 223, 128], [0, 0, 255], [0, 255, 0], [1, 190, 200], [64, 96, 0], [0, 128, 160], [0, 32, 64]]


def read_image(image_path):
    """
    读取图片（OpenCV）。
    :param image_path:
    :return:
    """
    return cv2.imread(image_path)


def read_video(video_path, interval, index=0):
    """
    读取视频输入，支持抽帧。
    :param video_path: 视频路径。
    :param interval: 抽帧间隔。
    :param index: 指定帧数，支持外部控制。
    :return: 抽取的一个或多个图像帧。
    """
    assert os.path.exists(video_path)
    # 读取视频
    video_capture = cv2.VideoCapture(video_path)
    # 视频抽帧
    num_sec = 0
    num_frame = 0
    frames = []
    while True:
        success, frame = video_capture.read()
        num_sec += 1  # timing
        if success:
            if num_sec % interval == 0:
                num_frame += 1  # 取到
                if index > 0 and index == num_frame:
                    return frame  # 只返回指定的 1 帧图像
                frames.append(frame)
        else:  # 视频结束
            print("Video ends.")
            break
    # 返回所有抽帧结果
    return frames


def read_det_bbox(image_path, det_suffix="_det.txt"):
    """
    读取图像的目标检测结果（用于演示）。
    :param image_path:
    :param det_suffix:
    :return:
    """
    assert os.path.exists(image_path)
    det_file_path = filename_only(image_path) + det_suffix
    print("Read det results from " + det_file_path)
    bbox_list = np.loadtxt(det_file_path)
    return bbox_list


def read_seg_mask(image_path, seg_suffix="_seg.txt"):
    """
    读取图像分割结果（用于演示）。
    :param image_path:
    :param seg_suffix:
    :return:
    """
    seg_file_path = filename_only(image_path) + seg_suffix
    seg_mask = np.loadtxt(seg_file_path, delimiter=',', dtype=int)
    return seg_mask


def read_boundary_xml(xml_path):
    """
    读取标注的软硬边界，XML格式。
    :param xml_path:
    :return:
    """
    # create element tree object
    tree = ET.parse(xml_path)
    root = tree.getroot()
    image_items = []
    # iterate image items
    for item in root.findall('./image'):
        image = {}
        polylines = []
        image['name'] = item.attrib['name']
        # iterate child elements of item
        for child in item:
            polyline = {}
            polyline['label'] = child.attrib['label']
            # 1266.10,1079.30;1265.40,1070.60
            points = []
            points_txt = child.attrib['points']
            xny_txt_arr = points_txt.split(";")
            for i in range(len(xny_txt_arr)):
                xy_txt = xny_txt_arr[i]
                xy = xy_txt.split(",")
                points.append(np.array(xy, np.float))
            polyline['points'] = points
            polylines.append(polyline)
        image['polylines'] = polylines
        image_items.append(image)
    return image_items


def get_boundary_labels(all_labels, image_path):
    """
    查找图像的边界标注信息。
    :param all_labels:
    :param image_path:
    :return:
    """
    filename = os.path.basename(image_path)
    for label in all_labels:
        if filename == label['name']:
            return label


def filename_only(file_path):
    """
    取文件名，不含扩展名。
    :param file_path:
    :return:
    """
    dir_name = os.path.dirname(file_path)
    filename0 = os.path.basename(file_path)
    filename_ext = filename0.split(".")
    filename1 = filename_ext[0] if len(filename_ext) > 0 else filename0
    # filename_ext = filename_ext[-1] if len(filename_ext) > 0 else None
    # filename1 = filename1 + new_ext
    return os.path.join(dir_name, filename1)


def write_image(image_path, image):
    """
    保存图像（视频帧）。
    :param image_path: 保存路径。
    :param image: 图像数据。
    :return:
    """
    cv2.imwrite(image_path, image)


def draw_edges(image, contours, thickness=5, edge_type=0):
    """
    在图片上画轮廓线。
    :param image:
    :param contours:
    :param thickness:
    :param edge_type: see edge_definitions.edge_pixels
    :return:
    """
    # image = cv2.polylines(image, [contour], isClosed, (0,255,0), 3)
    line_color = LABEL_PALETTE[edge_type]
    return cv2.drawContours(image, contours, -1, line_color, thickness=thickness)


def draw_linestring(image, linestrings, color, point_step=10, radius=2, thickness=1):
    """
    画Shapely.Linestring
    :param image:
    :param linestrings:
    :param color:
    :param point_step: 画点间隔
    :param radius:
    :param thickness:
    :return:
    """
    assert image is not None
    assert linestrings is not None
    for line in linestrings:
        # points = np.array(line.coords).flatten()  # 格式不限
        points = np.array(line).flatten()  # 格式不限
        for i in range(len(points) - 4):
            if i % 2 == 0:
                p1 = (int(points[i + 0]), int(points[i + 1]))
                p2 = (int(points[i + 2]), int(points[i + 3]))
                cv2.line(image, p1, p2, color=color, thickness=thickness)
                if radius > 0 and i % point_step == 0:  # 是否画点（只画少数点）
                    cv2.circle(image, p1, color=color, radius=radius, thickness=thickness)
                    # cv2.circle(image, p2, color=color, radius=radius, thickness=thickness)
    return image


def draw_lines(image, lines, color, radius=2, thickness=1):
    # colors = [(0, 255, 0), (200, 200, 200), (0, 0, 255), (255, 0, 0)]  # Red, Green, Blue
    # colors = [(55, 176, 55), (84, 225, 227)]
    # line_color = colors[line_type]
    assert image is not None
    assert lines is not None
    for line in lines:
        points = np.array(line).flatten()  # 格式不限
        for i in range(len(points) - 2):
            if i % 2 == 0:
                p1 = (int(points[i + 0]), int(points[i + 1]))
                p2 = (int(points[i + 2]), int(points[i + 3]))
                cv2.line(image, p1, p2, color=color, thickness=thickness)
                if radius > 0:  # 是否画点
                    cv2.circle(image, p1, color=color, radius=radius, thickness=thickness)
                    cv2.circle(image, p2, color=color, radius=radius, thickness=thickness)
    return image


def draw_text(image, text, thickness=3):
    [h, w, _] = image.shape
    org = (h - 50, w - 200)
    org = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 255)
    image = cv2.putText(image, text, org, font,
                        font_scale, color, thickness, cv2.LINE_AA)
    return image


def main():
    print("")


if __name__ == '__main__':
    main()
