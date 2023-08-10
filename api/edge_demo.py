"""
Soft/Hard edges demos.
"""
import os
import time

import io_helper
import model_helper
import edge_extractor
import edge_definitions
import edge_metric


def image_dir(dir_path):
    """
    演示处理文件夹内的测试样本。
    :param dir_path:
    :return:
    """
    assert os.path.exists(dir_path)
    positives = 0
    total_samples = 0
    total_time = 0
    for f in os.listdir(dir_path):
        if f.endswith(".jpg"):
            results = image(os.path.join(dir_path, f))
            total_samples += 1
            positives += 1 if results[-1] < 0.1 else 0
            total_time += results[2]

    average_time = total_time/total_samples
    print("Average processing time: %03f " % average_time)
    accuracy = positives/total_samples
    print("Accuracy: %02f" % accuracy)


def image(image_path):
    """
    Image input demo.
    :param image_path:
    :return:
    """
    # 1. 读取图片
    print("1. Read image from " + image_path)
    in_image = io_helper.read_image(image_path)
    assert in_image is not None
    # 2. 目标检测
    print("2. Perform object detection ... ")
    bbox_list = None  # model_helper.detect_mmdet(image)
    # if bbox_list is None:  # 测试环境无效，直接读取文件结果
    #     print("WARNING: Read det results from txt file.")
    #     bbox_list = io_helper.read_det_bbox(image_path)
    #     assert bbox_list is not None  # 仅用于演示
    # 3. 图像分割
    print("3. Perform image segmentation ...")
    seg_mask = None  # model_helper.segment_mmseg(image)
    if seg_mask is None:  # 测试环境无效，直接读取文件结果
        print("WARNING: Read seg results from txt file.")
        seg_mask = io_helper.read_seg_mask(image_path)
        assert seg_mask is not None  # 仅用于演示
    # 4. 边界提取
    print("4. Extract soft/hard edges ...")
    start_time = time.time()
    soft_hard_edges = edge_extractor.analyze_edges(seg_mask, bbox_list, None)
    end_time = time.time()  # 计时
    time_edges = end_time - start_time
    print("\t done in %03f seconds." % time_edges)
    assert soft_hard_edges is not None
    # 5. 计算精度/误差
    print("5. Calculate errors ...")
    filename = io_helper.filename_only(image_path)
    hard_pixel = edge_definitions.edge_pixels['hard_edge']
    soft_pixel = edge_definitions.edge_pixels['soft_edge']
    soft_edges = soft_hard_edges[soft_pixel]
    hard_edges = soft_hard_edges[hard_pixel]
    labels = io_helper.read_boundary_xml(filename + ".xml")
    abs_rel = edge_metric.abs_rel(labels, hard_edges, soft_edges)
    print("\t AbsRel = %02f." % abs_rel)
    # 6. 保存结果
    file_path = filename + "_soft_hard_edges.png"
    for s_cnts in soft_edges:  # 先画软边界
        in_image = io_helper.draw_edges(in_image, s_cnts, 3, soft_pixel)
    for h_cnts in hard_edges:  # 再画硬边界
        in_image = io_helper.draw_edges(in_image, h_cnts, 3, hard_pixel)
    io_helper.write_image(file_path, in_image)  # 保存图片

    # 返回结果
    return [0, 0, time_edges, abs_rel]


def video(video_path):
    """
    Video input demo.
    :param video_path:
    :return:
    """
    print("")


def main():
    # test_image_1 = os.path.join(os.getcwd(), '../examples/images_4/2022_08_25_17_00_33.jpg')
    test_image_1 = os.path.join(os.getcwd(), '../examples/images_4/2022_08_25_16_43_14.jpg')
    image(test_image_1)
    # test_image_dir = os.path.join(os.getcwd(), '../examples/images_4/')
    # image_dir(test_image_dir)


if __name__ == '__main__':
    main()
