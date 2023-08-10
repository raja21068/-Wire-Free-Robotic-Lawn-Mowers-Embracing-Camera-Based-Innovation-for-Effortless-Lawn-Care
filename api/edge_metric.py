"""
Calculate the metrics of Soft/Hard edge extraction.
"""
import os
import math

import numpy as np


def acc(labels, results):
    """
    Accuracy.
    :param labels:
    :param results:
    :return:
    """
    print("")


def abs_rel(image, labels, hard_edges, soft_edges):
    """
    针对每一张测试图像的误差计算：AbsRel = .
    :param image:
    :param labels:
    :param hard_edges:
    :param soft_edges:
    :return:
    """
    [h, _, _] = image.shape
    hard_ar = 0
    soft_ar = 0
    hard_lines = []
    soft_lines = []
    hard_refs = []  # referenced points in hard edges
    soft_refs = []  # referenced points in soft edges
    image_name = labels['name']
    polylines = labels['polylines']
    for polyline in polylines:
        line_label = polyline['label']
        points = polyline['points']
        if line_label in ['hard_boundary', 'unknown']:
            hard_lines.append(points)
        else:
            soft_lines.append(points)
    # 硬边界误差
    total_ar = 0
    total_pts = 0
    for line in hard_lines:
        refs = []
        for p in line:
            p1, p2 = find_nearest(p[0], p[1], hard_edges)
            if len(p1) > 0 and len(p2) > 0:  # Found
                dis, p_ref = vertical_distance(p, p1, p2)
                # Abs_Rel计算公式： abs_rel = sqrt(a^2+b^2)/(H-y)
                # ar = math.sqrt(abs(p[0] - p_ref[0])) + 0.25 * abs(p[1] - p_ref[1])
                sq_ab = (p[0] - p_ref[0]) * (p[0] - p_ref[0]) + (p[1] - p_ref[1]) * (p[1] - p_ref[1])
                ar = math.sqrt(sq_ab) / (h - p[1])
                refs.append(p_ref)
                total_ar += ar
                total_pts += 1
        hard_refs.append(refs)
    if total_pts > 0:  # len(hard_refs) > 0:
        hard_ar = total_ar / total_pts  # 平均值
    # 软边界误差
    total_ar = 0
    total_pts = 0
    for line in soft_lines:
        refs = []
        for p in line:
            p1, p2 = find_nearest(p[0], p[1], soft_edges)
            if len(p1) > 0 and len(p2) > 0:  # Found
                dis, p_ref = vertical_distance(p, p1, p2)
                # Abs_Rel计算公式：
                ar = math.sqrt(abs(p[0] - p_ref[0])) + 0.25 * abs(p[1] - p_ref[1])
                refs.append(p_ref)
                total_ar += ar
                total_pts += 1
        soft_refs.append(refs)
    if total_pts > 0:  # len(soft_refs):
        soft_ar = total_ar / total_pts  # 平均值
    result = {
        'name': image_name,
        'hard_abs_rel': hard_ar,
        'hard_lines': hard_lines,
        'hard_refs': hard_refs,
        'soft_abs_rel': soft_ar,
        'soft_lines': soft_lines,
        'soft_refs': soft_refs
        # TODO 图像上下部分误差计算
    }
    return result


def vertical_distance(p, p1, p2):
    """
    计算点 p 与 p1, p2 之间的垂直距离，并返回垂点作为距离参照点。
    :param p:
    :param p1:
    :param p2:
    :return:
    """
    p_ref = get_foot(p, p1, p2)
    # abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1)) / np.sqrt(np.square(x2-x1) + np.square(y2-y1))
    # dis = np.linalg.norm(np.cross(p1 - p, p - p2)) / np.linalg.norm(p1 - p)
    dis = distance_points(p, p_ref)
    return dis, p_ref


def get_foot(point_a, start_point, end_point):
    """
    求点到直线的垂点。
    :param point_a:
    :param start_point:
    :param end_point:
    :return:
    """
    start_x, start_y = start_point
    end_x, end_y = end_point
    pa_x,pa_y = point_a

    p_foot = [0, 0]
    if start_point[0] == end_point[0]:
        p_foot[0] = start_point[0]
        p_foot[1] = point_a[1]
        return p_foot

    k = (end_y - start_y) * 1.0 / (end_x - start_x)
    a = k
    b = -1.0
    c = start_y - k * start_x
    p_foot[0] = int((b * b * pa_x - a * b * pa_y - a * c)/(a * a + b * b))
    p_foot[1] = int((a * a * pa_y - a * b * pa_x - b * c)/(a * a + b * b))

    return p_foot


def find_nearest(x, y, contours):
    """
    从轮廓线中寻找2个距离最近的连续点。
    :param x:
    :param y:
    :param contours:
    :return:
    """
    p1 = []
    p2 = []
    # print(len(contours))
    min_dis = 0
    for cnt in contours:
        points1 = np.array(cnt)
        points2 = points1.flatten()
        for i in range(len(points2) - 4):
            if i % 2 == 0:  # 2个点
                x1 = points2[i+0]  # 轮廓线上坐标是相反的
                y1 = points2[i+1]
                x2 = points2[i+2]
                y2 = points2[i+3]
                dis1 = distance_points([x, y], [x1, y1])
                dis2 = distance_points([x, y], [x2, y2])
                if min_dis == 0:
                    min_dis = dis1 + dis2
                    p1 = [x1, y1]
                    p2 = [x2, y2]
                elif min_dis > dis1 + dis2:
                    min_dis = dis1 + dis2
                    p1 = [x1, y1]
                    p2 = [x2, y2]
    return p1, p2


def distance_points(p1, p2):
    """
    平面上2点的距离计算。
    :param p1:
    :param p2:
    :return:
    """
    a = p2[1] - p1[1]
    b = p2[0] - p1[0]
    c = math.sqrt(a * a + b * b)
    return c


def main():
    print("")


if __name__ == '__main__':
    main()
