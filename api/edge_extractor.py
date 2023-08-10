"""
Soft/Hard edges extractor.
"""
import os
import cv2
import numpy as np
from shapely import ops
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import LinearRing
from shapely.geometry import LineString, MultiLineString

import io_helper
import edge_definitions


def analyze_edges(seg_mask, bbox_list, depth_info):
    """
    分析并提取图像的软硬边界。
    :param seg_mask: 图像分割结果。
    :param bbox_list: 目标检测结果。
    :param depth_info: 深度信息（预留接口）。
    :return:
    """
    # 获取并合并硬区域（可得到硬边界）
    hard_pixel = edge_definitions.edge_pixels['hard_edge']
    hard_areas, hard_stats = merge_hard_areas(seg_mask)
    # 用硬区域去指导并获得剩余软区域（可得到软边界）
    soft_pixel = edge_definitions.edge_pixels['soft_edge']
    soft_areas, soft_stats = get_soft_areas(seg_mask, hard_areas, hard_stats)
    # 取检测结果的块（自下而上排序）
    # TODO
    # 返回边界线
    soft_edges = []
    for i, area in enumerate(soft_areas):
        contours, hierarchy = get_area_contours(area, soft_stats[i])
        soft_edges.append(contours)
    hard_edges = []
    for i, area in enumerate(hard_areas):
        contours, hierarchy = get_area_contours(area, hard_stats[i])
        hard_edges.append(contours)
    return {
        soft_pixel: soft_edges,
        hard_pixel: hard_edges
    }


def get_contours(mask):
    """
    Analyze and obtain contours of a mask using OpenCV.
    :param mask:
    :return:
    """
    mask = np.where(mask > 1, 1, mask)
    mask = mask.astype(np.uint8)  # 重要！！！
    mask = cv2.Canny(mask, 0, -1, L2gradient=True)
    # kernel = np.ones((3, 3), dtype=np.uint8)
    # mask = cv2.dilate(mask, kernel, 1)
    # ret, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    # 问题：直接画轮廓有一些边界是断的，具体参见测试代码
    # 解决：改为直接给2个相邻区域画轮廓？？？
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # hierarchy 用法：https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
    # [neighbor, previous, child, parent]
    # TODO 去掉小的轮廓
    # https://stackoverflow.com/questions/60259169/how-to-group-nearby-contours-in-opencv-python-zebra-crossing-detection
    return contours, hierarchy


def get_binary_contours(binary_mask):
    """
    分割结果已经二值化，直接取轮廓线。
    :param binary_mask:
    :return:
    """
    pixel_values = np.unique(binary_mask)
    # assert len(pixel_values) == 2
    if len(pixel_values) == 2:
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return contours, hierarchy
    return None, None


def get_area_contours(area, stats):
    """
    提取某一个区域的轮廓线，保证一个区域的轮廓线只有一条。
    :param stats:
    :param area:
    :return:
    """
    h, w = area.shape
    # area = area.astype(np.uint8)  # 重要！！！
    area = cv2.Canny(area, 0, -1, L2gradient=True)
    # 保证一个区域只有一条轮廓线
    if is_side_area(stats, w, h) is not None:
        kernel = np.ones((3, 3), dtype=np.uint8)
        area = cv2.dilate(area, kernel, 1)
    contours, hierarchy = cv2.findContours(area, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # p = 5  # padding size
    # if sides is not None:
    #     if sides[0] == 1:
    #         area = np.pad(area, ((0, p), (0, 0)), 'constant', constant_values=(0, 0))
    #         stats[0] = stats[0] + p  # x + 1
    #     if sides[1] == 1:
    #         area = np.pad(area, ((p, 0), (0, 0)), 'constant', constant_values=(0, 0))
    #         stats[1] = stats[1] + p  # y + 1
    #     if sides[2] == 1:
    #         area = np.pad(area, ((0, 0), (0, p)), 'constant', constant_values=(0, 0))
    #     if sides[3] == 1:
    #         area = np.pad(area, ((0, 0), (p, 0)), 'constant', constant_values=(0, 0))
    # for i, cnt in enumerate(contours):
    #     area = cv2.contourArea(cnt)
    #     # Small contours are ignored.
    #     # if area < 500: # 可以计算面积，去掉小的
    #     #     cv2.fillPoly(thresh_gray, pts=[c], color=0)
    #     #     continue
    #     # rect = cv2.minAreaRect(cnt)
    #     # box = cv2.boxPoints(rect)
    #     print(cnt)
    #     first_point = cnt[0, :, :][0]
    #     last_point = cnt[-1, :, :][0]
    #     print(first_point)
    #     print(last_point)
    return contours, hierarchy


def is_side_area(stats, im_width, im_height):
    """
    判断区域是否在图像靠边。
    :param stats: 每个连通区域的外接矩形的[x,y,width,height,面积]
    :param im_width:
    :param im_height:
    :return: None of not side area; or [0, 0, 0, 0] for each side (left, top, right, bottom).
    """
    [x, y, width, height, _] = stats
    sides = np.zeros(4)
    if x == 0:
        sides[0] = 1  # left side
    if y == 0:
        sides[1] = 1  # top side
    if x + width == im_width:
        sides[2] = 1  # right side
    if y + height == im_height:
        sides[3] = 1  # bottom side
    if np.max(sides) == 1:
        return sides
    else:
        return None


def get_soft_areas(seg_mask, hard_areas, hard_stats):
    """
    根据确定的硬区域，选择图像剩余的软区域。
    :param seg_mask:
    :param hard_areas:
    :param hard_stats:
    :return:
    """
    hard_pixel = edge_definitions.edge_pixels['hard_edge']
    # soft_pixel = edge_definitions.edge_pixels['soft_edge']
    for h_area in hard_areas:  # 把硬区域像素去除
        seg_mask = np.where(h_area == hard_pixel, 0, seg_mask)
    soft_areas, soft_stats = sorted_areas(seg_mask, True)
    # 去掉面积过小的区域（硬区域不会太小！！！）
    selected_areas = []
    selected_stats = []
    for i, area in enumerate(soft_areas):  # 检查硬区域的位置
        stats_i = soft_stats[i]
        num_area = stats_i[4]  # [x,y,width,height,面积]
        if num_area < 200:  # 或者改为图像占比
            continue  # 忽略
        selected_areas.append(area)
        selected_stats.append(stats_i)
    return selected_areas, selected_stats


def get_hard_edges(seg_mask):
    """
    新实现：基于Shapely实现硬边界提取。
    :param seg_mask:
    :return:
    """
    h, w = seg_mask.shape
    image_polygon = Polygon([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    seg_mask = seg_mask.astype(np.uint8)
    # 重新映射 soft=0 和 hard=1 的连通区域
    # soft_pixel = edge_definitions.edge_pixels['soft_edge']
    hard_pixel = edge_definitions.edge_pixels['hard_edge']
    soft_class_ids = edge_definitions.seg_soft_class_ids
    # 通过映射，得到硬区域
    hard_mask = seg_mask.copy()
    for class_id in edge_definitions.seg_id_class.keys():
        if class_id in soft_class_ids:  # 映射所有软分类
            hard_mask = np.where(hard_mask == class_id, 0, hard_mask)
        else:  # 其他类 = 硬分类
            hard_mask = np.where(hard_mask == class_id, 1, hard_mask)
    hard_contours, _ = get_binary_contours(hard_mask)  # 取硬区域的边界线
    hard_polygons = bottom_up_polygons(hard_contours)
    # 通过Shapely API处理硬区域关系
    found_hard_edges = []
    found_hard_polygons = []
    for i, polygon in enumerate(hard_polygons):
        [min_x, min_y, max_x, max_y] = polygon.bounds
        if polygon_contains(found_hard_polygons, polygon):
            continue  # （左右合并之后，）已包含区域，跳过
        new_polygon = None
        new_edge = None
        # if 横向跨越，合并后面的Polygon
        if max_x - min_x == w - 1:  # and i + 1 < len(hard_polygons):
            merged_polygon, left_right_edge = merge_top_polygons(image_polygon, polygon)
            new_polygon = merged_polygon
            new_edge = left_right_edge
        # if 纵向跨越，依据底边中心位置，合并左边或右边的Polygon
        elif max_y - min_y == h - 1:  # and i + 1 < len(hard_polygons):
            merged_polygon, top_bottom_edge = merge_horizonal_polygons(image_polygon, polygon)
            new_polygon = merged_polygon
            new_edge = top_bottom_edge
        elif 0 in polygon.bounds:  # 区域有一边在边框上
            merged_polygon, side_edge = merge_side_polygons(image_polygon, polygon)
            new_polygon = merged_polygon
            new_edge = side_edge
        else:  # 独立存在的区域
            # found_hard_edges.append(polygon.boundary)
            # found_hard_polygons.append(polygon)
            new_polygon = polygon
            new_edge = polygon.boundary
        # 检查得到的新区域，去掉覆盖的区域
        clean_hard_polygons = []
        clean_hard_edges = []
        for j, got_pg in enumerate(found_hard_polygons):  # 检查已处理区域
            got_edge = found_hard_edges[j]  # 去掉已处理但被包含区域
            if not new_polygon.contains(got_pg):
                clean_hard_polygons.append(got_pg)
                clean_hard_edges.append(got_edge)
        # 清理之后，重置之前的列表
        found_hard_edges = clean_hard_edges
        found_hard_polygons = clean_hard_polygons
        found_hard_edges.append(new_edge)
        found_hard_polygons.append(new_polygon)
    # 统一把边界整理为points
    hard_edges_in_points = []
    for line in found_hard_edges:
        points_list = edge_in_points(line)
        for points in points_list:
            hard_edges_in_points.append(points)
    return found_hard_polygons, hard_edges_in_points


def get_soft_edges(seg_mask, hard_polygons, hard_edges):
    """
    新实现：基于Shapely实现软边界提取。
    :param seg_mask:
    :param hard_polygons:
    :param hard_edges:
    :return:
    """
    h, w = seg_mask.shape
    image_polygon = Polygon([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    seg_mask = seg_mask.astype(np.uint8)
    soft_class_ids = edge_definitions.seg_soft_class_ids
    # 通过映射，得到整个软区域
    whole_hard_polygon = None
    for i, h_pg in enumerate(hard_polygons):
        if i == 0:
            whole_hard_polygon = h_pg
        else:
            whole_hard_polygon = whole_hard_polygon.union(h_pg)
    whole_soft_polygon = image_polygon.difference(whole_hard_polygon)
    # 通过映射，得到各个软区域
    all_soft_polygons = []
    for class_id in soft_class_ids:  # 映射所有软分类
        soft_mask = seg_mask.copy()
        soft_mask = np.where(soft_mask > class_id, 0, soft_mask)
        soft_mask = np.where(soft_mask < class_id, 0, soft_mask)
        # soft_mask = np.where(soft_mask == class_id, 1, soft_mask)
        soft_contours, _ = get_binary_contours(soft_mask)  # 取软区域的边界线
        if soft_contours is not None:
            soft_polygons = bottom_up_polygons(soft_contours)
            for polygon in soft_polygons:
                all_soft_polygons.append(polygon)
    # 处理软区域关系
    soft_edges = []
    soft_polygons = []
    left_soft_polygon = whole_soft_polygon
    for i, polygon in enumerate(all_soft_polygons):
        # is_contained = polygon_contains(hard_polygons, polygon)
        is_contained = False
        for h_pg in hard_polygons:
            if h_pg.contains(polygon):
                is_contained = True
                # continue  # 已包含在硬区域的，跳过
        if not is_contained:  # 取软区域的边（去掉边框上的点）
            # done_pg = image_polygon.difference(left_soft_polygon)
            # diff_pg = image_polygon.difference(polygon)
            # diff_pg = diff_pg.union(done_pg)
            # diff_pg = diff_pg.intersection(left_soft_polygon)
            # multiline = diff_pg.intersection(polygon)
            # soft_edges.append(ops.linemerge(multiline))
            # 继续处理剩余区域
            # left_soft_polygon = left_soft_polygon.difference(polygon)
            diff_pg = whole_soft_polygon.difference(polygon)
            multiline = diff_pg.intersection(polygon)
            if not multiline.is_empty:
                soft_polygons.append(polygon)
                soft_edges.append(ops.linemerge(multiline))
            # soft_edges.append(whole_soft_polygon.boundary)
            # break

    # 边界后处理
    soft_edges_in_points = []
    for line in soft_edges:
        # 统一把边界整理为 Points
        points_list = edge_in_points(line)
        for points in points_list:
            left_points = points
            # 去掉与硬边界重复的边界
            # s_line_str = LineString(left_points)
            # is_overlapped = line_overlapped(hard_edges, left_points)
            # 保存有效边界
            # if not is_overlapped:
            soft_edges_in_points.append(left_points)

    return soft_polygons, soft_edges_in_points


def polygon_contains(polygons, targe_polygon):
    """
    左右合并时，不会记录已处理区域，因此需要判断。
    :param polygons:
    :param targe_polygon:
    :return:
    """
    for polygon in polygons:
        if polygon.contains(targe_polygon):
            return True
    return False


def merge_top_polygons(image_polygon, polygon):
    """
    遇到横穿图像的硬区域，合并上面的所有区域。
    :param image_polygon:
    :param polygon:
    :return:
    """
    diff = image_polygon.difference(polygon)
    diff_pgs = None
    if type(diff) is Polygon:
        diff_pgs = [diff]
    else:
        diff_pgs = list(diff)
    # [min_x, min_y, max_x, max_y] = polygon.bounds
    num_pgs = len(diff_pgs)
    multiline = None
    if num_pgs == 1:  # 只有1个，当前区域占据一半图像
        # 硬区域靠上或靠底部了，得到的区域为除本区域之外的部分
        rest_polygon = diff_pgs[0]
        multiline = polygon.intersection(rest_polygon)
    # elif num_pgs == 2:  # 在中间切开
    #     top_polygon = diff_pgs[0]
    #     polygon = polygon.union(top_polygon)
    #     bottom_polygon = diff_pgs[1]
    #     multiline = polygon.intersection(bottom_polygon)
    else:  # 有多个区域，把上面的区域全部合并
        # 检查最下面（最后一个区域）与硬边界的位置关系
        bottom_polygon = None
        for below_pg in diff_pgs:
            if below_pg.bounds[3] >= polygon.bounds[3]:
                # 把硬区域下面（最多持平）的区域合并，以便反向求上面（待合并）的区域
                if bottom_polygon is None:
                    bottom_polygon = below_pg
                else:
                    bottom_polygon = bottom_polygon.union(below_pg)
            # else:  # 出现一个在硬区域上面了，就不用继续了
            #     break
        # 上面的区域（含当前硬区域）全部合并
        top_polygon = image_polygon.difference(bottom_polygon)
        # bottom_polygon = diff_pgs[-1]
        # top_polygon = diff_pgs[0]
        # bottom_polygon = None
        # for i in range(1, len(diff_pgs)):
        #     pg = diff_pgs[i]
        #     if pg.bounds[3] < polygon.bounds[3]:
        #         top_polygon = top_polygon.union(pg)
        #     elif bottom_polygon is None:
        #         bottom_polygon = pg
        #     else:  # 下面的全部整合（通常只有1个）
        #         bottom_polygon = bottom_polygon.union(pg)
        # top_polygon = top_polygon.union(polygon)
        # 上面的其余区域+当前硬区域，就是最终的硬区域
        # polygon = top_polygon.union(polygon)
        polygon = top_polygon
        # 再求一次区域排除，避免下面有软区域影响边界
        bottom_polygon = image_polygon.difference(polygon)
        multiline = polygon.intersection(bottom_polygon)

    left_to_right_edge = ops.linemerge(multiline)
    return polygon, left_to_right_edge


def merge_side_polygons(image_polygon, polygon):
    """
    贴边的区域（可能还有内空），与外围区域求交操作才能去掉边框上的边。
    :param image_polygon:
    :param polygon:
    :return:
    """
    [_, _, h, w] = image_polygon.bounds
    # 如果靠边区域有内空，会出现多个备选区域
    diff_polygons = image_polygon.difference(polygon)
    if type(diff_polygons) is Polygon:
        diff_polygons = [diff_polygons]
    # 仅保留贴4边（占3个以上顶点）的最大外围区域
    for d_pg in diff_polygons:
        num_corners = 0
        if d_pg.boundary.contains(Point([0, 0])):
            num_corners += 1
        if d_pg.boundary.contains(Point([0, w])):
            num_corners += 1
        if d_pg.boundary.contains(Point([h, 0])):
            num_corners += 1
        if d_pg.boundary.contains(Point([h, w])):
            num_corners += 1
        if num_corners >= 3:
            polygon = image_polygon.difference(d_pg)
            multiline = d_pg.intersection(polygon)
            break
    return polygon, ops.linemerge(multiline)


def merge_horizonal_polygons(image_polygon, polygon):
    """
    遇到上下贯穿的区域，合并水平方向的区域。
    :param image_polygon:
    :param polygon:
    :return:
    """
    [min_x, min_y, max_x, max_y] = image_polygon.bounds
    im_w = max_x - min_x + 1
    im_h = max_y - min_y + 1
    bottom_mid = Point([im_h - 1, int(im_w / 2)])
    # 切开整个图像区域
    diff = image_polygon.difference(polygon)
    diff_pgs = []
    if type(diff) is Polygon:
        diff_pgs = [diff]
    else:
        diff_pgs = list(diff)
    # assert len(diff_pgs) >= 1
    num_diffs = len(diff_pgs)
    left_pg = None
    right_pg = None
    if polygon.contains(bottom_mid):  # 底边中点在硬形状中，合并右边
        left_pg = diff_pgs[0]  # 左边区域是在硬区域左侧的（只有一个）
        if num_diffs == 1:  # 只剩下一边
            right_pg = polygon
        # elif num_diffs == 2:  #
        #     right_pg = polygon.union(diff_pgs[1])
        # else:  # 超2个，右边因奇特形状产生多个区域
        #     for i in range(len(diff_pgs)):
        #         if i == 0:  # 跳过第一块
        #             right_pg = polygon  # 合并右边全部
        #         right_pg = right_pg.union(diff_pgs[i])
        #     polygon = right_pg
        else:  # 硬区域右边全部整合，直接用全图像求
            right_pg = image_polygon.difference(left_pg)
            polygon = right_pg
    else:  # 底边中点不在硬形状中，合并左边
        if num_diffs == 1:  # 只剩下一边区域（右边的）
            left_pg = polygon
        else:  # 硬区域右边只有一个（最后一个）
            right_pg = diff_pgs[-1]
            # 左边全部整合为一个，直接用全图像求
            left_pg = image_polygon.difference(right_pg)
            polygon = left_pg  # 结果是左边这个区域
            # left_pg = polygon
            # right_pg = None
            # for i, pg in enumerate(diff_pgs):
            #     if pg.bounds[2] < polygon.bounds[2]:
            #         left_pg = left_pg.union(pg)
            #     elif right_pg is None:
            #         right_pg = pg
            #     else:
            #         right_pg = right_pg.union(pg)

    # 左右两的区域交线，即为硬边界
    multiline = left_pg.intersection(right_pg)
    top_to_bottom_edge = ops.linemerge(multiline)
    return polygon, top_to_bottom_edge


def bottom_up_polygons(contours):
    """
    根据轮廓线创建Polygon，并从下往上排序。
    :param contours:
    :return:
    """
    polygons = []
    for cnt in reversed(contours):  # 轮廓线是从上到下的
        polygon = Polygon(contour_to_polygon(cnt))
        if polygon is not None and polygon.area > 300:
            polygons.append(polygon)
    return polygons


def edge_in_points(linestring):
    """
    将LineString或MultiLineString转换为点列表。
    :param linestring:
    :return:
    """
    lines = []
    if type(linestring) is LineString:
        lines.append(linestring.coords)
    elif type(linestring) is MultiLineString:
        for line in linestring:
            lines.append(line.coords)
    return lines


def line_overlapped(line_list, points):
    """
    判断一组线中是否包含指定的线条
    :param line_list:
    :param points:
    :return:
    """
    num_points = len(points)
    # 只用3个点判断，提高效率
    tests = [0, int(num_points)/2, num_points-1]
    is_overlapped = False
    for line in line_list:
        line_str = LineString(line)
        overlaps = 0
        for i in tests:
            p = points[i]
            if line_str.contains(Point(p)):
                overlaps += 1
        if overlaps == len(tests):
            is_overlapped = True
    return is_overlapped


def contour_to_polygon(contour):
    """
    Convert a contour of an image to polygon,
    for positioning by Shapely APIs.
    :param contour:
    :return:
    """
    points = contour_to_points(contour)
    if len(points) > 2:
        return Polygon(points)
    else:
        return None


def contour_to_points(contour):
    points = []
    flat_points = contour.flatten()
    for i in range(len(flat_points)-1):
        if i % 2 == 0:
            p = (flat_points[i], flat_points[i+1])
            points.append(p)
    return points


def merge_hard_areas(seg_mask):
    """
    将所有分割结果映射为软硬区域。
    目的：（1）划分硬边界；（2）后续目标位置判断。
    步骤：（1）先作标签映射，像素改为软、硬2种；（2）通过OpenCV分割2个或多个连通域。
    结果：由于软区域定义为0，故只返回了硬区域；区域之外（像素=0）的均为软区域。
    :param seg_mask: 分割结果。
    :return: 软、硬连通区域，至少有2个。有时候会超过2个，因为连通域可能断开。
    """
    # 重新映射 soft=0 和 hard=1 的连通区域
    soft_pixel = edge_definitions.edge_pixels['soft_edge']
    hard_pixel = edge_definitions.edge_pixels['hard_edge']
    soft_class_ids = edge_definitions.seg_soft_class_ids
    for class_id in edge_definitions.seg_id_class.keys():
        if class_id in soft_class_ids:  # 映射所有软分类
            seg_mask = np.where(seg_mask == class_id, soft_pixel, seg_mask)
        else:  # if class_id not in soft_class_ids:  # 其他类 = 硬分类
            seg_mask = np.where(seg_mask == class_id, hard_pixel, seg_mask)
    # 硬区域二次处理
    seg_mask = expand_hard_areas(seg_mask)
    areas, stats = sorted_areas(seg_mask)
    # 去掉面积过小的区域（硬区域不会太小！！！）
    selected_areas = []
    selected_stats = []
    for i, area in enumerate(areas):  # 检查硬区域的位置
        stats_i = stats[i]
        num_area = stats_i[4]  # [x,y,width,height,面积]
        if num_area < 2000:  # 或者改为图像占比
            continue  # 忽略
        selected_areas.append(area)
        selected_stats.append(stats_i)
    # 合并之后，通常只剩下1个硬区域
    return selected_areas, selected_stats


def expand_hard_areas(seg_mask):
    """
    根据硬区域的位置，使之扩展以覆盖当前区域之外的区域。
    :param seg_mask:
    :return:
    """
    h, w = seg_mask.shape
    areas, stats = sorted_areas(seg_mask)
    hard_pixel = edge_definitions.edge_pixels['hard_edge']
    # 如果有多个区域，过滤小区域和硬边界以外区域
    for i, area in enumerate(areas):  # 检查硬区域的位置
        stats_i = stats[i]
        sides = is_side_area(stats_i, w, h)
        if sides is not None:  #
            if sides[0] == 1 and sides[2] == 1:
                # # 横向跨整个图像，去掉上面的软区域（自下而上）
                seg_mask = expand_area_topmost(seg_mask, area, stats_i, hard_pixel)
                # return expand_hard_areas(seg_mask)  # 递归
            elif sides[1] == 1 and sides[3] == 1:  # top + bottom sides
                # 纵向跨整个图像，判断下方中点位置（车位置）
                seg_mask = expand_area_horizontally(seg_mask, area, stats_i, hard_pixel)
    return seg_mask


def expand_area_topmost(seg_mask, area, stats, pixel):
    """
    !!! 草坪奇形怪状时会出错 !!!
    将区域延伸/扩张到图像最上方。如果是硬区域，相当于忽略硬边界以外（上）的所有软区域。
    :param seg_mask:
    :param area:
    :param stats:
    :param pixel: 像素值，一般为 1（硬），pixel = np.amax(area)。
    :return:
    """
    im_h, im_w = area.shape
    # stats: [x, y, width, height, 面积]
    [x, y, w, h, a] = stats
    assert im_w == w  # 只允许横跨图像的区域操作
    # 从下往上检查
    # seg_mask[:y, :] = pixel  # 区域顶点至图像顶边，全部修改
    for i in range(im_w):  # 检查所有列
        inside = False
        end_row = im_h - 1
        # !!! 草坪奇形怪状时会出错。
        for j in reversed(range(im_h)):  # 从下到上
            if not inside and area[j, i] == pixel:
                end_row = j
                inside = True  # 该列进入区域范围
            # if inside and area[j, i] == 0:  # 区域范围结束
                seg_mask[:end_row, i] = pixel  # 把该点以上像素填充
                break  # 不需要继续处理
        # end_row = 0
        # for j in range(im_h):  # 从上到下
    # 左上角位置及高度改变
    # stats = [0, 0, w, y + h - 1, a]
    return seg_mask


def expand_area_horizontally(seg_mask, area, stats, pixel):
    """
    !!! 草坪奇形怪状时会出错 !!!
    将区域横向延伸/扩张到图像左边或右边。左右扩张方向取决于下方中心位置是否在区域内。
    下方中心点（相机位置） 不在区域内，则向左扩张（等同于忽略左边区域）；
    下方中心点（相机位置） 在区域内，则向右扩张（等同于忽略右边区域）。
    :param seg_mask:
    :param area:
    :param stats:
    :param pixel:
    :return:
    """
    im_h, im_w = area.shape
    bottom_mid = [im_h-1, int(im_w/2)]
    bottom_mid_pixel = area[im_h-1, int(im_w/2)]
    right_to_left = None  # 检测顺序
    if bottom_mid_pixel == pixel:  # 相机位置在区域内，向右
        # order_cols = range(im_w)  # 递增
        right_to_left = False
    else:  # 相机位置不在区域内，向左
        # order_cols = reversed(range(im_w))  # 递减
        right_to_left = True
    for i in range(im_h):  # 检查所有行
        inside = False
        col_start = 0  # 向右
        col_end = im_w - 1
        col_orders = range(im_w)
        if right_to_left:  # 向左
            col_orders = reversed(range(im_w))
        # !!! 草坪奇形怪状时会出错。
        for j in col_orders:  # 左或右检查
            if not inside and area[i, j] == pixel:
                if right_to_left:  # 向左，开始列不变
                    col_end = j  # 确定结束列
                else:  # 向右，结束列不变
                    col_start = j  # 确定开始列
                inside = True  # 该列进入区域范围
            # if inside and area[j, i] == 0:  # 区域范围结束
                seg_mask[i, col_start:col_end] = pixel  # 把该点以上像素填充
                break  # 不需要继续处理

    return seg_mask


def soft_hard_edges(mask, area, stats):
    """
    检测区域的边界线，确定其软、硬属性。
    :param mask: 分割结果。用于判断边界线的类型。
    :param area: 分析的区域
    :param stats:
    :return:
    """
    class_id = np.amax(area)
    contours, hierarchy = get_area_contours(area, stats)
    if class_id not in edge_definitions.seg_soft_class_ids:
        # 当前类就是硬区域，其边界就是硬边界（按道理不用判断）
        return {  # 全部为硬边界
            edge_definitions.edge_pixels['soft_edge']: [],
            edge_definitions.edge_pixels['hard_edge']: contours
        }
    # 分析软区域的边界，根据相邻区域的性质决定是软或硬边界
    soft_contours = []
    hard_contours = []
    hard_pixel = edge_definitions.edge_pixels['hard_edge']
    soft_pixel = edge_definitions.edge_pixels['soft_edge']
    for contour in contours:
        s_cnt = []  # soft points
        h_cnt = []  # hard points
        # https://www.geeksforgeeks.org/find-co-ordinates-of-contours-using-opencv-python/
        # approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True)
        # draws boundary of contours.
        # cv2.drawContours(img2, [approx], 0, (0, 0, 255), 5)
        # Used to flatted the array containing the co-ordinates of the vertices.
        # n = approx.ravel()
        # i = 0
        # for j in n:
        #     if i % 2 == 0:
        #         x = n[i]
        #         y = n[i + 1]
        #         print("Point %d" % int(i / 2))
        #         if soft_hard_point(mask, class_id, x, y) == soft_pixel:
        #             s_cnt.append((x, y))
        #         elif soft_hard_point(mask, class_id, x, y) == hard_pixel:
        #             h_cnt.append((x, y))
        #     i = i + 1
        # 检查每一个点
        num_points, _, _ = contour.shape
        for i in range(num_points):
            xy = contour[i, :]
            x = xy[0][1]
            if x >= mask.shape[0]:
                print(i)
            y = xy[0][0]
            if soft_hard_point(mask, class_id, x, y, 1) == soft_pixel:
                s_cnt.append((y, x))
            elif soft_hard_point(mask, class_id, x, y, 1) == hard_pixel:
                h_cnt.append((y, x))
        # 保存拆分之后的软硬边界
        if len(s_cnt) > 0:
            soft_contours.append(contour_from_points(s_cnt))
        if len(h_cnt) > 0:
            hard_contours.append(contour_from_points(h_cnt))
    return {
        soft_pixel: soft_contours,
        hard_pixel: hard_contours
    }


def soft_hard_point(mask, class_id, x, y, p=3):
    """
    判断边界上某一点（边界线顶点）的软、硬属性。
    :param mask:
    :param class_id: 当前类
    :param x:
    :param y:
    :param p: 用于检查的点数
    :return:
    """
    if class_id not in edge_definitions.seg_soft_class_ids:
        # 当前类就是硬区域，其边界就是硬边界（按道理不用判断）
        return edge_definitions.edge_pixels['hard_edge']
    # h, w = mask.shape
    # pixel = mask[x, y]
    cover_ids = np.unique(pad_point(mask, x, y, p))
    if class_id not in cover_ids:  # 增加框大小
        cover_ids = np.unique(pad_point(mask, x, y, p+2))
    if class_id not in cover_ids:  #
        print("WARNING: box does not cover class id %d" % class_id)
    for near_id in cover_ids:  # 检查像素所在的分类
        if near_id not in edge_definitions.seg_soft_class_ids:
            # 发现硬边界区域
            return edge_definitions.edge_pixels['hard_edge']
    return edge_definitions.edge_pixels['soft_edge']


def pad_point(mask, x, y, p=3):
    """
    围绕点取一个小块区域，用于检查边界性质。
    :param mask:
    :param x:
    :param y:
    :param p: padding width
    :return:
    """
    h, w = mask.shape
    if x - p >= 0:
        left = x - p
    else:
        left = 0
    if x + p < w:
        right = x + p
    else:
        right = w
    if y - p >= 0:
        top = y - p
    else:
        top = 0
    if y + p < h:
        bottom = y + p
    else:
        bottom = h
    cover_box = mask[left:right, top:bottom]
    return cover_box


def contour_from_points(points):
    """
    https://stackoverflow.com/questions/50671524/how-to-draw-lines-between-points-in-opencv
    :param points:
    :return:
    """
    return np.array(points)


def close_contour(contours, width, height):
    """
    封闭轮廓线，使之成为一个形状，以判断它在图片上的位置。
    :param contours:
    :param width:
    :param height:
    :return:
    """
    tmp = np.zeros((height, width), np.uint8)
    image = io_helper.draw_edges(tmp, contours, 1)
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(image, kernel, iterations=3)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = get_contours(closing)
    print(len(contours))
    print(hierarchy)
    return contours, hierarchy


def segment_of_class(mask, class_id):
    """
    Get the area(s) from segmented mask by class id,
    before analyzing the contours using contours().
    :param mask:
    :param class_id:
    :return:
    """
    class_ids = np.unique(mask)
    if class_id not in class_ids:
        print("WARNING: no such class id.")
        return None
    # 仅保留指定类的分割结果（性能可优化，如一次性分离所有类）
    # 具体参见：areas_by_class()
    # area = np.where(mask != class_id, 0, mask)  # Not working
    segment = np.where(mask > class_id, 0, mask)
    segment = np.where(segment < class_id, 0, segment)

    # segment = segment.astype(np.uint8)
    # kernel = np.ones((15, 15), dtype=np.uint8)
    # # area = cv2.erode(area, kernel2, 1)
    # segment = cv2.morphologyEx(segment, cv2.MORPH_OPEN, kernel)
    # print(np.amax(segment))
    return segment


def areas_of_segment(segment_mask):
    """
    一个类的分割结果可能有多个区域，如多块草坪。
    通过连通域分析，从分割结果分离出来多个区域。
    如果需要自下而上，遍历时使用 for area in reversed(areas)
    潜在问题：分割效果不好的时，可能有一些区域可能很小。
    有利因素：图像下方（近）分割块通常效果好一些。
    :param segment_mask:
    :return:
    """
    assert segment_mask is not None
    h, w = segment_mask.shape  # 图像大小
    # 最大值即为类ID，其余值为0
    class_id = np.amax(segment_mask)
    # 连通域提取 http://www.codebaoku.com/it-python/it-python-250602.html
    segment_mask = segment_mask.astype(np.uint8)  # 重要！！！
    # num_labels, labels = cv2.connectedComponents(segment_mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segment_mask, connectivity=8)
    # 提取连通区域
    areas = []
    areas_stats = []
    for i in range(1, num_labels):
        index_xy = labels == i
        area = np.zeros((h, w), np.uint8)
        area[index_xy] = class_id  # 该区域赋值原始class_id
        # 去掉较小面积的区域
        # unique, counts = np.unique(area, return_counts=True)
        # class_counts = dict(zip(unique, counts))
        # num_pixels = class_counts[class_id]
        # 每个连通区域的外接矩形的[x,y,width,height,面积]
        num_pixels = stats[i, 4]
        if num_pixels > 100:  # 大于 10 个像素（需要调试）
            areas.append(area)
            areas_stats.append(stats[i, :])
    print("Found %d areas of class_id=%d" % (len(areas), class_id))
    return areas, areas_stats


def all_areas(mask):
    """
    将分割结果按照类划分成子区域。
    如需要按位置排序，请使用 sorted_areas()。
    :param mask: 图像分割结果。
    :return:
    """
    class_ids = np.unique(mask)
    areas = []
    stats = []
    for class_id in class_ids:
        segment_mask = segment_of_class(mask, class_id)
        areas_class, areas_stats = areas_of_segment(segment_mask)
        for i, area in enumerate(areas_class):
            areas.append(area)
            stats.append(areas_stats[i])
    return areas, stats


def sorted_areas(mask, bottom_up=True):
    """
    按分割区域位置从下到上的顺序排列类的区域，以支持自下而上边界分析。
    问题：同一个类的分割块可能有多个。
    解决：改为使用 areas_of_segment()，支持自下而上排序。
    :param mask:
    :param bottom_up:
    :return:
    """
    assert mask is not None
    h, w = mask.shape
    # 所有区域
    areas, stats = all_areas(mask)
    new_areas = []
    new_stats = []
    # stats: 每个连通区域的外接矩形的[x,y,width,height,面积]
    sorted_index = []
    while len(sorted_index) < len(areas):
        # Find the large y of bbox of area
        for i in range(len(areas)):  # check each area
            if i not in sorted_index:
                stat_i = stats[i]  # position
                bottom_yi = stat_i[1] + stat_i[3]
                pick_index = i  # 找到的位置
                for j in range(len(areas)):
                    # compare with the rest
                    if (j != i) and (j not in sorted_index):
                        stat_j = stats[j]  # position
                        bottom_yj = stat_j[1] + stat_j[3]
                        if bottom_up and bottom_yi < bottom_yj:
                            # 自下而上，找最大值
                            bottom_yi = bottom_yj
                            pick_index = j
                        elif not bottom_up and bottom_yi > bottom_yj:
                            # 自上而下，找最小值
                            bottom_yi = bottom_yj
                            pick_index = j
                # 记录当前找到的值
                sorted_index.append(pick_index)
                new_areas.append(areas[pick_index])
                new_stats.append(stats[pick_index])
    print(sorted_index)
    return new_areas, new_stats


def adjacence_of_areas(area1, area2):
    """
    获取2个区域的相邻边（线）。
    :param area1:
    :param area2:
    :return:
    """
    return []


def union_areas(area1, area2):
    """
    合并2个已处理相邻边的区域，以便继续合并其他相邻区域。
    :param area1:
    :param area2:
    :return:
    """
    area = None
    return area


def combine_area_bbox(area, bbox_inside):
    """
    当目标检测结果bbox跨越区域边界时，先融合区域与bbox。
    融合，实际上是从区域中挖去 bbox_inside 部分。
    :param area: 分割后的某个区域。
    :param bbox_inside: 检测框在区域之内的范围，由bbox_inside_area()得到。
    :return:
    """


def bbox_inside_area(bbox, area):
    """
    检查bbox是否在分割区域之内。
    :param bbox: 目标检测结果。
    :param area: 图像分割后的某个区域。
    :return: False - 不在区域之内；True - 在区域之内；[[bbox_inside],[bbox_outside]] - 跨区域边界。
    """
    print()


def main():
    test_image_1 = os.path.join(os.getcwd(), '../examples/images_3/test_1.jpg')
    test_mask = io_helper.read_seg_mask(test_image_1)
    h, w = test_mask.shape
    test_class_id = 11
    area = segment_of_class(test_mask, test_class_id)
    contours, hierarchy = get_contours(area)
    print(len(contours))
    print(hierarchy)
    contours, hierarchy = close_contour(contours, w, h)
    print(len(contours))
    print(hierarchy)


if __name__ == '__main__':
    main()
