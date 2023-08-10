import os
import unittest
import numpy as np

import io_helper
import edge_extractor
import edge_definitions

from shapely.geometry import LineString, MultiLineString

test_image_dir = os.path.join(os.getcwd(), '../examples/images_3/')
# test_image_1 = os.path.join(os.getcwd(), '../examples/images_3/test_1.jpg')
# test_image_1 = os.path.join(os.getcwd(), '../examples/images_4/2022_08_25_17_00_01.jpg')  # ！！！
# test_image_1 = os.path.join(os.getcwd(), '../examples/images_4/2022_08_25_16_52_12.jpg')  # ！！！
# test_image_1 = os.path.join(os.getcwd(), '../examples/images_4/2022_08_25_16_43_14.jpg')  # ！！！
# test_image_1 = os.path.join(os.getcwd(), '../examples/images_4/2022_08_25_17_01_15.jpg')  # left-right example
# test_image_1 = os.path.join(os.getcwd(), '../examples/images_4/2022_08_25_16_44_27.jpg')
# test_image_1 = os.path.join(os.getcwd(), '../examples/images_4/2022_08_25_16_46_27.jpg')
# test_image_1 = os.path.join(os.getcwd(), '../examples/images_4/2022_08_25_17_00_33.jpg')
# test_image_1 = os.path.join(os.getcwd(), '../examples/images_4/2022_08_25_16_24_40.jpg')
# test_image_1 = os.path.join(os.getcwd(), '../examples/images_4/2022_08_25_16_48_48.jpg')
# test_image_1 = os.path.join(os.getcwd(), '../examples/images_4/2022_08_25_17_01_51.jpg')  # top-bottom example
# test_image_1 = os.path.join(os.getcwd(), '../examples/images_5/Panoptic_ChengDu_1_80_1200.jpg')
# test_image_1 = os.path.join(os.getcwd(), '../examples/images_5/Panoptic_ChengDu_1_97_1455.jpg')
# test_image_1 = os.path.join(os.getcwd(), '../examples/images_5/Panoptic_ChengDu_1_374_5610.jpg')
# test_image_1 = os.path.join(os.getcwd(), '../examples/images_5/Panoptic_ChengDu_2_125_1875.jpg')
test_image_1 = os.path.join(os.getcwd(), '../examples/images_5/Panoptic_ChengDu_2_148_2220.jpg')
test_image_1 = os.path.join(os.getcwd(), '../examples/images_5/Panoptic_ChengDu_2_164_2460.jpg')
# test_image_1 = os.path.join(os.getcwd(), '../examples/images_5/Panoptic_ChengDu_3_20_300.jpg')
# test_image_1 = os.path.join(os.getcwd(), '../examples/images_5/Panoptic_ChengDu_3_42_630.jpg')


class EdgeExtractorTest(unittest.TestCase):

    def setUp(self):
        self.image = io_helper.read_image(test_image_1)
        self.mask = io_helper.read_seg_mask(test_image_1)

    def xtest_area_of_class(self):
        test_class_id = 11
        area = edge_extractor.area_of_class(self.mask, test_class_id)
        print("Area class id %d" % np.amax(area))
        assert np.amax(area) == test_class_id

    def xtest_areas_by_class(self):
        areas = edge_extractor.areas_by_class(self.mask)
        print("First class id %d" % np.amax(areas[0]))
        assert areas is not None
        assert len(areas) > 0
        assert np.amax(areas[0]) > 0

    def xtest_sorted_areas(self):
        sorted_areas = edge_extractor.sorted_areas(self.mask)
        print("First class id %d" % np.amax(sorted_areas[0]))
        assert sorted_areas is not None
        assert len(sorted_areas) > 0
        assert np.amax(sorted_areas[0]) > 0

    def xtest_get_contours(self):
        test_class_id = 1
        segment = edge_extractor.segment_of_class(self.mask, test_class_id)
        contours, hierarchy = edge_extractor.get_contours(segment)
        print(len(contours))
        # print(contours)
        print(hierarchy)
        image = io_helper.read_image(test_image_1)
        filename = io_helper.filename_only(test_image_1)
        for i in range(len(contours)+1):
            file_path = filename + ("_%d.png" % i)
            image = io_helper.draw_edges(image, contours[0:i], 3)
            io_helper.write_image(file_path, image)
            # 根据轮廓线判断效果

    def xtest_areas_of_segment(self):
        test_class_id = 1
        segment = edge_extractor.segment_of_class(self.mask, test_class_id)
        areas, stats = edge_extractor.areas_of_segment(segment)
        assert len(areas) > 0
        assert np.amax(areas[0]) == test_class_id
        filename = io_helper.filename_only(test_image_1)
        areas = reversed(areas)  # 改为自下而上遍历
        for i, area in enumerate(areas):
            contours, hierarchy = edge_extractor.get_contours(area)
            file_path = filename + ("_%d.png" % i)
            image_copy = self.image.copy()
            for j in range(len(contours)+1):
                image_copy = io_helper.draw_edges(image_copy, contours, 3)
            io_helper.write_image(file_path, image_copy)
            # 根据轮廓线判断效果

    def xtest_sorted_areas(self):
        # sorted_areas, stats = edge_extractor.all_areas(self.mask)
        sorted_areas, stats = edge_extractor.sorted_areas(self.mask, True)
        # image_4_mask = io_helper.read_seg_mask(test_image_4)
        # sorted_areas, stats = edge_extractor.sorted_areas(image_4_mask, True)
        assert len(sorted_areas) > 0
        # image = io_helper.read_image(test_image_4)
        # filename = io_helper.filename_only(test_image_4)
        image = self.image
        filename = io_helper.filename_only(test_image_1)
        for i, area in enumerate(sorted_areas):
            # if i == 3:
            contours, hierarchy = edge_extractor.get_contours(area)
            file_path = filename + ("_%d.png" % i)
            image_copy = image.copy()
            for j in range(len(contours)):
                image_copy = io_helper.draw_edges(image_copy, contours, 3)
            io_helper.write_image(file_path, image_copy)
            # 根据轮廓线判断效果

    def xtest_get_contours_of_area(self):
        sorted_areas, stats = edge_extractor.sorted_areas(self.mask, True)
        assert len(sorted_areas) > 0
        first_area = sorted_areas[0]
        contours, hierarchy = edge_extractor.get_contours(first_area)
        filename = io_helper.filename_only(test_image_1)
        [h, w, _] = self.image.shape
        blank_image = np.zeros((h, w), np.uint8)
        # blank_image[:, :] = 255
        blank_image = io_helper.draw_edges(blank_image, contours, 1)
        # 第二次提取轮廓
        contours, hierarchy = edge_extractor.get_contours(blank_image)
        for i in range(len(contours) + 1):
            file_path = filename + ("_%d.png" % i)
            image = self.image.copy()
            image = io_helper.draw_edges(image, contours[0:i], 3)
            io_helper.write_image(file_path, image)
            # 根据轮廓线判断效果

    def xtest_get_area_contours(self):
        sorted_areas, stats = edge_extractor.sorted_areas(self.mask, True)
        assert len(sorted_areas) > 0
        first_area = sorted_areas[0]
        first_stats = stats[0]
        class_id = np.amax(first_area)
        filename = io_helper.filename_only(test_image_1)
        contours, hierarchy = edge_extractor.get_area_contours(first_area, first_stats)
        for i in range(len(contours)):
            file_path = filename + ("_area0_%d_cnt%d.png" % (class_id, i))
            image = self.image.copy()
            image = io_helper.draw_edges(image, [contours[i]], 3)
            io_helper.write_image(file_path, image)
            # 根据轮廓线判断效果

    def xtest_hard_areas(self):
        # 测试 image_1
        filename = io_helper.filename_only(test_image_1)
        sorted_areas, stats_list = edge_extractor.merge_hard_areas(self.mask)
        assert len(sorted_areas) > 0
        hard_pixel = edge_definitions.edge_pixels['hard_edge']
        # soft_pixel = edge_definitions.edge_pixels['soft_edge']
        for i, area in enumerate(sorted_areas):
            class_id = np.amax(area)
            stats = stats_list[i]
            # 先画出来看看
            contours, hierarchy = edge_extractor.get_area_contours(area, stats)
            # num_cnts = len(contours)
            file_path = filename + ("_area%d_%d_cnt.png" % (i, class_id))
            image = self.image.copy()
            image = io_helper.draw_edges(image, contours, 3, hard_pixel)
            io_helper.write_image(file_path, image)

    def xtest_soft_areas(self):
        # 测试 image_1
        filename = io_helper.filename_only(test_image_1)
        hard_areas, hard_stats = edge_extractor.merge_hard_areas(self.mask)
        assert len(hard_areas) > 0
        hard_pixel = edge_definitions.edge_pixels['hard_edge']
        soft_pixel = edge_definitions.edge_pixels['soft_edge']
        soft_areas, soft_stats = edge_extractor.get_soft_areas(self.mask, hard_areas, hard_stats)
        # 画软边界
        image = self.image.copy()
        file_path = filename + "_soft_hard_edges.png"
        for i, area in enumerate(soft_areas):
            class_id = np.amax(area)
            stats = soft_stats[i]
            # 先画出来看看
            contours, hierarchy = edge_extractor.get_area_contours(area, stats)
            # num_cnts = len(contours)
            # file_path = filename + ("_area%d_%d_cnt.png" % (i, class_id))
            image = io_helper.draw_edges(image, contours, 3, soft_pixel)
            # io_helper.write_image(file_path, image)
        # 画硬边界
        for i, area in enumerate(hard_areas):
            class_id = np.amax(area)
            stats = hard_stats[i]
            # 先画出来看看
            contours, hierarchy = edge_extractor.get_area_contours(area, stats)
            # num_cnts = len(contours)
            # file_path = filename + ("_area%d_%d_cnt.png" % (i, class_id))
            image = io_helper.draw_edges(image, contours, 3, hard_pixel)
        io_helper.write_image(file_path, image)

    def xtest_analyze_edges(self):
        filename = io_helper.filename_only(test_image_1)
        edges = edge_extractor.analyze_edges(self.mask, None, None)
        hard_pixel = edge_definitions.edge_pixels['hard_edge']
        soft_pixel = edge_definitions.edge_pixels['soft_edge']
        soft_edges = edges[soft_pixel]
        hard_edges = edges[hard_pixel]
        image = self.image.copy()
        file_path = filename + "_soft_hard_edges.png"
        for s_cnts in soft_edges:  # 先画软边界
            image = io_helper.draw_edges(image, s_cnts, 2, soft_pixel)
        for h_cnts in hard_edges:  # 再画硬边界
            image = io_helper.draw_edges(image, h_cnts, 3, hard_pixel)
        # 保存图片
        io_helper.write_image(file_path, image)

    def test_get_hard_edges(self):
        hard_polygons, hard_edges = edge_extractor.get_hard_edges(self.mask)
        soft_polygons, soft_edges = edge_extractor.get_soft_edges(self.mask, hard_polygons, hard_edges)
        image = self.image.copy()
        filename = io_helper.filename_only(test_image_1)
        # for i, edge in enumerate(soft_edges):
        #     image = self.image.copy()
        #     file_path = filename + ("_soft_edge_%d.png" % i)
        #     image = io_helper.draw_linestring(
        #         image, [edge], (50, 255, 50), point_step=100, radius=3, thickness=1)
        #     io_helper.write_image(file_path, image)
        file_path = filename + "_hard_soft_edges.png"
        image = self.image.copy()
        image = io_helper.draw_linestring(
            image, soft_edges, (50, 255, 50), point_step=100, radius=1, thickness=1)
        image = io_helper.draw_linestring(
            image, hard_edges, (50, 50, 250), point_step=100, radius=3, thickness=2)
        # for i, edge in enumerate(hard_edges):
        #     # image = self.image.copy()
        #     file_path = filename + ("_hard_edge_%d.png" % i)
        #     image = io_helper.draw_linestring(
        #         image, [edge], (50, 50, 250), point_step=100, radius=3, thickness=1)
        #     io_helper.write_image(file_path, image)
        # # 保存图片
        io_helper.write_image(file_path, image)


if __name__ == "__main__":
    unittest.main()
