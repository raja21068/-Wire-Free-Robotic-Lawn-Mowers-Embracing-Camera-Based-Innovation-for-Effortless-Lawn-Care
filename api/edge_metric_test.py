import os
import unittest
import numpy as np

import io_helper
import edge_extractor
import edge_definitions
import edge_metric

test_image_dir = os.path.join(os.getcwd(), '../examples/test/test/')
test_boundary_labels = os.path.join(os.getcwd(), '../examples/boundary/annotations.xml')
# test_image_1 = os.path.join(os.getcwd(), '../examples/images_3/test_1.jpg')
# test_image_1 = os.path.join(os.getcwd(), '../examples/images_4/2022_08_25_17_00_01.jpg')
# test_image_1 = os.path.join(os.getcwd(), '../examples/images_4/2022_08_25_16_52_12.jpg')
# test_image_1 = os.path.join(os.getcwd(), '../examples/images_4/2022_08_25_16_43_14.jpg')
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


class EdgeDemoTest(unittest.TestCase):

    def setUp(self):
        self.image = io_helper.read_image(test_image_1)
        self.mask = io_helper.read_seg_mask(test_image_1)
        self.filename = io_helper.filename_only(test_image_1)
        all_labels = io_helper.read_boundary_xml(test_boundary_labels)
        self.labels = io_helper.get_boundary_labels(all_labels, test_image_1)

    def xtest_find_nearest(self):
        areas, stats = edge_extractor.sorted_areas(self.mask)
        assert len(areas) > 0
        first_area = areas[0]
        contours, _ = edge_extractor.get_area_contours(first_area, stats[0])
        assert len(contours) > 0
        p1, p2 = edge_metric.find_nearest(510, 100, contours)
        assert p1 is not None

    def test_abs_rel(self):
        # edges = edge_extractor.analyze_edges(self.mask, None, None)
        hard_polygons, hard_edges = edge_extractor.get_hard_edges(self.mask)
        soft_polygons, soft_edges = edge_extractor.get_soft_edges(self.mask, hard_polygons, hard_edges)
        hard_pixel = edge_definitions.edge_pixels['hard_edge']
        soft_pixel = edge_definitions.edge_pixels['soft_edge']
        # soft_edges = edges[soft_pixel]
        # hard_edges = edges[hard_pixel]
        image = self.image.copy()
        file_path = self.filename + "_soft_hard_edges_metric.png"
        print("Draw inferenced edges: ")
        hard_colors = edge_definitions.hard_edge_colors
        soft_colors = edge_definitions.soft_edge_colors
        image = io_helper.draw_linestring(
            image, soft_edges, soft_colors[1], point_step=100, radius=1, thickness=1)
        image = io_helper.draw_linestring(
            image, hard_edges, hard_colors[1], point_step=100, radius=3, thickness=2)
        result = edge_metric.abs_rel(self.image, self.labels, hard_edges, soft_edges)
        assert result is not None
        print(result['hard_abs_rel'])
        abs_rel_text = "AbsRel = %.02f%%" % (result['hard_abs_rel']*100)
        if result['hard_lines'] is not None:
            image = io_helper.draw_lines(image, result['hard_lines'], hard_colors[0], thickness=2, radius=3)
            # image = io_helper.draw_lines(image, result['hard_refs'], hard_colors[1], thickness=1, radius=2)
        if result['soft_lines'] is not None:
            image = io_helper.draw_lines(image, result['soft_lines'], soft_colors[0], thickness=2, radius=3)
            # image = io_helper.draw_lines(image, result['soft_refs'], soft_colors[1], thickness=1, radius=2)

        io_helper.draw_text(image, abs_rel_text)
        io_helper.write_image(file_path, image)


if __name__ == "__main__":
    unittest.main()
