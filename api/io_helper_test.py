import os
import unittest
import numpy as np

import io_helper

test_image_dir = os.path.join(os.getcwd(), '../examples/images_3/')


class IOHelperTest(unittest.TestCase):

    def setUp(self):
        print()

    def xtest_read_det_bbox(self):
        image_path = os.path.join(test_image_dir, "test_1.jpg")
        bbox_list = io_helper.read_det_bbox(image_path)
        assert bbox_list is not None
        assert len(bbox_list) > 0

    def xtest_read_seg_mask(self):
        image_path = os.path.join(test_image_dir, "test_1.jpg")
        seg_mask = io_helper.read_seg_mask(image_path)
        assert seg_mask is not None
        assert len(seg_mask) > 0

    def test_read_boundary_xml(self):
        label_path = os.path.join("../examples/boundary/annotations.xml")
        labels = io_helper.read_boundary_xml(label_path)
        assert len(labels) > 0


if __name__ == "__main__":
    unittest.main()
