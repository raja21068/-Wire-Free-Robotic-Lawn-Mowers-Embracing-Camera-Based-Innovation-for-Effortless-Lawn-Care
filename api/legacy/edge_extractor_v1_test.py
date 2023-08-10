""""""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest
import argparse

import numpy as np
from PIL import Image

import edge_extractor

test_image_dir = os.path.join(os.getcwd(), '../examples/test/test/')
test_image_det_dir = os.path.join(os.path.pardir, 'examples/test/test_det/')
test_image_seg_dir = os.path.join(os.path.pardir, 'examples/test/test_seg/')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', type=str, default="segmentation",
                        choices=edge_extractor.OPT_TYPES,  # ["segmentation", "seg", "detection", "det", "both"]
                        help="Base operation(s), segmentation, detection or both, for edge extraction.")
    parser.add_argument('image_path', type=str, help="Path of the target image.")
    parser.add_argument('--output_path', type=str, default='', help="Path to save the results.")
    parser.add_argument('--mask_path', type=str, help="Path of the segmentation mask.")
    parser.add_argument('--bbox_path', type=str, help="Path of the bounding boxes by detection.")
    parser.add_argument('--display', action='store_false', help="True or False option for display")
    return parser


class EdgeExtractorTest(unittest.TestCase):

    def setUp(self):
        self.parser = create_parser()
        self.args = self.parser.parse_args([
            # 'seg',  # Testing segmentation only.
            # os.path.join(test_image_dir, "test_1.jpg"),  # 'image_path'
            # '--mask_path', os.path.join(os.getcwd(), '../examples/test/test_seg/labels/test_1.txt'),
            # '--bbox_path', os.path.join(os.getcwd(), '../examples/test/test_det/labels/test_1.txt'),
            # 'det',  # Testing detection only.
            # os.path.join(test_image_dir, "test_3.jpg"),  # 'image_path'
            # '--mask_path', os.path.join(os.getcwd(), '../examples/test/test_seg/labels/test_3.txt'),
            # '--bbox_path', os.path.join(os.getcwd(), '../examples/test/test_det/labels/test_3.txt'),
            'both',  # Testing both segmentation and detection.
            os.path.join(test_image_dir, "test_2.jpg"),  # 'image_path'
            '--mask_path', os.path.join(os.getcwd(), '../examples/test/test_seg/labels/test_2.txt'),
            '--bbox_path', os.path.join(os.getcwd(), '../examples/test/test_det/labels/test_2.txt'),
            # os.path.join(test_image_dir, "test_2.jpg"),  # 'image_path'
            '--output_path', os.path.join(os.getcwd(), '../examples/test/output/')
        ])
        # self.assertEqual(parsed.something, 'test')
        # print("Load examples from " + test_image_dir)
        # self.test_image1 = Image.open(os.path.join(test_image_dir, "test_1.jpg"))
        # self.test_image2 = Image.open(os.path.join(test_image_dir, "test_2.jpg"))
        # self.test_image3 = Image.open(os.path.join(test_image_dir, "test_3.jpg"))
        # self.test_image1 = np.asarray(self.test_image1)
        # self.test_image2 = np.asarray(self.test_image2)
        # self.test_image3 = np.asarray(self.test_image3)
        # self.assertIsNotNone(self.test_image1)
        # self.assertIsNotNone(self.test_image2)
        # self.assertIsNotNone(self.test_image3)

    def testReadDetResults(self):
        self.test_image_det1 = edge_extractor.read_det_results(
            os.path.join(test_image_det_dir, "labels", "test_1.txt"))
        self.test_image_det2 = edge_extractor.read_det_results(
            os.path.join(test_image_det_dir, "labels", "test_2.txt"))
        self.test_image_det3 = edge_extractor.read_det_results(
            os.path.join(test_image_det_dir, "labels", "test_3.txt"))
        self.assertIsNotNone(self.test_image_det1)
        self.assertIsNotNone(self.test_image_det2)
        self.assertIsNotNone(self.test_image_det3)

    def testReadSegResults(self):
        self.test_image_seg1 = edge_extractor.read_seg_results(
            os.path.join(test_image_seg_dir, "labels", "test_1.txt"))
        self.test_image_seg2 = edge_extractor.read_seg_results(
            os.path.join(test_image_seg_dir, "labels", "test_2.txt"))
        self.test_image_seg3 = edge_extractor.read_seg_results(
            os.path.join(test_image_seg_dir, "labels", "test_3.txt"))
        self.assertIsNotNone(self.test_image_seg1)
        self.assertIsNotNone(self.test_image_seg2)
        self.assertIsNotNone(self.test_image_seg3)

    def testOverlayMask(self):
        self.test_image_seg1 = edge_extractor.read_seg_results(
            os.path.join(test_image_seg_dir, "labels", "test_1.txt"))
        seg_mask = self.test_image_seg1
        w, h = seg_mask.shape
        img_bhs = np.zeros((w, h))
        img_bhs_mask = edge_extractor.overlay_mask(img_bhs, seg_mask)
        self.assertIsNotNone(img_bhs_mask)

    def testOverlayBbox(self):
        self.test_image1 = Image.open(os.path.join(test_image_dir, "test_1.jpg"))
        w, h, c = np.asarray(self.test_image1).shape
        self.test_image_det1 = edge_extractor.read_det_results(
            os.path.join(test_image_det_dir, "labels", "test_1.txt"))
        det_bbox = self.test_image_det1
        img_bhs = np.zeros((w, h))
        bbox = det_bbox[0]
        img_bhs = edge_extractor.overlay_box(img_bhs, bbox[1:])
        self.assertIsNotNone(img_bhs)

    def testMain(self):
        edge_extractor.main(self.args)


if __name__ == "__main__":
    unittest.main()
