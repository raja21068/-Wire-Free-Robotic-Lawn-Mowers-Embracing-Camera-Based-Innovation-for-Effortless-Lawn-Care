import os
import unittest
import numpy as np

import io_helper

test_image_dir = os.path.join(os.getcwd(), '../examples/test/test/')
test_image_det_dir = os.path.join(os.path.pardir, 'examples/test/test_det/')
test_image_seg_dir = os.path.join(os.path.pardir, 'examples/test/test_seg/')


class ModelHelperTest(unittest.TestCase):

    def setUp(self):
        print()


if __name__ == "__main__":
    unittest.main()
