# APIs to extract the soft and/or hard edges

核心API位于 edge_extractor.parse()，目前支持基于分割与检测结果提取边界：

```python
def parse(image, seg_mask=None, bbox_list=None, img_depth=None):
```  

## 1. edge_extractor.py 源代码 

支持的检测与分割结果格式，参见：/examples/test/test_det/labels/ 和 /examples/test/test_seg/labels/。

其中，检测结果采用YOLO格式，即 bbox[0] 和 bbox[1] 为框中心坐标。

如果采用了不同类型的检测结果输出（YOLO与COCO格式不同），需要对 bbox 作相应的转换，即中心点坐标与左上角坐标转换。

此外，main() 函数输入参数为：

```python
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', type=str, default="segmentation",
                        # choices=OPT_TYPES,  # ["segmentation", "seg", "detection", "det", "both"]
                        help="Base operation(s), segmentation, detection or both, for edge extraction.")
    parser.add_argument('image_path', type=str, help="Path of the target image.")
    parser.add_argument('--output_path', type=str, default='', help="Path to save the results.")
    parser.add_argument('--mask_path', type=str, help="Path of the segmentation mask.")
    parser.add_argument('--bbox_path', type=str, help="Path of the bounding boxes by detection.")
    parser.add_argument('--display', action='store_false', help="True or False option for display")
    args = parser.parse_args()
    return args
```  

用法示例： 

```shell
$ cd api
$ python edge_extractor.py both \
    ../examples/test/test/test_1.jpg \
    --output_path ../examples/test/output/ \
    --mask_path ../examples/test/test_seg/labels/test_1.txt \
    --bbox_path ../examples/test/test_det/labels/test_1.txt
```  

## 2. edge_extractor_test.py 单元测试代码（及 main() 调用方法）

可以 setUp() 中切换测试配置，执行不同的测试用例：

```python
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
```  

## 3. 三个不同位置草坪图像的测试结果

- 只使用分割结果：[test_1.jpg](/examples/test/test/test_1.jpg)

![test_1_all.jpg](/examples/test/output/test_1_all.png "test_1_all.png")

- 同时使用分割与检测结果：[test_2.jpg](/examples/test/test/test_2.jpg)

![test_2_all.jpg](/examples/test/output/test_2_all.png "test_2_all.png")

- 只使用检测结果：[test_3.jpg](/examples/test/test/test_3.jpg)

![test_3_all.jpg](/examples/test/output/test_3_all.png "test_3_all.png")


