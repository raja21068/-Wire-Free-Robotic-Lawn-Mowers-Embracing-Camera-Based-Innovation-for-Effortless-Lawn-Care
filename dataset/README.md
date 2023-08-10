Sure, here's the translation of your text from Chinese to English:

---

# LF Dataset Releases
## Overview

| Dataset Type | Original COCO Selection | New Image Annotations | Panoramic Image Annotations | Total Samples | Total Labels |
| :--:  | :--:           | :----:       | :--: | :--:   | :--:    |
| Detection Dataset | 4074 | 2258 | 200 | 6532 | 40526 |
| Segmentation Dataset | 4251 | 2222 | 410 | 6883 | - |


Performance Metrics

| Model | mAP  | Precision | Recall |  mIoU  |  mAcc  |
| :--: | :--: | :----: | :--: | :--: | :--: |
| Detection (YOLOv5s-Light) | 0.622 | 0.769 | 0.568 | ~ | ~ |
| Detection (YOLOv5x-Heavy) | 0.68 | 0.793 | 0.639 | ~ | ~ |
| Segmentation (SegFormer-Light) | ~ | ~ | ~ | 0.556 | 0.655 |
| Segmentation (UpernetVit-Heavy) | ~ | ~ | ~ | 0.652 | 0.753 |

## 1. Dataset 

### 1.1 Detection Dataset	

&#8195;&#8195;The initial dataset includes 4073 images from the COCO dataset, 2258 manually annotated flat images, and 200 manually annotated panoramic images, totaling **6532** images. For images from COCO, we mapped and merged the categories needed for our real lawn scenes onto the original image detection annotations, transforming them into COCO's standard detection annotation format. We then reviewed all images, addressing misannotations and omissions. The manually added images are sourced from the internet and real-world scenes, annotated using the CVAT image detection annotation tool, and exported in COCO's standard detection annotation format.

        Detection dataset location:
        > California server/share/Share/Datasets/UESTC_COCO_Dataset/detection_v0.05
        > Baidu Pan link: https://pan.baidu.com/s/1Y8mBLXDSK3P9nz1Z0fh26g Access code: uet5

### 1.2 Segmentation Dataset

&#8195;&#8195;The initial dataset includes 4251 images from the COCO dataset, 2222 manually annotated flat images, and 410 manually annotated panoramic images, totaling **6883** images. For images from COCO, we mapped and merged the categories needed for our real lawn scenes, converting them into grayscale semantic segmentation annotations based on COCO-Stuff segmentation annotations. The manually added images are sourced from the internet and real-world scenes, annotated using the EISeg image segmentation annotation tool, and exported as grayscale annotations.

        Segmentation dataset location:
        > California server/share/Share/Datasets/UESTC_COCO_Dataset/segmentation_v0.02
        > Baidu Pan link: https://pan.baidu.com/s/1NR9uKnoBKFWGCHB0K6MZhQ Access code: pymd

## 2. Detection

        Detection model location:
        California server/share/Share/Model_Release/det_yolov5x_20220717_v0.01.tar
        Baidu Pan link: https://pan.baidu.com/s/1ue-c-fmKF6lawG50vbqD1g Access code: blpb

### 2.1 Dataset

The detection dataset is divided into training and validation sets with a ratio of approximately 9:1. Detailed quantity statistics are as follows:

- Training set: 6251 images
- Validation set: 695 images

### 2.2 Training Models

#### 2.2.1 Lightweight Models:

> For practical deployment and usage

1. ###### CenterNet/MobileNet v2

Network structure diagram

<img src="https://s2.loli.net/2022/07/20/W7yVebQz1k8idsR.jpg" alt="Centernet_construction.jpg" style="zoom: 50%;" />

Model configuration (based on COCO2017):

| Backbone      | AP/fps     | Flip Ap/FPS | Multi-scale AP / FPS |
| ----------- | -------- | --------- | ------------------ |
| Hourglass-104 | 40.3/14  | 42.2/7.8  | 45.1/1.4           |
| DLA-34        | 37.4/52  | 39.2/28   | 41.7/4             |
| ResNet-101    | 34.6/45  | 36.2/25   | 39.3/4             |
| ResNet-18     | 28.1/142 | 30.0/71   | 33.2/12            |

Training metrics

| Backbone     | mAP   | FPS  | Recall |
| ------------ | ----- | ---- | ------ |
| MobileNet v2 | 0.401 | 100  | 0.358  |

2. ###### YOLOv5/YOLOv5s

Network structure diagram:
<img src="https://s2.loli.net/2022/07/19/tSa3VewmdfH8J7K

.jpg" alt="yolo_structure.jpg" style="zoom: 50%;" />

Model configuration (based on COCO2017):

|  Model  | Size(pixels) | mAP@0.5:0.95 | mAP@0.5 | Speed V100 b32(ms) | Params(M) | FLOPs(b) |
| :-----: | :----------: | :----------: | :-----: | :----------------: | :-------: | :------: |
| YOLOv5n |     640      |     28.0     |  45.7   |        0.6         |    1.9    |   4.5    |
| YOLOv5s |     640      |     37.4     |  56.8   |        0.9         |    7.2    |   16.5   |
| YOLOv5m |     640      |     45.4     |  64.1   |        1.7         |   21.2    |    49    |
| YOLOv5l |     640      |      49      |  67.3   |        2.7         |   46.5    |  109.1   |
| YOLOv5x |     640      |     50.7     |  68.9   |        4.8         |   86.7    |  205.7   |

Training metrics

|  Model  | Precision |  mAP  | Recall |  BPR  |
| :-----: | :-------: | :---: | :----: | :---: |
| YOLOv5s |   0.679   | 0.572 | 0.541  | 0.994 |

<img src="https://s2.loli.net/2022/07/20/YZRSrmEvKlsnway.png" alt="yolov5s_0.05.png" style="zoom:67%;" />

#### 2.2.2 Heavy Models:

> For data and model iteration

###### YOLOv5/YOLOv5x

Network structure diagram:
> Same as 2.2.1 Part 2

Model configuration:
> Same as 2.2.1 Part 2

Training metrics

|Model |Precision |mAP|Recall|BPR |
| :----: | :----: | :---: | :--: | :---: |
| YOLOv5x | 0.728   | 0.658 | 0.616  | 0.994 |

<img src="https://s2.loli.net/2022/07/20/qVEu259lFD8J6ZI.png" alt="yolov5_0.05.png" style="zoom: 67%;" />

### 2.3 Detection Instance Results

<img src="https://s2.loli.net/2022/07/19/oIPqtHjA79G5J3c.jpg" height="150px"> <img src="https://s2.loli.net/2022/07/19/yS1KhL7MtBGl4uo.jpg" height="150px"> <img src="https://s2.loli.net/2022/07/19/EyjMAYk1CV8Iwuc.jpg" height="150px">


## 3. Segmentation
        Segmentation model location:
        California server/share/Share/Model_Release/seg_sefFormer_20220717_v0.01.tar
        Baidu Pan link: https://pan.baidu.com/s/1FmDUklE08opdsnX5TzcBfQ Access code: tnyr
        
### 3.1 Dataset

Training set size: 6581

Test set size: 700 images

### 3.2 Training Models
#### 3.2.1 Lightweight Models

###### BiSeNet/BiSeNetv2

Network structure diagram:

<img src="https://s2.loli.net/2022/07/20/CDmVnLr7aE1PY2v.png" alt="bisenet_construction.png" style="zoom: 50%;" />

Model configuration:

- Based on Cityscape:

|           | mIoU-ss | mIoU-ssc | mIoU-msf | mIoU-mscf | fps(fp16/fp32) |
| :-------: | :-----: | :------: | :------: | :-------: | :------------: |
| bisenetv1 |  75.44  |  76.94   |  77.45   |   78.86   |     68/23      |
| bisenetv2 |  74.95  |  75.58   |  76.53   |   77.08   |     59/21      |

- Based on COCO-St

uff

|           | mIoU-ss | mIoU-ssc | mIoU-msf | mIoU-mscf |
| :-------: | :-----: | :------: | :------: | :-------: |
| bisenetv1 |  31.49  |  31.42   |  32.46   |   32.55   |
| bisenetv2 |  30.49  |  30.55   |  31.81   |   31.73   |

> **ss** means single-scale evaluation,
> **ssc** means single-scale crop evaluation,
> **msf** means multi-scale evaluation with flip augmentation,
> **mscf** means multi-scale crop evaluation with flip evaluation.

Relevant metrics:

| aAcc | mIoU  | mAcc   |
| :-: | :--: | :---: |
| 0.86 | 0.605 | 0.7288 |

![bisenet_0.02.png](https://s2.loli.net/2022/07/20/bJZVELANuf9jzlh.png)

#### 3.2.2 Heavy Models

###### SegFormer/MiT-B5

Network structure diagram:
<img src="https://s2.loli.net/2022/07/19/v8RSGXrP2FxkKI1.png" alt="SegFormer_structure.png" style="zoom: 50%;" />

Model configuration:

<table style="width: 100%; height: 100%; margin-left: auto; margin-right: auto;">
<tbody>
<tr style="height: 14px;">
<td style="width: 86.6667px; height: 45px; text-align: center;" rowspan="2">&nbsp;&nbsp;Encoder Model Size</td>
<td class="xl65" style="height: 14px; width: 125.333px; text-align: center;" colspan="2">Params</td>
<td class="xl65" style="height: 14px; width: 146.667px; text-align: center;" colspan="2">ADE20K</td>
<td class="xl65" style="height: 14px; width: 154.667px; text-align: center;" colspan="2">Cityscapes</td>
<td class="xl65" style="height: 14px; width: 124.667px; text-align: center;" colspan="2">COCO-Stuff</td>
</tr>
<tr style="height: 31px;">
<td class="xl65" style="height: 31px; width: 62.6667px;">Encoder</td>
<td class="xl65" style="width: 62.6667px; height: 31px;">Decoder</td>
<td class="xl65" style="width: 50.6667px; height: 31px;">Flops</td>
<td class="xl65" style="width: 96px; height: 31px;">mIoU(SS/MS)</td>
<td class="xl65" style="height: 31px; width: 56.6667px;">Flops</td>
<td class="xl65" style="width: 98px; height: 31px;">mIoU(SS/MS)</td>
<td class="xl65" style="width: 50.6667px; height: 31px;">Flops</td>
<td class="xl65" style="width: 74px; height: 31px;">mIoU(SS)</td>
</tr>
<tr style="height: 22.3333px;">
<td style="width: 86.6667px; height: 22.3333px;">MiT-B0</td>
<td style="width: 62.6667px; height: 22.3333px; ">3.4</td>
<td style="width: 62.6667px; height: 22.3333px;">0.4</td>
<td style="width: 50.6667px; height: 22.3333px;">8.4</td>
<td style="width: 96px; height: 22.3333px;">37.4/39.0</td>
<td style="width: 56.6667px; height: 22.3333px;">125.5</td>
<td style="width: 98px; height: 22.3333px;">76.2/78.1</td>
<td style="width: 50.6667px; height: 22.3333px; ">8.4</td>
<td style="width: 74px; height: 22.3333px;">35.6</td>
</tr>
<tr style="height: 22px;">
<td style="width: 86.6667px; height: 22px;">MiT-B1</td>
<td style="width: 62.6667px; height: 22px;">13.1</td>
<td style="width: 62.6667px; height: 22px;">0.6</td>
<td style="width: 50.6667px; height: 22px;">15.9</td>
<td style="width: 96px; height: 22px;">42.2/43.1</td>
<td style="width: 56.6667px; height: 22px;">243.7</td>
<td style="width: 98px; height: 22px;">78.5/80.0</td>
<td style="width: 50.6667px; height: 22px;">15.9</td>
<td style="width: 74px; height: 22px;">40.2</td>
</tr>
<tr style="height: 22px;">
<td style="width: 86.6667px; height: 22px;">MiT-B2</td>
<td style="width: 62.6667px; height: 22px;">24.2</td>
<td style="width: 62.6667px; height: 22px;">3.3</td>
<td style="width: 50.6667px; height: 22px;">62.4</td>
<td style="width: 96px; height: 22px;">46.5/47.5</td>
<td style="width: 56.6667px; height: 22px;">717.1</td>
<td style="width: 98px; height: 22px;">81.0/82.2</td>
<td style="width: 50.6667px; height: 22px;">62.4</td>
<td style="width: 74px; height: 22px;">44.6</td>
</tr>
<tr style="height: 22px;">
<td style="width: 86.6667px; height: 22px;">MiT-B3</td>
<td style="width: 62.6667px;

 height: 22px;">34.8</td>
<td style="width: 62.6667px; height: 22px;">4.5</td>
<td style="width: 50.6667px; height: 22px;">116.7</td>
<td style="width: 96px; height: 22px;">47.8/48.9</td>
<td style="width: 56.6667px; height: 22px;">1311.6</td>
<td style="width: 98px; height: 22px;">81.5/82.7</td>
<td style="width: 50.6667px; height: 22px;">116.7</td>
<td style="width: 74px; height: 22px;">47.4</td>
</tr>
<tr style="height: 22px;">
<td style="width: 86.6667px; height: 22px;">MiT-B4</td>
<td style="width: 62.6667px; height: 22px;">60.7</td>
<td style="width: 62.6667px; height: 22px;">8.8</td>
<td style="width: 50.6667px; height: 22px;">294.6</td>
<td style="width: 96px; height: 22px;">48.6/49.7</td>
<td style="width: 56.6667px; height: 22px;">2370.1</td>
<td style="width: 98px; height: 22px;">82.0/83.1</td>
<td style="width: 50.6667px; height: 22px;">294.6</td>
<td style="width: 74px; height: 22px;">49.0</td>
</tr>
</tbody>
</table>

Relevant metrics:

| Model | aAcc  | mIoU  | mIoU50 | mIoU75 |
| :----: | :---: | :---: | :----: | :----: |
| Mit-B0 | 87.8  | 0.596 | 0.817  | 0.625  |
| Mit-B1 | 89.7  | 0.614 | 0.835  | 0.647  |
| Mit-B2 | 90.7  | 0.625 | 0.848  | 0.662  |
| Mit-B3 | 91.2  | 0.631 | 0.854  | 0.669  |
| Mit-B4 | 91.6  | 0.636 | 0.859  | 0.674  |
| Mit-B5 | 92.0  | 0.642 | 0.864  | 0.679  |

![segFormer_0.02.png](https://s2.loli.net/2022/07/20/iwNDPYeZktbfSvF.png)

## 4. Future Plan

- Improve on detection models' performance
- Expand segmentation model application scenarios
- Continuously improve the dataset and provide updates
- Optimize model deployment for various scenarios

---

Please note that the translations provided here might not be perfect and might need some manual adjustment for accurate interpretation due to the technical nature of the content and potential differences in terminology between languages.
