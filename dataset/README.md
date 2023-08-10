# LF Dataset Releases
## Overview

| 数据集 | COCO原有数据筛选 | 全新图像标注数 | 全景图像标注数 | 总样本数 | 总标签数 |
| :--:  | :--:           | :----:       | :--: | :--:   | :--:    |
| 检测数据集 |4074 |2258 |200|6532 |40526 |
| 分割数据集 |4251 |2222 |410 |6883 | |


性能指标

| 模型 | mAP  | Precision | Recall |  mIoU  |  mAcc  |
| :--: | :--: | :----: | :--: | :--: | :--: |
| 检测(YOLOv5s-轻) | 0.622 | 0.769 | 0.568 | ~ | ~ |
| 检测(YOLOv5x-重) | 0.68 | 0.793 | 0.639 | ~ | ~ |
| 分割(SegFormer-轻) | ~ | ~ | ~ | 0.556 | 0.655 |
| 分割(UpernetVit-重) | ~ | ~ | ~ | 0.652 | 0.753 |

## 1. Dataset 

### 1.1 检测数据集	

&#8195;&#8195;初始数据集包含来自COCO数据集（4073张），手动补充标注的平面图片（2258张），手动补充标注的全景图片（200张），共计**6532**张。其中来自COCO的图片，我们在其原本的图像检测标注的基础上，映射并融合了我们现实草坪场景所需要的类别，统一转换为COCO的标准检测标注格式，之后再遍历所有图片，针对错标、漏标现象进行补充完善。手动补充的图片来自互联网图片和真实场景拍摄的平面图片和全景图片，采用CVAT图像检测标注工具完成标注，并导出为COCO的标准检测标注格式。

        检测数据集位置：
        >加州服务器/share/Share/Datasets/UESTC_COCO_Dataset/detection_v0.05
        >百度网盘链接：https://pan.baidu.com/s/1Y8mBLXDSK3P9nz1Z0fh26g 提取码：uet5
        
        
### 1.2 分割数据集

&#8195;&#8195;初始数据集包含来自COCO数据集（4251张），手动补充标注的平面图片（2222张） ，手动补充标注的全景图片（410张） ，共计**6883**张。其中来自COCO的图片，我们在COCO-Stuff分割标注的基础上，映射并融合了我们现实草坪场景所需要的类别，统一转换为语义分割的灰度图标注。手动补充的图片来自互联网图片和真实场景拍摄的平面图片和全景图片，采用EISeg图像分割标注工具完成标注，并导出为灰度图标注格式。


        分割数据集位置：
        >加州服务器/share/Share/Datasets/UESTC_COCO_Dataset/segmentation_v0.02
        >百度网盘链接：https://pan.baidu.com/s/1NR9uKnoBKFWGCHB0K6MZhQ 提取码：pymd

## 2. Detection

        检测模型位置：
        加州服务器/share/Share/Model_Release/det_yolov5x_20220717_v0.01.tar
        百度网盘链接：https://pan.baidu.com/s/1ue-c-fmKF6lawG50vbqD1g 提取码：blpb
        
### 2.1 数据集

将检测数据集划分为训练数据集以及验证数据集，划分比例约为9:1，详细数量统计如下：

- 训练集数量：6251张

- 验证集数量：695张

### 2.2 训练模型

#### 2.2.1 轻量模型：

> 用于实际部署以及使用

1. ###### CenterNet/MobileNet v2

网络结构示意图

<img src="https://s2.loli.net/2022/07/20/W7yVebQz1k8idsR.jpg" alt="Centernet_construction.jpg" style="zoom: 50%;" />

模型配置（基于COCO2017）：

| Backbone      | AP/fps     | Flip Ap/FPS | Multi-scale AP / FPS |
| ----------- | -------- | --------- | ------------------ |
| Hourglass-104 | 40.3/14  | 42.2/7.8  | 45.1/1.4           |
| DLA-34        | 37.4/52  | 39.2/28   | 41.7/4             |
| ResNet-101    | 34.6/45  | 36.2/25   | 39.3/4             |
| ResNet-18     | 28.1/142 | 30.0/71   | 33.2/12            |


训练指标

| Backbone     | mAP   | FPS  | Recall |
| ------------ | ----- | ---- | ------ |
| MobileNet v2 | 0.401 | 100  | 0.358  |

2. ###### YOLOv5/YOLOv5s

网络结构图：
<img src="https://s2.loli.net/2022/07/19/tSa3VewmdfH8J7K.jpg" alt="yolo_structure.jpg" style="zoom: 50%;" />

模型配置（基于COCO2017）：


|  Model  | Size(pixels) | mAP@0.5:0.95 | mAP@0.5 | Speed V100 b32(ms) | Params(M) | FLOPs(b) |
| :-----: | :----------: | :----------: | :-----: | :----------------: | :-------: | :------: |
| YOLOv5n |     640      |     28.0     |  45.7   |        0.6         |    1.9    |   4.5    |
| YOLOv5s |     640      |     37.4     |  56.8   |        0.9         |    7.2    |   16.5   |
| YOLOv5m |     640      |     45.4     |  64.1   |        1.7         |   21.2    |    49    |
| YOLOv5l |     640      |      49      |  67.3   |        2.7         |   46.5    |  109.1   |
| YOLOv5x |     640      |     50.7     |  68.9   |        4.8         |   86.7    |  205.7   |


训练指标

|  Model  | Precision |  mAP  | Recall |  BPR  |
| :-----: | :-------: | :---: | :----: | :---: |
| YOLOv5s |   0.679   | 0.572 | 0.541  | 0.994 |

<img src="https://s2.loli.net/2022/07/20/YZRSrmEvKlsnway.png" alt="yolov5s_0.05.png" style="zoom:67%;" />

#### 2.2.2 重型模型：

> 用于数据与模型迭代

###### YOLOv5/YOLOv5x

网络结构图：
>同 2.2.1 第 2部分

模型配置：
>同 2.2.1 第 2部分

训练指标

|Model |Precision |mAP|Recall|BPR |
| :----: | :----: | :---: | :--: | :---: |
| YOLOv5x | 0.728   | 0.658 | 0.616  | 0.994 |

<img src="https://s2.loli.net/2022/07/20/qVEu259lFD8J6ZI.png" alt="yolov5_0.05.png" style="zoom: 67%;" />

### 2.3 检测实例效果展示

<img src="https://s2.loli.net/2022/07/19/oIPqtHjA79G5J3c.jpg" height="150px"> <img src="https://s2.loli.net/2022/07/19/yS1KhL7MtBGl4uo.jpg" height="150px"> <img src="https://s2.loli.net/2022/07/19/EyjMAYk1CV8Iwuc.jpg" height="150px">


## 3. Segmentation
        分割模型位置：
        加州服务器/share/Share/Model_Release/seg_sefFormer_20220717_v0.01.tar
        百度网盘链接：https://pan.baidu.com/s/1FmDUklE08opdsnX5TzcBfQ 提取码：tnyr
        
### 3.1 数据集

训练集大小：6581

测试集大小：700张

### 3.2 训练模型
#### 3.2.1 轻量模型

###### BiSeNet/bisenetv2

网络结构图：

<img src="https://s2.loli.net/2022/07/20/CDmVnLr7aE1PY2v.png" alt="bisenet_construction.png" style="zoom: 50%;" />

模型配置：

- 基于Cityscape：

|           | mIoU-ss | mIoU-ssc | mIoU-msf | mIoU-mscf | fps(fp16/fp32) |
| :-------: | :-----: | :------: | :------: | :-------: | :------------: |
| bisenetv1 |  75.44  |  76.94   |  77.45   |   78.86   |     68/23      |
| bisenetv2 |  74.95  |  75.58   |  76.53   |   77.08   |     59/21      |

- 基于COCO-Stuff

|           | mIoU-ss | mIoU-ssc | mIoU-msf | mIoU-mscf |
| :-------: | :-----: | :------: | :------: | :-------: |
| bisenetv1 |  31.49  |  31.42   |  32.46   |   32.55   |
| bisenetv2 |  30.49  |  30.55   |  31.81   |   31.73   |

> **ss** means single scale evaluation,
> **ssc** means single scale crop evaluation,
> **msf** means multi-scale evaluation with flip augment
>  **mscf** means multi-scale crop evaluation with flip evaluation

相关指标：

| aAcc | mIoU  | mAcc   |
| :-: | :--: | :---: |
| 0.86 | 0.605 | 0.7288 |

![bisenet_0.02.png](https://s2.loli.net/2022/07/20/bJZVELANuf9jzlh.png)

#### 3.2.2 重型模型

###### Segformer/MiT-B5

网络结构图：
<img src="https://s2.loli.net/2022/07/19/v8RSGXrP2FxkKI1.png" alt="SegFormer_structon.png" style="zoom: 50%;" />

模型配置：

<table style="width: 100%; height: 100%; margin-left: auto; margin-right: auto;">
<tbody>
<tr style="height: 14px;">
<td style="width: 86.6667px; height: 45px; text-align: center;" rowspan="2">&nbsp;&nbsp;Enconder Model Size</td>
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
<td style="width: 62.6667px; height: 22px;">44.0</td>
<td style="width: 62.6667px; height: 22px;">3.3</td>
<td style="width: 50.6667px; height: 22px;">79.0</td>
<td style="width: 96px; height: 22px;">49.4/50.0</td>
<td style="width: 56.6667px; height: 22px;">962.9</td>
<td style="width: 98px; height: 22px;">81.7/83.3</td>
<td style="width: 50.6667px; height: 22px;">79.0</td>
<td style="width: 74px; height: 22px;">45.5</td>
</tr>
<tr style="height: 22px;">
<td style="width: 86.6667px; height: 22px;">MiT-B4</td>
<td style="width: 62.6667px; height: 22px;">60.8</td>
<td style="width: 62.6667px; height: 22px;">3.3</td>
<td style="width: 50.6667px; height: 22px;">95.7</td>
<td style="width: 96px; height: 22px;">50.3/51.1</td>
<td style="width: 56.6667px; height: 22px;">1240.6</td>
<td style="width: 98px; height: 22px;">82.3/83.9</td>
<td style="width: 50.6667px; height: 22px;">95.7</td>
<td style="width: 74px; height: 22px;">46.5</td>
</tr>
<tr style="height: 22px;">
<td style="width: 86.6667px; height: 22px;">MiT-B5</td>
<td style="width: 62.6667px; height: 22px;">81</td>
<td style="width: 62.6667px; height: 22px;">3.3</td>
<td style="width: 50.6667px; height: 22px;">183.3</td>
<td style="width: 96px; height: 22px;">51/51.8</td>
<td style="width: 56.6667px; height: 22px;">1460.4</td>
<td style="width: 98px; height: 22px;">82.4/84.0</td>
<td style="width: 50.6667px; height: 22px;">111.6</td>
<td style="width: 74px; height: 22px;">46.7</td>
</tr>
</tbody>
</table>
相关指标

|  aAcc   |  mIoU  |  mAcc |
| :---: | :--: | :-: |
|  0.9126 |  0.743 | 0.85 |

<img src="https://s2.loli.net/2022/07/20/91ncVLJoavCTY8i.png" style="zoom:80%">


### 3.3 分割的实例效果展示

<img src="https://s2.loli.net/2022/07/19/W3Oa71C5nkHDPrJ.jpg" height="190px"> <img src="https://s2.loli.net/2022/07/19/BiXG7J6UZrwdzIy.jpg" height="190px"> <img src="https://s2.loli.net/2022/07/19/lJCm4c1sMtDEHhg.jpg" height="190px">



## 4. Unprocessed Data

| 时间       | 来源          | 类型     | 大小    | 处理程度         | 样本 | 已处理 |
| ---------- | ------------- | -------- | ------- | ---------------- | ---- | ------ |
| 2022/05/25 | CN_Huzhou     | 全景视频 | 202 MB  | 展开、抽帧       | -    | -      |
| 2022/06/20 | US_California | 平面视频 | 3.71 GB | 压缩、抽帧、标注 | 1933 | 629    |
| 2022/06/26 | US_Colorado   | 平面视频 | 441 MB  | 压缩、抽帧、标注 | 119  | 27     |
| 2022/07/08 | CN_Huzhou     | 平面视频 | 340 MB  | 压缩、抽帧、标注 | 836  | 652    |
| 2022/07/08 | CN_Huzhou     | 平面图像 | 6.5 MB  | 压缩、标注       | 18   | 18     |
| 2022/07/13 | US_Michigan   | 平面视频 | 3.57 GB | -                | -    | -      |
| 2022/07/16 | CN_Huzhou     | 全景视频 | 466 MB  | 展开、抽帧       | -    | -      |
| 2022/07/16 | CN_Huzhou     | 平面视频 | 37 MB   | 压缩             | -    | -      |

>*完成标注后但由于场景匹配度不足或质量不符合要求而丢弃的图片数量：**1353**张

