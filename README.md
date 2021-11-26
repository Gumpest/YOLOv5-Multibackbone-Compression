# YOLOv5-Compression

![](https://img.shields.io/badge/Update-News-blue.svg?style=plastic)

2021.10.30 复现TPH-YOLOv5

2021.10.31 完成替换backbone为Ghostnet

2021.11.02 完成替换backbone为Shufflenetv2

2021.11.05 完成替换backbone为Mobilenetv3Small

2021.11.10 完成EagleEye对YOLOv5系列剪枝支持

2021.11.14 完成MQBench对YOLOv5系列量化支持

2021.11.16 完成替换backbone为EfficientNetLite-0

2021.11.26 完成替换backbone为PP-LCNet-1x

## Requirements

环境安装

```shell
pip install -r requirements.txt
```


## Evaluation metric

Visdrone DataSet (1-5 size is 608，6-8 size is 640)

| Model          | mAP   | mAP@50 | Parameters(M) | GFLOPs |
| -------------- | ----- | ------ | ------------- | ------ |
| YOLOv5n        | 13    | 26.2   | 1.78          | 4.2    |
| YOLOv5s        | 18.4  | 34     | 7.05          | 15.9   |
| YOLOv5m        | 21.6  | 37.8   | 20.91         | 48.2   |
| YOLOv5l        | 23.2  | 39.7   | 46.19         | 108.1  |
| YOLOv5x        | 24.3  | 40.8   | 86.28         | 204.4  |
| YOLOv5xP2      | 30.00 | 49.29  | 90.96         | 314.2  |
| YOLOv5xP2 CBAM | 30.13 | 49.40  | 91.31         | 315.1  |
| YOLOv5x-TPH    | 30.26 | 49.39  | 86.08         | 238.9  |

训练脚本实例：

```shell
nohup python train.py --data VisDrone.yaml --weights yolov5n.pt --cfg models/yolov5n.yaml --epochs 300 --batch-size 8 --img 608 --nosave --device 0,1 --sync-bn >> yolov5n.txt &
```
## Multi-Backbone

### 精度区

#### TPH-YOLOv5 ![](https://img.shields.io/badge/Model-BeiHangUni-yellowgreen.svg?style=plastic)

消融实验如下：

| Model                 | mAP   | mAP@50 | Parameters(M) | GFLOPs |
| --------------------- | ----- | ------ | ------------- | ------ |
| YOLOv5x(604)          | 24.3  | 40.8   | 86.28         | 204.4  |
| YOLOv5xP2(640)        | 30.1  | 49.3   | 90.96         | 314.2  |
| YOLOv5xP2 CBAM(640)   | 30.13 | 49.40  | 91.31         | 315.1  |
| **YOLOv5x-TPH**(640)  | 30.26 | 49.39  | 96.76         | 345.0  |
| **YOLOv5x-TPH(1536)** | 38.00 | 60.56  | 94.34         | 268.1  |

组件：P2检测头、CBAM、Transformer Block

结构图如下：

![TPH-YOLOv5](https://github.com/Gumpest/YOLOv5-Multibackbone-Compression/blob/main/img/TPH-YOLOv5.png)

注意：

1、避免TransBlock导致显存爆炸，MultiAttentionHead中将注意力头的数量减少至4，并且FFN中的两个全连接层从linear(c1, 4*c1)改为linear(c1, c1)，去掉GELU函数

2、TransBlock的数量会根据YOLO规模的不同而改变，标准结构作用于YOLOv5m

3、当YOLOv5x为主体与标准结构的区别是：（1）首先去掉14和19的CBAM模块（2）降低与P2关联的通道数（128）（3）在输出头之前会添加SPP模块，注意SPP的kernel随着P的像素减小而减小（4）在CBAM之后进行输出（5）只保留backbone以及最后一层输出的TransBlock（6）采用BiFPN作为neck

4、更改不同Loss分支的权重：如下图，当训练集的分类与置信度损失还在下降时，验证集的分类与置信度损失开始反弹，说明出现了过拟合，需要降低这两个任务的权重

消融实验如下：

| box  | cls  | obj  | acc       |
| ---- | ---- | ---- | --------- |
| 0.05 | 0.5  | 1.0  | 37.90     |
| 0.05 | 0.3  | 0.7  | **38.00** |
| 0.05 | 0.2  | 0.4  | 37.5      |

![Loss](https://github.com/Gumpest/YOLOv5-Multibackbone-Compression/blob/main/img/image-20211109150606998.png)

### 轻量区

| Model                     | mAP       | mAP@50 | Parameters(M) | GFLOPs   | FPS@CPU | TrainCost(h) | Memory Cost(G) |
| ------------------------- | --------- | ------ | ------------- | -------- | ------- | ------------ | -------------- |
| YOLOv5s                   | 18.4      | 34     | 7.05          | 15.9     |         | 17.38        | 1.46           |
| YOLOv5l-Ghostnet          | 18.4      | 33.8   | 24.27         | 42.4     |         | 27.44        | 4.97           |
| YOLOv5l-ShufflenetV2      | 16.48     | 31.1   | 21.27         | 40.5     |         | 10.98        | 2.41           |
| YOLOv5l-Mobilenetv3Small  | 16.55     | 31.2   | **20.38**     | **38.4** |         | **10.19**    | 5.30           |
| YOLOv5l-EfficientNetLite0 | **19.12** | **35** | 23.01         | 43.9     |         | 13.94        | 2.04           |
| YOLOv5l-PP-LCNet          | 17.63     | 32.8   | 21.64         | 41.7     |         | 18.52        | **1.66**       |

#### GhostNet-YOLOv5 ![](https://img.shields.io/badge/Model-HuaWei-orange.svg?style=plastic)

![GhostNet](https://github.com/Gumpest/YOLOv5-Multibackbone-Compression/blob/main/img/GhostNet.jpg)

（1）为保持一致性，下采样的DW的kernel_size均等于3

（2）neck部分与head部分沿用YOLOv5l原结构

（3）中间通道人为设定（expand）

#### ShuffleNet-YOLOv5 ![](https://img.shields.io/badge/Model-Megvii-orange.svg?style=plastic)

![Shffulenet](https://github.com/Gumpest/YOLOv5-Multibackbone-Compression/blob/main/img/Shffulenet.png)

（1）Focus Layer不利于芯片部署，频繁的slice操作会让缓存占用严重

（2）避免多次使用C3 Leyer以及高通道的C3 Layer（违背G1与G3准则）

（3）中间通道不变

#### MobileNetv3Small-YOLOv5 ![](https://img.shields.io/badge/Model-Google-orange.svg?style=plastic)

![Mobilenetv3s](https://github.com/Gumpest/YOLOv5-Multibackbone-Compression/blob/main/img/Mobilenetv3s.jpg)

（1）原文结构，部分使用Hard-Swish激活函数以及SE模块

（2）Neck与head部分嫁接YOLOv5l原结构

（3）中间通道人为设定（expand）

#### EfficientNetLite0-YOLOv5 ![](https://img.shields.io/badge/Model-Google-orange.svg?style=plastic)

![efficientlite](https://github.com/Gumpest/YOLOv5-Multibackbone-Compression/blob/main/img/efficientlite.jpg)

（1）使用Lite0结构，且不使用SE模块

（2）针对dropout_connect_rate，手动赋值(随着idx_stage变大而变大)

（3）中间通道一律*6（expand）

#### PP-LCNet-YOLOv5  ![](https://img.shields.io/badge/Model-Baidu-orange.svg?style=plastic)

![PP-LCNet](https://github.com/Gumpest/YOLOv5-Multibackbone-Compression/blob/main/img/PP-LCNet.png)

（1）使用PP-LCNet-1x结构，在网络末端使用SE以及5*5卷积核

（2）SeBlock压缩维度为原1/16

（3）中间通道不变

## MQBench Quantize Aware Training

 MQBench是实际硬件部署下评估量化算法的框架，进行各种适合于硬件部署的量化训练（QAT）
### Requirements
- PyTorch == 1.8.1
### Install MQBench Lib ![](https://img.shields.io/badge/Tec-Sensetime-brightgreen.svg?style=plastic)
由于MQBench目前还在不断更新，选择0.0.2稳定版本作为本仓库的量化库。
```shell
git clone https://github.com/ZLkanyo009/MQBench.git
cd MQBench
python setup.py build
python setup.py install
```
### Usage
训练脚本实例：

```shell
python train.py --data VisDrone.yaml --weights yolov5n.pt --cfg models/yolov5n.yaml --epochs 300 --batch-size 8 --img 608 --nosave --device 0,1 --sync-bn --quantize --BackendType NNIE
```
## Pruning

基于YOLOv5的块状结构设计，该仓库采用基于搜索的通道剪枝方法[EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning](https://arxiv.org/abs/2007.02491)。核心思想是随机搜索到大量符合目标约束的子网，然后快速更新校准BN层的均值与方差参数，并在验证集上测试校准后全部子网的精度。精度最高的子网拥有最好的架构，经微调恢复后能达到较高的精度。

![eagleeye](https://github.com/Cydia2018/YOLOv5-Multibackbone-Compression/blob/main/img/eagleeye.png)

这里为Conv、C3、SPP和SPPF模块设计了通道缩放比例用于搜索。具体来说有以下缩放系数：

- Conv模块的输出通道数
- C3模块中cv2块和cv3块的输出通道数
- C3模块中若干个bottleneck中的cv1块的输出通道数

### Usage

1. 正常训练模型

```shell
python train.py --data data/VisDrone.yaml --imgsz 640 --weights yolov5s.pt --cfg models/prunModels/yolov5s-pruning.yaml --device 0
```

（注意训练其他版本的模型，参考models/yolov5s-visdrone.yaml进行修改。目前只支持原版v5架构）

2. 搜索最优子网

```shell
python prune_eagleeye.py --weights path_to_trained_yolov5_model --cfg models/prunModels/yolov5s-pruning.yaml --data data/VisDrone.yaml --path path_to_pruned_yolov5_yaml --max_iter maximum number of arch search --remain_ratio the whole FLOPs remain ratio
```

3. 微调恢复精度

```shell
python train.py --data data/VisDrone.yaml --imgsz 640 --weights path_to_pruned_yolov5_model --cfg path_to_pruned_yolov5_yaml --device 0
```

## To do

- [x] Multibackbone: MobilenetV3-small
- [x] Multibackbone: ShufflenetV2
- [x] Multibackbone: Ghostnet
- [x] Multibackbone: EfficientNet-Lite0
- [x] Multibackbone: PP-LCNet
- [x] Multibackbone: TPH-YOLOv5
- [ ] Multibackbone: Swin-YOLOv5
- [ ] Pruner: Network slimming
- [x] Pruner: EagleEye
- [ ] Pruner: OneShot (L1, L2, FPGM), ADMM, NetAdapt, Gradual, End2End
- [x] Quantization: MQBench
- [ ] Knowledge Distillation
