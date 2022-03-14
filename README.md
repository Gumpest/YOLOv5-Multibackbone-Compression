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

2021.12.12 完成SwinTrans-YOLOv5（C3STR）

2021.12.15 完成Slimming对YOLOv5系列剪枝支持

## Requirements

```shell
pip install -r requirements.txt
```

## Multi-Backbone Substitution for YOLOs

### 1、Base Model

Train on Visdrone DataSet (*Input size is 608*)

| No.  | Model   | mAP  | mAP@50 | Parameters(M) | GFLOPs |
| ---- | ------- | ---- | ------ | ------------- | ------ |
| 1    | YOLOv5n | 13.0 | 26.20  | 1.78          | 4.2    |
| 2    | YOLOv5s | 18.4 | 34.00  | 7.05          | 15.9   |
| 3    | YOLOv5m | 21.6 | 37.80  | 20.91         | 48.2   |
| 4    | YOLOv5l | 23.2 | 39.70  | 46.19         | 108.1  |
| 5    | YOLOv5x | 24.3 | 40.80  | 86.28         | 204.4  |

### 2、Higher Precision Model

#### A、TPH-YOLOv5 ![](https://img.shields.io/badge/Model-BeiHangUni-yellowgreen.svg?style=plastic)

Train on Visdrone DataSet (*6-7 size is 640，8 size is 1536*)

| No.  | Model          | mAP  | mAP@50 | Parameters(M) | GFLOPs |
| ---- | -------------- | ---- | ------ | ------------- | ------ |
| 6    | YOLOv5xP2      | 30.0 | 49.29  | 90.96         | 314.2  |
| 7    | YOLOv5xP2 CBAM | 30.1 | 49.40  | 91.31         | 315.1  |
| 8    | YOLOv5x-TPH    | 40.7 | 63.00  | 112.97        | 270.8  |

###### Usage：

```shell
nohup python train.py --data VisDrone.yaml --weights yolov5n.pt --cfg models/yolov5n.yaml --epochs 300 --batch-size 8 --img 608 --device 0,1 --sync-bn >> yolov5n.txt &
```

###### Composition：

**P2 Head、CBAM、TPH、BiFPN、SPP**

<img src="https://github.com/Gumpest/YOLOv5-Multibackbone-Compression/blob/main/img/TPH-YOLOv5.png" alt="TPH-YOLOv5" width="600px" height="300px" />

1、TransBlock的数量会根据YOLO规模的不同而改变，标准结构作用于YOLOv5m

2、当YOLOv5x为主体与标准结构的区别是：（1）首先去掉14和19的CBAM模块（2）降低与P2关联的通道数（128）（3）在输出头之前会添加SPP模块，注意SPP的kernel随着P的像素减小而减小（4）在CBAM之后进行输出（5）只保留backbone以及最后一层输出的TransBlock（6）采用BiFPN作为neck

3、更改不同Loss分支的权重：如下图，当训练集的分类与置信度损失还在下降时，验证集的分类与置信度损失开始反弹，说明出现了过拟合，需要降低这两个任务的权重

消融实验如下：

| box  | cls  | obj  | acc       |
| ---- | ---- | ---- | --------- |
| 0.05 | 0.5  | 1.0  | 37.90     |
| 0.05 | 0.3  | 0.7  | **38.00** |
| 0.05 | 0.2  | 0.4  | 37.5      |

<img src="https://github.com/Gumpest/YOLOv5-Multibackbone-Compression/blob/main/img/image-20211109150606998.png" alt="loss" width="600px" />

#### B、SwinTrans-YOLOv5![](https://img.shields.io/badge/Model-Microsoft-yellow.svg?style=plastic)

```shell
pip install timm
```

###### Usage：

```shell
python train.py --data VisDrone.yaml --weights yolov5x.pt --cfg models/accModels/yolov5xP2CBAM-Swin-BiFPN-SPP.yaml --hyp data/hyps/hyp.visdrone.yaml --epochs 60 --batch-size 4 --img 1536 --nohalf
```

（1）Window size由***7***替换为检测任务常用分辨率的公约数***8***

（2）create_mask封装为函数，由在init函数执行变为在forward函数执行

（3）若分辨率小于window size或不是其公倍数时，在其右侧和底部Padding

*debug：在计算完之后需要反padding回去，否则与cv2支路的img_size无法对齐*

（4）forward函数前后对输入输出reshape

（5）验证C3STR时，需要手动关闭默认模型在half精度下验证（*--nohalf*）

### 3、Slighter Model

Train on Visdrone DataSet (*1 size is 608，2-6 size is 640*)

| No   | Model                     | mAP       | mAP@50 | Parameters(M) | GFLOPs   | TrainCost(h) | Memory Cost(G) | PT File                                                      | FPS@CPU |
| ---- | ------------------------- | --------- | ------ | ------------- | -------- | ------------ | -------------- | ------------------------------------------------------------ | ------- |
| 1    | YOLOv5l                   | 23.2      | 39.7   | 46.19         | 108.1    |              |                |                                                              |         |
| 2    | YOLOv5l-GhostNet          | 18.4      | 33.8   | 24.27         | 42.4     | 27.44        | 4.97           | [PekingUni Cloud](https://disk.pku.edu.cn:443/link/35BD905E65DE091E2A58316B20BBE775) |         |
| 3    | YOLOv5l-ShuffleNetV2      | 16.48     | 31.1   | 21.27         | 40.5     | 10.98        | 2.41           | [PekingUni Cloud](https://disk.pku.edu.cn:443/link/A5ED89B7B190FCF1C8187A0A8AF20C4F) |         |
| 4    | YOLOv5l-MobileNetv3Small  | 16.55     | 31.2   | **20.38**     | **38.4** | **10.19**    | 5.30           | [PekingUni Cloud](https://disk.pku.edu.cn:443/link/EE375ED30AAD3F2B3FA5055DD6F4964C) |         |
| 5    | YOLOv5l-EfficientNetLite0 | **19.12** | **35** | 23.01         | 43.9     | 13.94        | 2.04           | [PekingUni Cloud](https://disk.pku.edu.cn:443/link/45E65A080C4574036EE274B7BD83B7EA) |         |
| 6    | YOLOv5l-PP-LCNet          | 17.63     | 32.8   | 21.64         | 41.7     | 18.52        | **1.66**       | [PekingUni Cloud](https://disk.pku.edu.cn:443/link/7EBE07BA6D7985C7053BF0A8F2591464) |         |

#### A、GhostNet-YOLOv5 ![](https://img.shields.io/badge/Model-HuaWei-orange.svg?style=plastic)

<img src="https://github.com/Gumpest/YOLOv5-Multibackbone-Compression/blob/main/img/GhostNet.jpg" alt="GhostNet" width="500px" height="250px" />

（1）为保持一致性，下采样的DW的kernel_size均等于3

（2）neck部分与head部分沿用YOLOv5l原结构

（3）中间通道人为设定（expand）

#### B、ShuffleNetV2-YOLOv5 ![](https://img.shields.io/badge/Model-Megvii-orange.svg?style=plastic)

<img src="https://github.com/Gumpest/YOLOv5-Multibackbone-Compression/blob/main/img/Shffulenet.png" alt="Shffulenet" width="400px" />

（1）Focus Layer不利于芯片部署，频繁的slice操作会让缓存占用严重

（2）避免多次使用C3 Leyer以及高通道的C3 Layer（违背G1与G3准则）

（3）中间通道不变

#### C、MobileNetv3Small-YOLOv5 ![](https://img.shields.io/badge/Model-Google-orange.svg?style=plastic)

<img src="https://github.com/Gumpest/YOLOv5-Multibackbone-Compression/blob/main/img/Mobilenetv3s.jpg" alt="Mobilenetv3s" width="500px" />

（1）原文结构，部分使用Hard-Swish激活函数以及SE模块

（2）Neck与head部分嫁接YOLOv5l原结构

（3）中间通道人为设定（expand）

#### D、EfficientNetLite0-YOLOv5 ![](https://img.shields.io/badge/Model-Google-orange.svg?style=plastic)


<img src="https://github.com/Gumpest/YOLOv5-Multibackbone-Compression/blob/main/img/efficientlite.jpg" alt="efficientlite" width="500px" />

（1）使用Lite0结构，且不使用SE模块

（2）针对dropout_connect_rate，手动赋值(随着idx_stage变大而变大)

（3）中间通道一律*6（expand）

#### E、PP-LCNet-YOLOv5  ![](https://img.shields.io/badge/Model-Baidu-orange.svg?style=plastic)


<img src="https://github.com/Gumpest/YOLOv5-Multibackbone-Compression/blob/main/img/PP-LCNet.png" alt="PP-LCNet" width="500px" />


（1）使用PP-LCNet-1x结构，在网络末端使用SE以及5*5卷积核

（2）SeBlock压缩维度为原1/16

（3）中间通道不变

## Pruning for YOLOs

| Model                | mAP  | mAP@50 | Parameters(M) | GFLOPs | FPS@CPU |
| -------------------- | ---- | ------ | ------------- | ------ | ------- |
| YOLOv5s              | 18.4 | 34     | 7.05          | 15.9   |         |
| YOLOv5n              | 13   | 26.2   | 1.78          | 4.2    |         |
| YOLOv5s-EagleEye@0.6 | 14.3 | 27.9   | 4.59          | 9.6    |         |

### 1、Prune Strategy

（1）基于YOLOv5块状结构设计，对Conv、C3、SPP(F)模块进行剪枝，具体来说有以下：

- Conv模块的输出通道数
- C3模块中cv2块和cv3块的输出通道数
- C3模块中若干个bottleneck中的cv1块的输出通道数

（2）八倍通道剪枝（outchannel = 8*n）

（3）ShortCut、concat皆合并剪枝

### 2、Prune Tools

#### （1）EagleEye

[EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning](https://arxiv.org/abs/2007.02491)

基于搜索的通道剪枝方法，核心思想是随机搜索到大量符合目标约束的子网，然后快速更新校准BN层的均值与方差参数，并在验证集上测试校准后全部子网的精度。精度最高的子网拥有最好的架构，经微调恢复后能达到较高的精度。

![eagleeye](https://github.com/Cydia2018/YOLOv5-Multibackbone-Compression/blob/main/img/eagleeye.png)

##### Usage

1. 正常训练模型

```shell
python train.py --data data/VisDrone.yaml --imgsz 640 --weights yolov5s.pt --cfg models/prunModels/yolov5s-pruning.yaml --device 0
```

（注意训练其他模型，参考/prunModels/yolov5s-pruning.yaml进行修改，目前已支持v6架构）

2. 搜索最优子网

```shell
python pruneEagleEye.py --weights path_to_trained_yolov5_model --cfg models/prunModels/yolov5s-pruning.yaml --data data/VisDrone.yaml --path path_to_pruned_yolov5_yaml --max_iter maximum number of arch search --remain_ratio the whole FLOPs remain ratio --delta 0.02
```

3. 微调恢复精度

```shell
python train.py --data data/VisDrone.yaml --imgsz 640 --weights path_to_Eaglepruned_yolov5_model --cfg path_to_pruned_yolov5_yaml --device 0
```

#### （2）Network Slimming

[Learning Efficient Convolutional Networks through Network Slimming](https://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf)

##### Usage

1. 模型BatchNorm Layer \gamma 稀疏化训练

```shell
python train.py --data data/VisDrone.yaml --imgsz 640 --weights yolov5s.pt --cfg models/prunModels/yolov5s-pruning.yaml --device 0 --sparse
```

（注意训练其他模型，参考/prunModels/yolov5s-pruning.yaml进行修改，目前已支持v6架构）

2. BatchNorm Layer剪枝

```shell
python pruneSlim.py --weights path_to_sparsed_yolov5_model --cfg models/prunModels/yolov5s-pruning.yaml --data data/VisDrone.yaml --path path_to_pruned_yolov5_yaml --global_percent 0.6 --device 3
```

3. 微调恢复精度

```shell
python train.py --data data/VisDrone.yaml --imgsz 640 --weights path_to_Slimpruned_yolov5_model --cfg path_to_pruned_yolov5_yaml --device 0
```

## Quantize Aware Training for YOLOs

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

## Deploy
目前已支持TensorRT及NCNN部署，详见[YOLOv5-Multibackbone-Compression/deploy](https://github.com/Gumpest/YOLOv5-Multibackbone-Compression/blob/main/deploy)

## To do

- [x] Multibackbone: MobileNetV3-small
- [x] Multibackbone: ShuffleNetV2
- [x] Multibackbone: GhostNet
- [x] Multibackbone: EfficientNet-Lite0
- [x] Multibackbone: PP-LCNet
- [x] Multibackbone: TPH-YOLOv5
- [x] Module: SwinTrans（C3STR）
- [ ] Module: Deformable Convolution
- [x] Pruner: Network Slimming
- [x] Pruner: EagleEye
- [ ] Pruner: OneShot (L1, L2, FPGM), ADMM, NetAdapt, Gradual, End2End
- [x] Quantization: MQBench
- [ ] Knowledge Distillation

## Acknowledge

感谢TPH-YOLOv5作者Xingkui Zhu 

官方实现[cv516Buaa/tph-yolov5 (github.com)](https://github.com/cv516Buaa/tph-yolov5)

感谢[ZJU-lishuang/yolov5_prune: yolov5剪枝，支持v2,v3,v4,v6版本的yolov5 (github.com)](https://github.com/ZJU-lishuang/yolov5_prune)