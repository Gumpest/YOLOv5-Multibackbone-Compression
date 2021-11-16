# YOLOv5-Compression



## Update News

2021.10.30 复现TPH-YOLOv5

2021.10.31 完成替换backbone为Ghostnet

2021.11.02 完成替换backbone为Shufflenetv2

2021.11.05 完成替换backbone为Mobilenetv3Small

## Requirements

环境安装

`pip install -r requirements.txt`


## Evaluation metric

Visdrone DataSet (1-5 size is 608，6-8 size is 640)

| Model          | mAP   | mAP@50 | Parameters(M) | GFLOPs | FPS@CPU |
| -------------- | ----- | ------ | ------------- | ------ | ------- |
| YOLOv5n        | 13    | 26.2   | 1.78          | 4.2    |         |
| YOLOv5s        | 18.4  | 34     | 7.05          | 15.9   |         |
| YOLOv5m        | 21.6  | 37.8   | 20.91         | 48.2   |         |
| YOLOv5l        | 23.2  | 39.7   | 46.19         | 108.1  |         |
| YOLOv5x        | 24.3  | 40.8   | 86.28         | 204.4  |         |
| YOLOv5xP2      | 30.00 | 49.29  | 90.96         | 314.2  |         |
| YOLOv5xP2 CBAM | 30.13 | 49.40  | 91.31         | 315.1  |         |
| YOLOv5x-TPH    | 30.26 | 49.39  | 86.08         | 238.9  |         |

训练脚本实例：

```shell
nohup python train.py --data VisDrone.yaml --weights yolov5n.pt --cfg models/yolov5n.yaml --epochs 300 --batch-size 8 --img 608 --nosave --device 0,1 --sync-bn >> yolov5n.txt &
```

## Multi-Backbone

### 精度区

#### TPH-YOLOv5

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

| box  | cls  | obj  | acc   |
| ---- | ---- | ---- | ----- |
| 0.05 | 0.5  | 1.0  | 37.90 |
| 0.05 | 0.3  | 0.7  | 38.00 |
| 0.05 | 0.2  | 0.4  |       |

![Loss](https://github.com/Gumpest/YOLOv5-Multibackbone-Compression/blob/main/img/image-20211109150606998.png)

### 轻量区

| Model                    | mAP   | mAP@50 | Parameters(M) | GFLOPs | FPS@CPU | TrainCost(h) | Memory Cost(G) |
| ------------------------ | ----- | ------ | ------------- | ------ | ------- | ------------ | -------------- |
| YOLOv5s                  | 18.4  | 34     | 7.05          | 15.9   |         |              |                |
| YOLOv5l-Ghostnet         | 18.4  | 33.8   | 24.27         | 42.4   |         | 27.44        | 4.97           |
| YOLOv5l-Shufflenet       | 16.48 | 31.1   | 21.27         | 40.5   |         | 10.98        | 2.41           |
| YOLOv5l-Mobilenetv3Small | 16.55 | 31.2   | 20.38         | 38.4   |         | 10.19        | 5.3            |

#### Ghostnet-YOLOv5

（1）为保持一致性，下采样的DW的kernel_size均等于3

（2）neck部分与head部分沿用YOLOv5l原结构

#### Shufflenet-YOLOv5

（1）Focus Layer不利于芯片部署，频繁的slice操作会让缓存占用严重

（2）避免多次使用C3 Leyer以及高通道的C3 Layer（违背G1与G3准则）

#### Mobilenetv3Small-YOLOv5

（1）尊重原文结构，精确使用hard-Swish以及SE层

（2）neck部分与head部分沿用YOLOv5l原结构

## MQBench Quantize Aware Training
 MQBench是在实际硬件部署下评估量化算法的基准和框架。可以使用MQBench进行各种适合于硬件部署的量化训练。
### Prerequisites
- PyTorch == 1.8.1
### Install MQBench Lib
由于MQBench目前还在不断更新，选择0.0.2稳定版本作为本仓库的量化库。
```
git clone https://github.com/ZLkanyo009/MQBench.git
cd MQBench
python setup.py build
python setup.py install
```
### How to run QAT
训练脚本实例：

```shell
python train.py --data VisDrone.yaml --weights yolov5n.pt --cfg models/yolov5n.yaml --epochs 300 --batch-size 8 --img 608 --nosave --device 0,1 --sync-bn --quantize --BackendType NNIE
```
## To do

- [x] Multibackbone: Mobilenetv3-small
- [x] Multibackbone: Shufflenetv2
- [x] Multibackbone: Ghostnet
- [ ] Multibackbone: EfficientNet-Lite
- [x] Multibackbone: TPH-YOLOv5
- [ ] Pruner: Network slimming
- [ ] Pruner: OneShot (L1, L2, FPGM), ADMM, NetAdapt, Gradual, End2End
- [ ] Quantization: 4bit QAT
- [ ] Knowledge Distillation