# YOLOv5-Compression



## Update News

2021.10.30 复现TPH-YOLOv5

2021.10.31 完成替换backbone为Ghostnet

## Requirements

环境安装

`pip install -r requirements.txt`


## Evaluation metric

Visdrone DataSet (1-5 size is 608，6-8 size is 640)

| Model              | mAP   | mAP@50 | Parameters(M) | GFLOPs | FPS@CPU |
| ------------------ | ----- | ------ | ------------- | ------ | ------- |
| YOLOv5n            | 13    | 26.2   | 1.78          | 4.2    |         |
| YOLOv5s            | 18.4  | 34     | 7.05          | 15.9   |         |
| YOLOv5m            | 21.6  | 37.8   | 20.91         | 48.2   |         |
| YOLOv5l            | 23.2  | 39.7   | 46.19         | 108.1  |         |
| YOLOv5x            | 24.3  | 40.8   | 86.28         | 204.4  |         |
| YOLOv5xP2          | 30.00 | 49.29  | 90.96         | 314.2  |         |
| YOLOv5xP2 CBAM     | 30.13 | 49.40  | 91.31         | 315.1  |         |
| YOLOv5xP2 CBAM TPH |       |        | 86.08         | 238.9  |         |

训练脚本实例：

```shell
nohup python train.py --data VisDrone.yaml --weights yolov5n.pt --cfg models/yolov5n.yaml --epochs 300 --batch-size 8 --img 608 --nosave --device 0,1 --sync-bn >> yolov5n.txt &
```

## Multi-Backbone

### 精度区

#### TPH-YOLOv5

消融实验如下：

| Model           | mAP   | mAP@50 | Parameters(M) | GFLOPs | FPS@CPU |
| --------------- | ----- | ------ | ------------- | ------ | ------- |
| YOLOv5x         | 24.3  | 40.8   | 86.28         | 204.4  |         |
| YOLOv5xP2       | 30.1  | 49.3   | 90.96         | 314.2  |         |
| YOLOv5xP2 CBAM  | 30.13 | 49.40  | 91.31         | 315.1  |         |
| **YOLOv5x-TPH** | 25.17 | 42.83  | 86.08         | 238.9  |         |

组件：P2检测头、CBAM、Transformer Block

结构图如下：

![TPH-YOLOv5](https://github.com/Gumpest/YOLOv5-Multibackbone-Compression/blob/main/TPH-YOLOv5.png)

注意：

（1）避免TransBlock导致显存爆炸，MultiAttentionHead中将注意力头的数量减少至4，并且FFN中的两个全连接层从linear(c1, 4*c1)改为linear(c1, c1)，去掉GELU函数

（2）TransBlock的数量会根据YOLO规模的不同而同步改变

（3）TPH-YOLOv5显卡需求较高，若根据结构图进行复现，RTX3090下只能支持batch size=1，img_size=508，可根据个人需求适当减小TransBlock的数量

### 轻量区

| Model              | mAP   | mAP@50 | Parameters(M) | GFLOPs | FPS@CPU | TrainCost@h |
| ------------------ | ----- | ------ | ------------- | ------ | ------- | ----------- |
| YOLOv5s            | 18.4  | 34     | 7.05          | 15.9   |         |             |
| YOLOv5l-Ghostnet   | 18.4  | 33.8   | 24.27         | 42.4   |         | 27.44       |
| YOLOv5l-Shufflenet | 16.48 | 31.1   | 21.27         | 40.5   |         | 10.98       |

#### Ghostnet-YOLOv5

（1）为保持一致性，下采样的DW的kernel_size均等于3

（2）neck部分与head部分仍采用YOLOv5l原结构

#### Shufflenet-YOLOv5

（1）Focus Layer不利于芯片部署，频繁的slice操作会让缓存占用严重

（2）避免多次使用C3 Leyer以及高通道的C3 Layer（违背G1与G3准则）



## To do

- [ ] Multibackbone: mobilenetv3
- [x] Multibackbone: Shufflenetv2
- [x] Multibackbone: Ghostnet
- [x] Multibackbone: TPH-YOLOv5
- [ ] Pruning: Network slimming
- [ ] Quantization: 4bit QAT
- [ ] Knowledge Distillation