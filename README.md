# YOLOv5-Compression



## Update News



## Requirements

环境安装

`pip install -r requirements.txt`


## Evaluation metric

Visdrone

| Model   | mAP  | mAP@50 | Parameters(M) | GFLOPs | FPS@CPU |
| ------- | ---- | ------ | ------------- | ------ | ------- |
| YOLOv5n | 13   | 26.2   | 1.78          | 4.2    |         |
| YOLOv5s | 18.4 | 34     | 7.05          | 15.9   |         |
| YOLOv5m | 21.6 | 37.8   | 20.91         | 48.2   |         |
| YOLOv5l | 23.2 | 39.7   | 46.19         | 108.1  |         |
| YOLOv5x | 24.3 | 40.8   | 86.28         | 204.4  |         |

yolov5n

```shell
nohup python train.py --data VisDrone.yaml --weights yolov5n.pt --cfg models/yolov5n.yaml --epochs 300 --batch-size 8 --img 608 --nosave --device 0,1 --sync-bn >> yolov5n.txt &
```

## To do

- [ ] Multibackbone: mobilenetv3
- [ ] Multibackbone: shufflenetv2
- [ ] Multibackbone: ghostnet
- [ ] Multibackbone: TPH-YOLOv5
- [ ] Pruning: Network slimming
- [ ] Quantization: 4bit QAT
- [ ] Knowledge Distillation