# Deploy
In this section, we are trying to deploy light-weight, pruned or quantized YOLOv5 via different inference framework on different devices.

## TensorRT
### 1. Install
```shell
pip install -U nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com
```
### 2. Export
```shell
python export_onnx_trt.py --weights yolov5s.pt --device 0
```
Now get **yolov5s.engine**.
### 3. Detect
```shell
python export_onnx_trt.py --weights yolov5s.engine --device 0
```


## ncnn
### 1. Install
Check [Tencent/ncnn](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux) for help.
### 2. Export
```shell
python export_onnx_trt.py --weights yolov5s.pt --device 0 --train --simplify
```
Now get **yolov5s.onnx**. Then use NCNN's *onnx2ncnn* tool to convert *.onnx to *.param and *.bin.  
Navigate to *ncnn/build/tools/onnx*, run
```shell
./onnx2ncnn yolov5s.onnx yolov5s.param yolov5s.bin
```
You can also use NCNN's *ncnnoptimize* tool to reduce model size.
### 3. Detect
We provide **yolov5_ncnn.cpp** for detection and timekeeping. Build it with NCNN, and run
```shell
./yolov5 test.jpg yolov5s
```

## Experiment
### On RTX3090
| Backend  | Model                   | File Size | latency(ms per img)|
| -------- | --------------------    |  ------   | ------- |
| TensorRT |YOLOv5s                  | 17M       |   2.3   |
| TensorRT |YOLOv5s-EagleEye@0.6     | 11M       |   2.0   |
| TensorRT |YOLOv5l-MobileNetv3Small | 44M       |   2.9   |
| TensorRT |YOLOv5l-EfficientNetLite0| 47M       |   3.0   |
|          |                         |           |         |
|ncnn(Vulkan)|YOLOv5s                |14M        |   235   |
| ncnn(Vulkan)|YOLOv5s-EagleEye@0.6  |7.5M       |   215   |

*Input size is 640x640.*  

### On Jetson Xavier NX
| Backend    | Model                | File Size | latency(ms per img)|
| --------   | -------------------- |  ------   | ------- |
|ncnn(Vulkan)|YOLOv5s               |  14M      |   520   |
|ncnn(Vulkan)|YOLOv5s-EagleEye@0.6  |  7.5M     |   610   |

*Input size is 640x640.*  


##  More statistics is coming soon...