# YOLOv5-Face ONNX Inference

This repository contains code and instructions for performing face detection with facial landmarks using YOLOv5-Face inference with ONNX Runtime.

## Features

- Inference using ONNX Runtime GPU (CUDA) for maximum performance
- Face detection with 5 facial landmarks (eyes, nose, mouth corners)
- Fast NMS using TorchVision for real-time inference
- Easy-to-use Python scripts for inference
- Supports multiple input formats: image, video, or webcam

## Installation

### Clone the Repository

```bash
git clone https://github.com/yakhyo/yolov5-face-onnx-inference.git
cd yolov5-face-onnx-inference
```

### Install Required Packages

```bash
pip install -r requirements.txt
```

**Note:** This project uses `onnxruntime-gpu` for GPU acceleration. Make sure you have CUDA installed on your system. If you only have CPU, replace `onnxruntime-gpu` with `onnxruntime` in `requirements.txt`.

## Weights

### Download Pre-trained Models

You can download YOLOv5-Face ONNX models using the provided download script:

```bash
# Download YOLOv5s-Face model
bash download.sh yolov5s_face

# Download YOLOv5m-Face model
bash download.sh yolov5m_face
```

### Available Models

- `yolov5s_face.onnx` - Small model (faster, less accurate)
- `yolov5m_face.onnx` - Medium model (balanced)

> **Note:** The weights are saved in FP32 format.

## Usage

### Image Inference

```bash
python main.py --weights weights/yolov5s_face.onnx --source path/to/image.jpg --save-img --view-img
```

### Video Inference

```bash
# Save video results
python main.py --weights weights/yolov5m_face.onnx --source path/to/video.mp4 --save-img

# Save and display video results
python main.py --weights weights/yolov5m_face.onnx --source path/to/video.mp4 --save-img --view-img
```

### Webcam Inference

```bash
# Display is auto-enabled for webcam, press 'q' to quit
python main.py --weights weights/yolov5s_face.onnx --source 0
```

### Command Line Arguments

```
usage: main.py [-h] [--weights WEIGHTS] [--source SOURCE] [--img-size IMG_SIZE [IMG_SIZE ...]]
               [--conf-thres CONF_THRES] [--iou-thres IOU_THRES] [--max-det MAX_DET]
               [--save-img] [--view-img] [--project PROJECT] [--name NAME]

options:
  -h, --help            show this help message and exit
  --weights WEIGHTS     Path to ONNX model file
  --source SOURCE       Path to image/video file or webcam index
  --img-size IMG_SIZE [IMG_SIZE ...]
                        Inference size h,w
  --conf-thres CONF_THRES
                        Confidence threshold
  --iou-thres IOU_THRES
                        NMS IoU threshold
  --max-det MAX_DET     Maximum detections per image
  --save-img            Save detected images
  --view-img            Display results (auto-enabled for webcam)
  --project PROJECT     Save results to project/name
  --name NAME           Save results to project/name
```

## Project Structure

```
yolov5-face-onnx-inference/
├── models/
│   ├── __init__.py
│   └── yolov5.py          # YOLOv5-Face ONNX inference class
├── utils/
│   └── general.py         # Utility functions (LoadMedia, letterbox, etc.)
├── weights/
│   ├── yolov5s_face.onnx  # Small model
│   └── yolov5m_face.onnx  # Medium model
├── main.py                # Main inference script
├── download.sh            # Script to download model weights
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Model Output

The model detects faces and provides:
- Bounding box coordinates (x1, y1, x2, y2)
- Confidence score
- 5 facial landmarks:
  - Left eye
  - Right eye
  - Nose tip
  - Left mouth corner
  - Right mouth corner

## Performance

- **CPU**: ~50-100ms per image (640x640)
- **GPU (CUDA)**: ~10-20ms per image (640x640)

## Reference

1. [YOLOv5](https://github.com/ultralytics/yolov5)
2. [YOLOv5-Face](https://github.com/deepcam-cn/yolov5-face)
3. [YOLOv5 ONNX Inference](https://github.com/yakhyo/yolov5-onnx-inference)

## License

Copyright (c) 2024 Yakhyokhuja Valikhujaev. All rights reserved.

## Author

**Yakhyokhuja Valikhujaev**

