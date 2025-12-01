"""
YOLOv5-Face ONNX Inference Class

Author: Yakhyokhuja Valikhujaev
Date: 2025-12-01
Description: YOLOv5-Face ONNX inference with facial landmarks
Copyright (c) 2025 Yakhyokhuja Valikhujaev. All rights reserved.
"""

from typing import List, Tuple

import numpy as np
import onnxruntime
import torch
import torchvision


class YOLOv5:
    """YOLOv5-Face ONNX inference class."""

    def __init__(
        self,
        model_path: str,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        img_size: int = 640,
        max_det: int = 300,
        nms_mode: str = "torchvision",
    ) -> None:
        """
        Initialize YOLOv5-Face ONNX model.

        Args:
            model_path (str): Path to ONNX model file
            conf_thres (float, optional): Confidence threshold for detections. Defaults to 0.25.
            iou_thres (float, optional): IoU threshold for NMS. Defaults to 0.45.
            img_size (int, optional): Input image size. Defaults to 640.
            max_det (int, optional): Maximum number of detections. Defaults to 300.
            nms_mode (str, optional): NMS calculation method ('torchvision', 'numpy'). Defaults to 'torchvision'.
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.max_det = max_det

        # Set NMS mode with automatic fallback
        if nms_mode == "torchvision":
            self.nms_mode = nms_mode
        else:
            print("Warning: NumPy NMS is not supported, falling back to TorchVision NMS")
            self.nms_mode = nms_mode

        # Initialize model
        self._initialize_model(model_path)

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the model on the given image and return predictions.

        Args:
            image (np.ndarray): Input image (preprocessed/resized).

        Returns:
            Tuple: (boxes, scores, landmarks) where:
                - boxes: [N, 4] bounding boxes in xyxy format
                - scores: [N] confidence scores
                - landmarks: [N, 10] facial landmarks (5 points * 2 coords)
        """
        if not isinstance(image, np.ndarray) or len(image.shape) != 3:
            raise ValueError("Input image must be a numpy array with 3 dimensions (H, W, C).")

        detections = self.detect(image)

        if len(detections) == 0:
            return np.array([]), np.array([]), np.array([])

        boxes = detections[:, :4]
        scores = detections[:, 4]
        landmarks = detections[:, 5:]

        return boxes, scores, landmarks

    def _initialize_model(self, model_path: str) -> None:
        """
        Initialize the model from the given path.

        Args:
            model_path (str): Path to .onnx model.
        """
        try:
            self.session = onnxruntime.InferenceSession(
                model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            # Get model info
            self.output_names = [x.name for x in self.session.get_outputs()]
            self.input_names = [x.name for x in self.session.get_inputs()]

            # Get model metadata
            meta = self.session.get_modelmeta()
            if meta.custom_metadata_map:
                self.stride = int(meta.custom_metadata_map.get("stride", 32))
            else:
                self.stride = 32

            print(f"Loaded ONNX model: {model_path}")
            print(f"Input names: {self.input_names}")
            print(f"Output names: {self.output_names}")
            print(f"Stride: {self.stride}")
            print(f"NMS mode: {self.nms_mode}")
        except Exception as e:
            print(f"Failed to load the model: {e}")
            raise

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference.

        Args:
            img (np.ndarray): Input image (BGR format, already resized with letterbox)

        Returns:
            np.ndarray: Preprocessed image tensor
        """
        # Convert BGR to RGB
        img = img[:, :, ::-1]

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Transpose to CHW format and add batch dimension
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)

        return img

    def postprocess(self, predictions: np.ndarray) -> np.ndarray:
        """
        Postprocess model predictions.

        Args:
            predictions (np.ndarray): Raw model output

        Returns:
            np.ndarray: Filtered detections [x1, y1, x2, y2, conf, landmarks...]
        """
        # predictions shape: (1, 25200, 16)
        # 16 = [x, y, w, h, obj_conf, cls_conf, 10 landmarks (5 points * 2 coords)]
        predictions = predictions[0]  # Remove batch dimension

        # Filter by confidence
        mask = predictions[:, 4] >= self.conf_thres
        predictions = predictions[mask]

        if len(predictions) == 0:
            return np.array([])

        # Convert from xywh to xyxy
        boxes = self.xywh2xyxy(predictions[:, :4])

        # Get confidence scores
        scores = predictions[:, 4]

        # Get landmarks (5 points, 10 coordinates)
        landmarks = predictions[:, 5:15].copy()

        # Apply NMS
        if self.nms_mode == "torchvision":
            # Better performance
            indices = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), self.iou_thres).numpy()
        else:
            indices = self.nms(boxes, scores, self.iou_thres)

        if len(indices) == 0:
            return np.array([])

        # Filter detections and limit to max_det
        indices = indices[: self.max_det]
        boxes = boxes[indices]
        scores = scores[indices]
        landmarks = landmarks[indices]

        # Combine results
        detections = np.concatenate([boxes, scores[:, None], landmarks], axis=1)

        return detections

    def xywh2xyxy(self, x: np.ndarray) -> np.ndarray:
        """
        Convert bounding box format from xywh to xyxy.

        Args:
            x (np.ndarray): Boxes in [x, y, w, h] format

        Returns:
            np.ndarray: Boxes in [x1, y1, x2, y2] format
        """
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
        return y

    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """
        Non-Maximum Suppression (NumPy implementation).

        Args:
            boxes (np.ndarray): Bounding boxes [x1, y1, x2, y2]
            scores (np.ndarray): Confidence scores
            iou_threshold (float): IoU threshold

        Returns:
            List[int]: Indices of boxes to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def detect(self, img: np.ndarray) -> np.ndarray:
        """
        Run face detection on image.

        Args:
            img (np.ndarray): Input image (BGR format, already resized with letterbox)

        Returns:
            np.ndarray: Detections [x1, y1, x2, y2, conf, landmarks...]
        """
        # Preprocess
        input_tensor = self.preprocess(img)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # Postprocess
        detections = self.postprocess(outputs[0])

        return detections
