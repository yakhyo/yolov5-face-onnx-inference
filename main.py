"""
YOLOv5-Face ONNX Inference Script

Author: Yakhyokhuja Valikhujaev
Date: 2025-12-01
Description: YOLOv5-Face ONNX inference with facial landmarks
Copyright (c) 2025 Yakhyokhuja Valikhujaev. All rights reserved.

Usage:
    python main.py --weights weights/yolov5s-face.onnx --source assets/bus.jpg
    python main.py --weights weights/yolov5m-face.onnx --source 0  # webcam
    python main.py --weights weights/yolov5s-face.onnx --source video.mp4
"""

import argparse
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

from models import YOLOv5
from utils.general import LoadMedia, check_img_size, increment_path, scale_boxes


def draw_face_detections(image: np.ndarray, box: np.ndarray, score: float, landmarks: np.ndarray) -> None:
    """
    Draw face bounding box and landmarks on image.

    Args:
        image: Input image
        box: Bounding box [x1, y1, x2, y2]
        score: Confidence score
        landmarks: Facial landmarks [10] (5 points * 2 coords)
    """
    x1, y1, x2, y2 = map(int, box)

    # Draw bounding box
    color = (0, 255, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Draw confidence score
    label = f"Face {score:.2f}"
    font_size = min(image.shape[:2]) * 0.0006
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)

    # Create filled rectangle for text background
    cv2.rectangle(image, (x1, y1 - int(1.3 * text_height)), (x1 + text_width, y1), color, -1)

    # Put text on the image
    cv2.putText(
        image,
        label,
        (x1, y1 - int(0.3 * text_height)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )

    # Draw facial landmarks
    landmarks = landmarks.reshape(5, 2).astype(int)
    for i, (lx, ly) in enumerate(landmarks):
        # Red for eyes, blue for nose, green for mouth
        if i < 2:
            landmark_color = (0, 0, 255)  # Red for eyes
        elif i == 2:
            landmark_color = (255, 0, 0)  # Blue for nose
        else:
            landmark_color = (0, 255, 0)  # Green for mouth
        cv2.circle(image, (lx, ly), 3, landmark_color, -1)


def run_face_detection(
    weights: str,
    source: str,
    img_size: List[int],
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    save_img: bool,
    view_img: bool,
    project: str,
    name: str,
) -> None:
    """
    Run face detection on image, video, or webcam.

    Args:
        weights: Path to ONNX model file
        source: Path to image/video file or webcam index
        img_size: Inference image size [height, width]
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        max_det: Maximum detections per image
        save_img: Whether to save results
        view_img: Whether to display results
        project: Save results to project/name
        name: Save results to project/name
    """
    # Create save directory
    if save_img:
        save_dir = increment_path(Path(project) / name)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = YOLOv5(weights, conf_thres, iou_thres, img_size[0], max_det)
    img_size = check_img_size(img_size, s=model.stride)

    # Load media
    dataset = LoadMedia(source, img_size=tuple(img_size))

    # Auto-enable view_img for webcam
    if dataset.type == "webcam":
        view_img = True
        print("Webcam detected - auto-enabling display. Press 'q' to quit.")

    # For writing video and webcam
    vid_writer = None
    if save_img and dataset.type in ["video", "webcam"]:
        cap = dataset.cap
        save_path = str(save_dir / f"result_{os.path.basename(source)}.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Run inference
    for resized_image, original_image, status in dataset:
        # Model inference
        boxes, scores, landmarks = model(resized_image)

        # Scale bounding boxes to original image size
        if len(boxes) > 0:
            boxes = scale_boxes(resized_image.shape, boxes, original_image.shape).round()

            # Scale landmarks to original image size
            scale = min(
                resized_image.shape[0] / original_image.shape[0],
                resized_image.shape[1] / original_image.shape[1],
            )
            dw = (resized_image.shape[1] - original_image.shape[1] * scale) / 2
            dh = (resized_image.shape[0] - original_image.shape[0] * scale) / 2

            for i in range(5):
                landmarks[:, i * 2] = (landmarks[:, i * 2] - dw) / scale
                landmarks[:, i * 2 + 1] = (landmarks[:, i * 2 + 1] - dh) / scale

            # Draw detections
            for box, score, landmark in zip(boxes, scores, landmarks):
                draw_face_detections(original_image, box, score, landmark)

        # Print results
        n_faces = len(boxes)
        status += f"{n_faces} face{'s' * (n_faces != 1)}"
        print(status)

        # Display results
        if view_img:
            cv2.imshow("YOLOv5-Face ONNX Inference", original_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
                break

        # Save results
        if save_img:
            if dataset.type == "image":
                save_path = str(save_dir / f"result_{dataset.frame:04d}.jpg")
                cv2.imwrite(save_path, original_image)
            elif dataset.type in ["video", "webcam"]:
                vid_writer.write(original_image)

    # Cleanup
    if save_img and vid_writer is not None:
        vid_writer.release()

    if save_img:
        print(f"Results saved to {save_dir}")

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLOv5-Face ONNX Inference")
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/yolov5s_face.onnx",
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Path to image/video file or webcam index",
    )
    parser.add_argument(
        "--img-size",
        nargs="+",
        type=int,
        default=[640],
        help="Inference size h,w",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.45,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum detections per image",
    )
    parser.add_argument(
        "--save-img",
        action="store_true",
        help="Save detected images",
    )
    parser.add_argument(
        "--view-img",
        action="store_true",
        help="Display results (auto-enabled for webcam)",
    )
    parser.add_argument(
        "--project",
        default="runs",
        help="Save results to project/name",
    )
    parser.add_argument(
        "--name",
        default="exp",
        help="Save results to project/name",
    )

    args = parser.parse_args()
    args.img_size = args.img_size * 2 if len(args.img_size) == 1 else args.img_size

    return args


def main() -> None:
    """Main function."""
    params = parse_args()

    run_face_detection(
        weights=params.weights,
        source=params.source,
        img_size=params.img_size,
        conf_thres=params.conf_thres,
        iou_thres=params.iou_thres,
        max_det=params.max_det,
        save_img=params.save_img,
        view_img=params.view_img,
        project=params.project,
        name=params.name,
    )


if __name__ == "__main__":
    main()
