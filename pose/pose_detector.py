import torch
import cv2
from ultralytics import YOLO

class PoseDetector:
    def __init__(self):
        # YOLOv8 pose model (auto-downloads)
        self.model = YOLO("yolov8n-pose.pt")
        self.device = "cpu"
        self.model.to(self.device)

    def extract_pose(self, frame):
        results = self.model(frame, verbose=False)

        if len(results[0].keypoints) == 0:
            return None

        keypoints = results[0].keypoints.xy[0]  # (17, 2)
        return keypoints
