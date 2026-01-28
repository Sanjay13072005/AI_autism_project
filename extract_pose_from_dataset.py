import os
import cv2
import torch
from ultralytics import YOLO
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DATASET_DIR = r"C:\Users\HAI\Music\cnn_autism\autism_Dataset"
SAVE_DIR = r"C:\Users\HAI\Music\cnn_autism\pose_features"
os.makedirs(SAVE_DIR, exist_ok=True)

# Read class folders dynamically
CLASSES = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

LABEL_MAP = {cls: idx for idx, cls in enumerate(CLASSES)}

print("Detected classes:", LABEL_MAP)

# =========================
# LOAD YOLOv8 POSE MODEL
# =========================
model = YOLO("yolov8n-pose.pt")
model.to("cpu")

X = []
y = []

# =========================
# PROCESS DATASET
# =========================
for cls in CLASSES:
    cls_path = os.path.join(DATASET_DIR, cls)
    print(f"\nProcessing class: {cls}")

    for img_name in tqdm(os.listdir(cls_path)):
        img_path = os.path.join(cls_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        results = model(img, verbose=False)

        # If no pose detected, skip
        if results[0].keypoints is None:
            continue
        if len(results[0].keypoints.xy) == 0:
            continue

        # Take first detected person and CLONE
        keypoints = results[0].keypoints.xy[0].clone()  # (17,2)

        h, w = img.shape[:2]

        # Normalize safely (NO inplace ops on inference tensor)
        keypoints = keypoints / torch.tensor([w, h])

        # Flatten → 34 features
        feature_vector = keypoints.reshape(-1)

        X.append(feature_vector)
        y.append(LABEL_MAP[cls])

# =========================
# SAVE TENSORS
# =========================
if len(X) == 0:
    raise RuntimeError("❌ No pose data extracted. Check images.")

X = torch.stack(X)
y = torch.tensor(y, dtype=torch.long)

torch.save(X, os.path.join(SAVE_DIR, "X.pt"))
torch.save(y, os.path.join(SAVE_DIR, "y.pt"))

print("\n✅ PHASE 2 COMPLETED SUCCESSFULLY")
print("X shape:", X.shape)
print("y shape:", y.shape)
