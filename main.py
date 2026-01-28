import os
import sys
import cv2
import time
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==================================================
# PATH FIX
# ==================================================
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from camera.ip_camera import get_camera

# ==================================================
# CONFIG (UNCHANGED ACTIVITY)
# ==================================================
RUN_THRESHOLD = 0.030
WALK_THRESHOLD = 0.008

MOTION_AVG_FRAMES = 10
VOTE_FRAMES = 8

# -------- Sleep --------
EAR_THRESHOLD = 0.20
SLEEP_TIME = 10  # seconds eyes must stay closed

# ==================================================
# MODELS
# ==================================================
pose_model = YOLO("yolov8n-pose.pt")

# -------- Face Landmarker --------
base_options = python.BaseOptions(
    model_asset_path="face_landmarker.task"
)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)
face_detector = vision.FaceLandmarker.create_from_options(options)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ==================================================
# CAMERA (IP WEBCAM)
# ==================================================
cap = get_camera(source="ip")

# ==================================================
# BUFFERS
# ==================================================
prev_kp = None
motion_buffer = deque(maxlen=MOTION_AVG_FRAMES)
activity_buffer = deque(maxlen=VOTE_FRAMES)

eye_closed_start = None
sleeping = False

print("✅ Activity + Eye-based Sleep Monitoring Started")

# ==================================================
# HELPERS
# ==================================================
def leg_motion(curr, prev):
    if prev is None:
        return 0.0
    idx = [13, 14, 15, 16]
    return torch.norm(curr[idx] - prev[idx]).item()

def joint_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosang = torch.dot(ba, bc) / (torch.norm(ba) * torch.norm(bc) + 1e-6)
    return torch.acos(torch.clamp(cosang, -1.0, 1.0)).item()

def eye_aspect_ratio(lm, w, h):
    def ear(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    left = np.array([[lm[i].x * w, lm[i].y * h] for i in LEFT_EYE])
    right = np.array([[lm[i].x * w, lm[i].y * h] for i in RIGHT_EYE])
    return (ear(left) + ear(right)) / 2

# ==================================================
# MAIN LOOP
# ==================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    activity = "No person"
    confidence = 0.0

    # ================= POSE (ACTIVITY) =================
    results = pose_model(frame, verbose=False)

    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        kp = results[0].keypoints.xy[0].clone()
        kp[:, 0] /= w
        kp[:, 1] /= h

        motion = leg_motion(kp, prev_kp)
        prev_kp = kp
        motion_buffer.append(motion)
        avg_motion = sum(motion_buffer) / len(motion_buffer)

        left_knee = joint_angle(kp[11], kp[13], kp[15])
        right_knee = joint_angle(kp[12], kp[14], kp[16])
        avg_knee = (left_knee + right_knee) / 2

        if avg_motion > RUN_THRESHOLD:
            raw_activity = "running"
            confidence = 0.95
        elif avg_motion > WALK_THRESHOLD:
            raw_activity = "walking"
            confidence = 0.90
        else:
            if avg_knee < 2.3:
                raw_activity = "sitting"
                confidence = 0.88
            else:
                raw_activity = "standing"
                confidence = 0.85

        activity_buffer.append(raw_activity)
        activity = max(set(activity_buffer), key=activity_buffer.count)

        # ================= EYES (SLEEP STATE) =================
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        face_result = face_detector.detect(mp_image)

        if face_result.face_landmarks:
            lm = face_result.face_landmarks[0]
            ear_val = eye_aspect_ratio(lm, w, h)

            if ear_val < EAR_THRESHOLD:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                elif time.time() - eye_closed_start >= SLEEP_TIME:
                    sleeping = True
            else:
                eye_closed_start = None
                sleeping = False

            if sleeping:
                activity = "sleeping"
                confidence = min(
                    (time.time() - eye_closed_start) / SLEEP_TIME, 1.0
                )

    # ================= DISPLAY =================
    cv2.rectangle(frame, (0, 0), (520, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Activity: {activity}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.putText(frame, f"Confidence: {confidence:.2f}",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2)

    cv2.imshow("Smart AI Autism – Final Monitor", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
