import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- CONFIG ----------------
EAR_THRESHOLD = 0.20
SLEEP_TIME = 10  # seconds eyes must stay closed

# ---------------- LOAD FACE LANDMARKER ----------------
base_options = python.BaseOptions(
    model_asset_path="face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# ---------------- CAMERA (LAPTOP) ----------------
cap = cv2.VideoCapture(0)

eye_closed_start = None
sleeping = False

print("✅ Eye-based Sleep Detection Started (FINAL & STABLE)")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)
    status = "AWAKE"

    if result.face_landmarks:
        lm = result.face_landmarks[0]

        left_eye = np.array([[lm[i].x * w, lm[i].y * h] for i in LEFT_EYE])
        right_eye = np.array([[lm[i].x * w, lm[i].y * h] for i in RIGHT_EYE])

        ear_val = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

        if ear_val < EAR_THRESHOLD:
            if eye_closed_start is None:
                eye_closed_start = time.time()
            elif time.time() - eye_closed_start >= SLEEP_TIME:
                sleeping = True
        else:
            eye_closed_start = None
            sleeping = False

        status = "SLEEPING" if sleeping else "AWAKE"

    cv2.rectangle(frame, (0, 0), (360, 60), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0) if status == "AWAKE" else (0, 0, 255),
        2
    )

    cv2.imshow("Sleep Monitor (Laptop Webcam)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
