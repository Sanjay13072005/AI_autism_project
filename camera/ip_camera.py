import cv2

def get_camera(source="ip"):
    """
    source: 'ip' for phone camera, 'local' for laptop webcam
    """
    if source == "ip":
        # CHANGE THIS TO YOUR PHONE IP
        url = "http://10.123.47.217:8080/video"
        cap = cv2.VideoCapture(url)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("❌ Camera not accessible")

    return cap
