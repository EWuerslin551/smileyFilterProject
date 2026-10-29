"""
Live webcam filter: replaces each detected face with smiley.png

Needs in the same folder:
    - smiley.png (transparent PNG)
    - YuNet model (downloaded automatically)

Press 'q' to quit.
"""

import os
import sys
import urllib.request
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(SCRIPT_DIR, "face_detection_yunet_2023mar.onnx")
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

SMILEY_PATH = os.path.join(SCRIPT_DIR, "smiley.png")

# Increase or decrease if the smiley is too large/small
FACE_SCALE = 1.5


def load_model():
    """Download the YuNet model if needed."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100000:
        return

    print("Downloading YuNet face model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


def overlay_smiley(frame, smiley, face):
    """
    Overlay a smiley centered on a detected face.
    face format:
    [x, y, w, h, eye1_x, eye1_y, eye2_x, eye2_y, nose_x, nose_y, ...]
    """

    face_x, face_y, face_w, face_h = map(int, face[:4])

    # Resize smiley based on face width
    target_size = int(face_w * FACE_SCALE)

    resized = cv2.resize(
        smiley,
        (target_size, target_size),
        interpolation=cv2.INTER_AREA
    )

    h, w = resized.shape[:2]

    center_x = face_x + face_w // 2
    center_y = face_y + face_h // 2

    x = center_x - w // 2
    y = center_y - h // 2

    frame_h, frame_w = frame.shape[:2]

    # Clip to screen boundaries
    if x < 0:
        resized = resized[:, -x:]
        w = resized.shape[1]
        x = 0

    if y < 0:
        resized = resized[-y:, :]
        h = resized.shape[0]
        y = 0

    if x + w > frame_w:
        resized = resized[:, :frame_w - x]
        w = resized.shape[1]

    if y + h > frame_h:
        resized = resized[:frame_h - y, :]
        h = resized.shape[0]

    if w <= 0 or h <= 0:
        return frame

    roi = frame[y:y+h, x:x+w]

    alpha = resized[:, :, 3:4].astype(np.float32) / 255.0

    blended = (
        roi.astype(np.float32) * (1 - alpha)
        + resized[:, :, :3].astype(np.float32) * alpha
    )

    frame[y:y+h, x:x+w] = blended.astype(np.uint8)

    return frame


def main():
    load_model()

    smiley = cv2.imread(SMILEY_PATH, cv2.IMREAD_UNCHANGED)

    if smiley is None:
        print(f"Error: Could not load {SMILEY_PATH}")
        return

    if smiley.shape[2] != 4:
        print("Error: smiley.png must have transparency (RGBA).")
        return

    if sys.platform.startswith("win"):
        backend = cv2.CAP_DSHOW
    else:
        backend = cv2.CAP_ANY

    cap = cv2.VideoCapture(0, backend)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH,
        "",
        (640, 480),
        score_threshold=0.8
    )

    print("Running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        detector.setInputSize((w, h))

        _, faces = detector.detect(frame)

        if faces is not None:
            for face in faces:
                frame = overlay_smiley(frame, smiley, face)

        cv2.imshow("Smiley Filter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()