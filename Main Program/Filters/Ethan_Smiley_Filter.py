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
import math
from collections import deque
# Remember the last 3 detected faces
face_history = deque(maxlen=5)

import numpy as np
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MAIN_DIR = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(
    MAIN_DIR, 
    "Visuals", 
    "face_detection_yunet_2023mar.onnx"
)

MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

SMILEY_PATH = os.path.join(MAIN_DIR, "Assets", "smiley.png")
STEVE_PATH = os.path.join(MAIN_DIR, "Assets", "steve_crushed.png")

# Increase or decrease if the smiley is too large/small
FACE_SCALE = 2


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

    face_x, face_y, face_w, face_h, eye1_x, eye1_y, eye2_x, eye2_y = map(int, face[:8])


    # Resize smiley based on face width
    target_size = int(face_w * FACE_SCALE)

    if target_size < 100: # Removes filter for a person if they are more than 3ish tiles
        return frame

    resized = cv2.resize(
        smiley,
        (target_size, target_size),
        interpolation=cv2.INTER_AREA
    )

    h, w = resized.shape[:2]

    center_x = face_x + face_w // 2
    center_y = face_y + face_h // 2

    dy = (eye1_y-eye2_y)
    dx = (eye1_x-eye2_x)

    rotationAngle = math.degrees(
        math.atan2(dy, dx)
    ) + 180

    print(rotationAngle)
    
    M = cv2.getRotationMatrix2D(
        (w // 2, h // 2),
        -rotationAngle,
        1
    )

    resized = cv2.warpAffine(resized, M, (w, h))

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
    steve = cv2.imread(STEVE_PATH, cv2.IMREAD_UNCHANGED)

    if smiley is None:
        print(f"Error: Could not load {SMILEY_PATH}")
        return

    if smiley.shape[2] != 4:
        print("Error: smiley.png must have transparency (RGBA).")
        return

    if steve is None:
        print(f"Error: Could not load {STEVE_PATH}")
        return

    if steve.shape[2] != 4:
        print("Error: steve.png must have transparency (RGBA).")
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

    starttime = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        detector.setInputSize((w, h))

        _, faces = detector.detect(frame)
        
        currentfilter = steve if (time.time() - starttime)%10 >= 5 else smiley
        
        if faces is not None:
            if len(faces) == 1:
                face_history.append(faces[0])
                smoothed_face = np.mean(face_history, axis=0)
                frame = overlay_smiley(frame,currentfilter,smoothed_face)
            else:
                face_history.clear()
                for face in faces:
                    frame = overlay_smiley(frame, currentfilter, face)
        elif len(face_history) > 0:
            smoothed_face = np.mean(face_history, axis=0)
            frame = overlay_smiley(frame,currentfilter,smoothed_face)
            face_history.popleft()

        cv2.imshow("Smiley Filter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
