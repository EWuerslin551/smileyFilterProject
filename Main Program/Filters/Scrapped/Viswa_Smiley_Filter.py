"""
Smiley Face Hallway Filter

Requirements:
    pip install opencv-python numpy

Files:
    smiley.png
    face_detection_yunet_2023mar.onnx
    (downloaded automatically if missing)

Press Q to quit.
"""

import math
import os
import sys
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np


# ==========================================================
# CONFIGURATION
# ==========================================================

@dataclass
class Config:
    # Face detection
    DETECTION_THRESHOLD: float = 0.80
    MIN_FACE_WIDTH: int = 120
    EDGE_MARGIN: int = 40

    # Face orientation
    LOOKING_THRESHOLD: float = 0.15

    # Overlay
    FACE_SCALE: float = 2.0

    # Smoothing
    SMOOTHING_FACTOR: float = 0.20

    # Camera
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480


CONFIG = Config()


# ==========================================================
# FILE PATHS
# ==========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MAIN_DIR = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(
    MAIN_DIR,
    "Visuals",
    "face_detection_yunet_2023mar.onnx"
)

SMILEY_PATH = os.path.join(
    SCRIPT_DIR,
    "smiley.png"
)

MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

# ==========================================================
# DATA CLASSES
# ==========================================================

@dataclass
class FaceState:
    left_eye: tuple
    right_eye: tuple
    nose: tuple
    mouth_left: tuple
    mouth_right: tuple


# ==========================================================
# MODEL LOADING
# ==========================================================

def load_model():

    if (
        os.path.exists(MODEL_PATH)
        and os.path.getsize(MODEL_PATH) > 100_000
    ):
        return

    print("Downloading YuNet model...")

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_PATH
    )

    print("Download complete.")


# ==========================================================
# SMOOTHING
# ==========================================================

def smooth_point(old_point, new_point):

    alpha = CONFIG.SMOOTHING_FACTOR

    return (
        alpha * new_point[0]
        + (1 - alpha) * old_point[0],

        alpha * new_point[1]
        + (1 - alpha) * old_point[1]
    )


def smooth_face(old_face: FaceState, new_face: FaceState):

    return FaceState(
        left_eye=smooth_point(
            old_face.left_eye,
            new_face.left_eye
        ),

        right_eye=smooth_point(
            old_face.right_eye,
            new_face.right_eye
        ),

        nose=smooth_point(
            old_face.nose,
            new_face.nose
        ),

        mouth_left=smooth_point(
            old_face.mouth_left,
            new_face.mouth_left
        ),

        mouth_right=smooth_point(
            old_face.mouth_right,
            new_face.mouth_right
        )
    )


# ==========================================================
# FACE ORIENTATION
# ==========================================================

def looking_at_camera(
    left_eye,
    right_eye,
    mouth_left,
    mouth_right
):
    """
    Very rough frontal-face estimator.

    If the midpoint of the eyes and midpoint
    of the mouth are nearly vertically aligned,
    the person is approximately facing forward.
    """

    eye_center_x = (
        left_eye[0]
        + right_eye[0]
    ) / 2

    mouth_center_x = (
        mouth_left[0]
        + mouth_right[0]
    ) / 2

    eye_distance = abs(
        right_eye[0]
        - left_eye[0]
    )

    if eye_distance < 1:
        return False

    horizontal_offset = abs(
        eye_center_x
        - mouth_center_x
    )

    return (
        horizontal_offset <
        eye_distance * CONFIG.LOOKING_THRESHOLD
    )


# ==========================================================
# OVERLAY
# ==========================================================

def overlay_smiley(
    frame,
    smiley,
    nose,
    left_eye,
    right_eye
):

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    eye_distance = math.hypot(dx, dy)

    angle = math.degrees(
        math.atan2(dy, dx)
    )

    smiley_h, smiley_w = smiley.shape[:2]

    target_width = int(
        eye_distance * CONFIG.FACE_SCALE
    )

    scale = target_width / smiley_w

    smiley_center = (
        smiley_w / 2,
        smiley_h / 2
    )

    transform = cv2.getRotationMatrix2D(
        smiley_center,
        -angle,
        scale
    )

    transform[0, 2] += (
        nose[0] - smiley_center[0]
    )

    transform[1, 2] += (
        nose[1] - smiley_center[1]
    )

    frame_h, frame_w = frame.shape[:2]

    warped = cv2.warpAffine(
        smiley,
        transform,
        (frame_w, frame_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    alpha = (
        warped[:, :, 3:4]
        .astype(np.float32)
        / 255.0
    )

    blended = (
        frame * (1 - alpha)
        + warped[:, :, :3] * alpha
    )

    return blended.astype(np.uint8)


# ==========================================================
# MAIN
# ==========================================================

def main():

    load_model()

    smiley = cv2.imread(
        SMILEY_PATH,
        cv2.IMREAD_UNCHANGED
    )

    if smiley is None:
        print(
            f"Error: Could not load '{SMILEY_PATH}'"
        )
        return

    if smiley.shape[2] != 4:
        print(
            "Error: smiley.png must contain transparency "
            "(RGBA PNG)."
        )
        return

    backend = (
        cv2.CAP_DSHOW
        if sys.platform.startswith("win")
        else cv2.CAP_ANY
    )

    cap = cv2.VideoCapture(
        0,
        backend
    )

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CONFIG.CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CONFIG.CAMERA_HEIGHT
    )

    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH,
        "",
        (
            CONFIG.CAMERA_WIDTH,
            CONFIG.CAMERA_HEIGHT
        ),
        score_threshold=CONFIG.DETECTION_THRESHOLD
    )

    previous_faces = {}

    print("Running. Press Q to quit.")

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        frame_h, frame_w = frame.shape[:2]

        detector.setInputSize(
            (frame_w, frame_h)
        )

        _, faces = detector.detect(frame)

        if faces is not None:

            for face in faces:

                x, y, w, h = face[:4]

                # Ignore small/distant faces

                if w < CONFIG.MIN_FACE_WIDTH:
                    continue

                # Ignore faces near edges

                margin = CONFIG.EDGE_MARGIN

                if (
                    x < margin
                    or y < margin
                    or x + w > frame_w - margin
                    or y + h > frame_h - margin
                ):
                    continue

                eye_a = (
                    float(face[4]),
                    float(face[5])
                )

                eye_b = (
                    float(face[6]),
                    float(face[7])
                )

                left_eye, right_eye = sorted(
                    [eye_a, eye_b]
                )

                current_face = FaceState(
                    left_eye=left_eye,

                    right_eye=right_eye,

                    nose=(
                        float(face[8]),
                        float(face[9])
                    ),

                    mouth_left=(
                        float(face[10]),
                        float(face[11])
                    ),

                    mouth_right=(
                        float(face[12]),
                        float(face[13])
                    )
                )

                # Simple face tracker

                face_id = (
                    round(x / 50),
                    round(y / 50)
                )

                if face_id in previous_faces:

                    current_face = smooth_face(
                        previous_faces[face_id],
                        current_face
                    )

                previous_faces[face_id] = current_face

                if not looking_at_camera(
                    current_face.left_eye,
                    current_face.right_eye,
                    current_face.mouth_left,
                    current_face.mouth_right
                ):
                    continue

                frame = overlay_smiley(
                    frame,
                    smiley,
                    current_face.nose,
                    current_face.left_eye,
                    current_face.right_eye
                )

        cv2.imshow(
            "Smiley Face Filter",
            frame
        )

        match cv2.waitKey(1) & 0xFF:

            case ord("q"):
                break

        # Keep memory usage bounded

        if len(previous_faces) > 100:
            previous_faces.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()