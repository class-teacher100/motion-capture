"""Real-time 3D pose estimation using the MediaPipe Tasks PoseLandmarker.

PoseLandmarker outputs 33 landmarks with per-landmark depth (z), plus a
hip-centered metric "world" landmark set. We extract the 17 COCO-standard
keypoints (a subset of the 33) so the rest of the pipeline — locomotion
features and the Unity 17-keypoint contract — stays unchanged, and we add a
true 3D channel (kp3d) for full-body mirroring with depth.

Uses the Tasks API (mp.tasks.vision) rather than the legacy mp.solutions,
which is no longer shipped in recent mediapipe builds (e.g. on Python 3.14).
The .task model is downloaded on first run.
"""

import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from gesture_mapper import GestureMapper
from pose_sender import PoseSender, build_packet

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # face
    (5, 6),                                      # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (5, 11), (6, 12), (11, 12),                 # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]

# COCO keypoint index -> MediaPipe Pose landmark index.
# MediaPipe's 33 landmarks include every COCO joint, so this is a pure remap.
COCO_FROM_MP = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

KP_CONF_THRESHOLD = 0.5     # MediaPipe "visibility" treated as keypoint confidence
MIN_DET_CONF = 0.5
MIN_PRESENCE_CONF = 0.5
MIN_TRACK_CONF = 0.5
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 1280, 720

# Model variant: "lite" (fastest) | "full" (balanced) | "heavy" (most accurate).
MODEL_VARIANT = "full"
MODEL_PATH = f"pose_landmarker_{MODEL_VARIANT}.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    f"pose_landmarker_{MODEL_VARIANT}/float16/latest/"
    f"pose_landmarker_{MODEL_VARIANT}.task"
)


def ensure_model() -> str:
    """Download the PoseLandmarker .task model on first run; return its path."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")
    return MODEL_PATH


def extract_coco(image_landmarks, world_landmarks):
    """Remap MediaPipe's 33 landmarks to the 17 COCO keypoints.

    Returns:
        kp_norm   (17, 2) normalized 0..1 image coords
        kp_conf   (17,)   visibility per keypoint
        kp_world  (17, 3) metric, hip-centered 3D coords
    """
    kp_norm = np.zeros((17, 2), dtype=float)
    kp_conf = np.zeros(17, dtype=float)
    kp_world = np.zeros((17, 3), dtype=float)

    for coco_i, mp_i in enumerate(COCO_FROM_MP):
        lm = image_landmarks[mp_i]
        kp_norm[coco_i] = (lm.x, lm.y)
        kp_conf[coco_i] = lm.visibility
        wl = world_landmarks[mp_i]
        kp_world[coco_i] = (wl.x, wl.y, wl.z)

    return kp_norm, kp_conf, kp_world


def draw_pose(frame: cv2.Mat, kp_px, confidences) -> None:
    h, w = frame.shape[:2]
    points = {}
    for i, (kp, conf) in enumerate(zip(kp_px, confidences)):
        if conf < KP_CONF_THRESHOLD:
            continue
        x, y = int(kp[0]), int(kp[1])
        if 0 <= x < w and 0 <= y < h:
            points[i] = (x, y)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)

    for a, b in SKELETON:
        if a in points and b in points:
            cv2.line(frame, points[a], points[b], (0, 200, 255), 2)


def print_pose_data(kp_px, kp_world, confidences) -> None:
    print("\n--- Pose ---")
    for i, (kp, w3, conf) in enumerate(zip(kp_px, kp_world, confidences)):
        if conf >= KP_CONF_THRESHOLD:
            print(f"  {KEYPOINT_NAMES[i]:>16}: px=({kp[0]:6.1f}, {kp[1]:6.1f})  "
                  f"3d=({w3[0]:+.2f}, {w3[1]:+.2f}, {w3[2]:+.2f})  vis={conf:.2f}")


def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    model_path = ensure_model()
    print(f"MediaPipe PoseLandmarker ({MODEL_VARIANT}) ready.")
    print("\nControls: [q] quit  [p] print keypoints  [g] print locomotion features\n")

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=MIN_DET_CONF,
        min_pose_presence_confidence=MIN_PRESENCE_CONF,
        min_tracking_confidence=MIN_TRACK_CONF,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    mapper = GestureMapper()
    sender = PoseSender()

    prev_time = time.perf_counter()
    last_ts_ms = -1
    print_next = False
    print_gesture = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe expects an RGB mp.Image; OpenCV gives BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode needs a strictly increasing timestamp in ms.
        ts_ms = max(last_ts_ms + 1, int(time.perf_counter() * 1000))
        last_ts_ms = ts_ms
        result = landmarker.detect_for_video(mp_image, ts_ms)

        h, w = frame.shape[:2]
        detected = bool(result.pose_landmarks) and bool(result.pose_world_landmarks)

        if detected:
            kp_norm, kp_conf, kp_world = extract_coco(
                result.pose_landmarks[0],
                result.pose_world_landmarks[0],
            )

            kp_px = kp_norm.copy()
            kp_px[:, 0] *= w
            kp_px[:, 1] *= h

            draw_pose(frame, kp_px, kp_conf)
            if print_next:
                print_pose_data(kp_px, kp_world, kp_conf)

            features = mapper.compute(kp_px, kp_conf)
            sender.send(build_packet(kp_norm, kp_conf, kp_world, features))
            if print_gesture:
                print(f"Features: {features}")
        else:
            sender.send_neutral()
            mapper.reset()

        print_next = False
        print_gesture = False

        now = time.perf_counter()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Detected: {'yes' if detected else 'no'}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"MediaPipe 3D ({MODEL_VARIANT})", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        cv2.imshow("Pose Estimation (MediaPipe 3D)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            print_next = True
        elif key == ord("g"):
            print_gesture = True

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    sender.close()


if __name__ == "__main__":
    main()
