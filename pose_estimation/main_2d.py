"""2-D pose estimation with YOLO11-pose → fighting-game keyboard control.

Requires ultralytics (and torch).  Install with:
    uv add ultralytics   # or: pip install ultralytics

Run:
    uv run python main_2d.py

Controls: [q] quit  [p] print keypoints  [g] print fight state  [space] pause
"""

import time

import cv2
import numpy as np

try:
    import torch
    from ultralytics import YOLO
except ImportError as e:
    raise SystemExit(
        "ultralytics is not installed.\n"
        "Install it with:  uv add ultralytics  (or  pip install ultralytics)"
    ) from e

from gesture_mapper import FightingGestureMapper2D
from fighting_keyboard import FightingKeyboardController

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

CONF_THRESHOLD = 0.5
KP_CONF_THRESHOLD = 0.5
MODEL_NAME = "yolo11n-pose.pt"
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 1280, 720

_FIGHT_LABEL = {
    'move_left': 'A', 'move_right': 'D', 'crouch': 'S', 'jump': 'W',
    'weak_punch': 'I', 'med_punch': 'O', 'strong_punch': 'P',
    'weak_kick': 'J', 'med_kick': 'K', 'strong_kick': 'L',
}


def draw_pose(frame: cv2.Mat, keypoints, confidences) -> None:
    h, w = frame.shape[:2]
    points = {}
    for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
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


def draw_fight_state(frame, state: dict, paused: bool) -> None:
    active = [v for k, v in _FIGHT_LABEL.items() if state.get(k)]
    text = ' '.join(active) if active else '-'
    cv2.putText(frame, f"Keys: {text}", (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
    label = "FIGHTING 2D [PAUSED]" if paused else "FIGHTING 2D"
    color = (100, 100, 100) if paused else (200, 200, 200)
    cv2.putText(frame, label, (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)


def select_primary(kps_xy, kps_conf, min_conf: float = 0.4) -> int:
    """Return index of the person with the largest valid-keypoint bounding box."""
    best, best_area = 0, -1.0
    for i, (kps, confs) in enumerate(zip(kps_xy, kps_conf)):
        valid = kps[confs >= min_conf]
        if len(valid) < 2:
            continue
        area = float(
            (valid[:, 0].max() - valid[:, 0].min()) *
            (valid[:, 1].max() - valid[:, 1].min())
        )
        if area > best_area:
            best_area, best = area, i
    return best


def print_pose_data(keypoints, confidences) -> None:
    print("\n--- Pose (2D) ---")
    for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
        if conf >= KP_CONF_THRESHOLD:
            print(f"  {KEYPOINT_NAMES[i]:>16}: ({kp[0]:6.1f}, {kp[1]:6.1f})  conf={conf:.2f}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = YOLO(MODEL_NAME)
    model.to(device)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    fight_mapper = FightingGestureMapper2D()
    fight_kb = FightingKeyboardController()
    fight_paused = False

    print("FIGHTING MODE (2D/YOLO) — keyboard output active.")
    print("Controls: [q] quit  [p] print keypoints  [g] print fight state  [space] pause\n")

    prev_time = time.perf_counter()
    print_next = False
    print_gesture = False
    fight_state: dict = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            result = results[0]

            detected = result.keypoints is not None and len(result.keypoints) > 0

            if detected:
                kps_xy = result.keypoints.xy.cpu().numpy()      # (N, 17, 2)
                kps_conf = result.keypoints.conf.cpu().numpy()  # (N, 17)

                primary = select_primary(kps_xy, kps_conf)
                kp_px = kps_xy[primary]
                kp_conf = kps_conf[primary]

                draw_pose(frame, kp_px, kp_conf)
                if print_next:
                    print_pose_data(kp_px, kp_conf)

                fight_state = fight_mapper.compute(kp_px, kp_conf)
                if not fight_paused:
                    fight_kb.update(fight_state)
                if print_gesture:
                    print(f"Fight state: {fight_state}")
            else:
                fight_mapper.reset()
                fight_kb.release_all()
                fight_state = {}

            draw_fight_state(frame, fight_state, fight_paused)

            print_next = False
            print_gesture = False

            now = time.perf_counter()
            fps = 1.0 / (now - prev_time)
            prev_time = now

            n_persons = len(result.keypoints) if detected else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"Persons: {n_persons}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow("Pose Estimation 2D (YOLO11) — Fighting", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                print_next = True
            elif key == ord("g"):
                print_gesture = True
            elif key == ord(" "):
                fight_paused = not fight_paused
                fight_kb.release_all()
                print(f"Fighting keyboard {'PAUSED' if fight_paused else 'RESUMED'}")
    finally:
        fight_kb.release_all()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
