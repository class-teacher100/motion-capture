"""Real-time pose estimation using YOLO11-pose and OpenCV."""

import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
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

CONF_THRESHOLD = 0.5
KP_CONF_THRESHOLD = 0.5
MODEL_NAME = "yolo11n-pose.pt"
CAMERA_INDEX = 0


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


def select_primary(kps_xy, kps_conf, min_conf: float = 0.4) -> int:
    """Return index of person with the largest valid-keypoint bounding box."""
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


def print_pose_data(person_idx: int, keypoints, confidences) -> None:
    print(f"\n--- Person {person_idx + 1} ---")
    for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
        if conf >= KP_CONF_THRESHOLD:
            print(f"  {KEYPOINT_NAMES[i]:>16}: ({kp[0]:6.1f}, {kp[1]:6.1f})  conf={conf:.2f}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(MODEL_NAME)
    model.to(device)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\nControls: [q] quit  [p] print keypoints  [g] print locomotion features\n")

    mapper = GestureMapper()
    sender = PoseSender()

    prev_time = time.perf_counter()
    print_next = False
    print_gesture = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        result = results[0]

        if result.keypoints is not None and len(result.keypoints) > 0:
            kps_xy = result.keypoints.xy.cpu().numpy()      # (N, 17, 2)
            kps_conf = result.keypoints.conf.cpu().numpy()  # (N, 17)

            for i, (keypoints, confidences) in enumerate(zip(kps_xy, kps_conf)):
                draw_pose(frame, keypoints, confidences)
                if print_next:
                    print_pose_data(i, keypoints, confidences)

            primary = select_primary(kps_xy, kps_conf)
            features = mapper.compute(kps_xy[primary], kps_conf[primary])

            # Normalize keypoints to 0..1 for full-body mirroring Unity-side.
            h, w = frame.shape[:2]
            kp_norm = kps_xy[primary].astype(float).copy()
            kp_norm[:, 0] /= w
            kp_norm[:, 1] /= h

            sender.send(build_packet(kp_norm, kps_conf[primary], features))
            if print_gesture:
                print(f"Features (person {primary + 1}): {features}")
        else:
            sender.send_neutral()
            mapper.reset()

        print_next = False
        print_gesture = False

        now = time.perf_counter()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        n_persons = len(result.keypoints) if result.keypoints is not None else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Persons: {n_persons}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Device: {device.upper()}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        cv2.imshow("Pose Estimation (YOLO11)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            print_next = True
        elif key == ord("g"):
            print_gesture = True

    cap.release()
    cv2.destroyAllWindows()
    sender.close()


if __name__ == "__main__":
    main()
