"""UDP sender: streams pose packets to Unity on localhost.

Packet schema v3 carries both layers in a single datagram:
  - kp      : flat [x0,y0, x1,y1, ... x16,y16] normalized 0..1 image coords (34 floats)
  - kp_conf : 17 per-keypoint confidences (MediaPipe visibility)
  - kp3d    : flat [x0,y0,z0, ... x16,y16,z16] 3D world landmarks, hip-centered
              metres (51 floats). Drives true 3D bone orientation in Unity.
  - forward / turn / jump / confidence : intentional locomotion features

All arrays are kept flat (not [[x,y],...]) so Unity's JsonUtility can
deserialize them into float[] (JsonUtility does not support jagged arrays).

v3 adds kp3d on top of v2. The 2D kp/kp_conf fields are unchanged so the
locomotion layer and any 2D fallback keep working.
"""

import json
import math
import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

N_KEYPOINTS = 17

NEUTRAL_PACKET = {
    "v": 3,
    "kp": [0.0] * (N_KEYPOINTS * 2),
    "kp_conf": [0.0] * N_KEYPOINTS,
    "kp3d": [0.0] * (N_KEYPOINTS * 3),
    "forward": 0.0,
    "turn": 0.0,
    "jump": False,
    "confidence": 0.0,
}


class PoseSender:
    def __init__(self, ip: str = UDP_IP, port: int = UDP_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (ip, port)

    def send(self, packet: dict) -> None:
        data = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        self._sock.sendto(data, self._addr)

    def send_neutral(self) -> None:
        self.send(NEUTRAL_PACKET)

    def close(self) -> None:
        self._sock.close()


def _finite(v, default: float = 0.0) -> float:
    """Coerce NaN/Inf to a default so we never emit non-standard JSON tokens.

    json.dumps writes NaN/Infinity literally (invalid JSON); Unity's JsonUtility
    then parses them into float.NaN, which corrupts bone rotations and can wipe
    the whole rig. Sanitizing here keeps the wire format clean."""
    v = float(v)
    return v if math.isfinite(v) else default


def build_packet(kp_norm, kp_conf, kp_world, features: dict) -> dict:
    """Combine 2D + 3D keypoints with locomotion features into a v3 packet.

    kp_norm  : (17, 2) array-like, normalized 0..1 image coords (x/width, y/height)
    kp_conf  : (17,)   array-like confidences (MediaPipe visibility)
    kp_world : (17, 3) array-like, 3D world landmarks in metres (hip-centered)
    features : dict from GestureMapper.compute (forward/turn/jump/confidence)
    """
    kp_flat = []
    for x, y in kp_norm:
        kp_flat.append(round(_finite(x), 4))
        kp_flat.append(round(_finite(y), 4))

    kp3d_flat = []
    for x, y, z in kp_world:
        kp3d_flat.append(round(_finite(x), 4))
        kp3d_flat.append(round(_finite(y), 4))
        kp3d_flat.append(round(_finite(z), 4))

    return {
        "v": 3,
        "kp": kp_flat,
        "kp_conf": [round(_finite(c), 3) for c in kp_conf],
        "kp3d": kp3d_flat,
        "forward": _finite(features["forward"]),
        "turn": _finite(features["turn"]),
        "jump": bool(features["jump"]),
        "confidence": _finite(features["confidence"]),
    }
