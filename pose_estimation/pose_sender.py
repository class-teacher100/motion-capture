"""UDP sender: streams pose packets to Unity on localhost.

Packet schema v2 carries both layers in a single datagram:
  - kp      : flat [x0,y0, x1,y1, ... x16,y16] normalized 0..1 (34 floats)
  - kp_conf : 17 per-keypoint confidences
  - forward / turn / jump / confidence : intentional locomotion features

kp is kept flat (not [[x,y],...]) so Unity's JsonUtility can deserialize it
into a float[] (JsonUtility does not support jagged arrays).
"""

import json
import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

N_KEYPOINTS = 17

NEUTRAL_PACKET = {
    "v": 2,
    "kp": [0.0] * (N_KEYPOINTS * 2),
    "kp_conf": [0.0] * N_KEYPOINTS,
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


def build_packet(kp_norm, kp_conf, features: dict) -> dict:
    """Combine normalized keypoints with locomotion features into a v2 packet.

    kp_norm  : (17, 2) array-like, normalized 0..1 (x/width, y/height)
    kp_conf  : (17,)   array-like confidences
    features : dict from GestureMapper.compute (forward/turn/jump/confidence)
    """
    kp_flat = []
    for x, y in kp_norm:
        kp_flat.append(round(float(x), 4))
        kp_flat.append(round(float(y), 4))
    return {
        "v": 2,
        "kp": kp_flat,
        "kp_conf": [round(float(c), 3) for c in kp_conf],
        "forward": features["forward"],
        "turn": features["turn"],
        "jump": features["jump"],
        "confidence": features["confidence"],
    }
