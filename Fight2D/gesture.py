"""Gesture detection for Fight2D.

Converts MediaPipe COCO keypoints (2D + 3D world) into a single action string.
Priority: jump > kick > punch > move > None.

3D world coordinate convention (MediaPipe hip-centered, metric):
  x = right, y = up (positive), z = toward camera (negative = closer)
"""

import numpy as np

# COCO keypoint indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW,    R_ELBOW    = 7, 8
L_WRIST,    R_WRIST    = 9, 10
L_HIP,      R_HIP      = 11, 12
L_KNEE,     R_KNEE     = 13, 14

CONF_MIN = 0.40

# Move
LEAN_DEADZONE = 0.06   # normalized units below which lean is ignored
LEAN_SCALE    = 0.20   # lean at which output saturates

# Jump
JUMP_HOLD_FRAMES = 3
JUMP_COOLDOWN    = 45  # frames

# Punch (3D world, metres)
PUNCH_WRIST_Z   = -0.25  # wrist_z - shoulder_z must be below this
PUNCH_ELBOW_Z   = -0.10  # elbow_z - shoulder_z must be below this (arm extended)
PUNCH_COOLDOWN  = 25     # frames

# Kick (3D world, metres)
KICK_KNEE_Y    = 0.20    # single knee world_Y above mean-hip world_Y
KICK_REST_MAX  = 0.10    # opposite knee must stay within this of rest to avoid false positives
KICK_HOLD_FRAMES = 4
KICK_COOLDOWN    = 40    # frames


class GestureDetector:
    def __init__(self) -> None:
        self._jump_hold   = 0
        self._jump_cd     = 0
        self._kick_hold   = 0
        self._kick_cd     = 0
        self._kick_side   = None   # "L" or "R" while building up hold
        self._punch_cd    = 0

    # ------------------------------------------------------------------ public
    def update(
        self,
        kp_norm:  np.ndarray,   # (17, 2) normalized image coords, x left→right
        kp_world: np.ndarray,   # (17, 3) metric, hip-centered
        kp_conf:  np.ndarray,   # (17,)   visibility
    ) -> str | None:
        self._tick_cooldowns()

        action = (
            self._check_jump(kp_norm, kp_conf)
            or self._check_kick(kp_world, kp_conf)
            or self._check_punch(kp_world, kp_conf)
            or self._check_move(kp_norm, kp_conf)
        )
        return action

    def reset(self) -> None:
        self._jump_hold = 0
        self._kick_hold = 0
        self._kick_side = None
        self._punch_cd  = max(0, self._punch_cd)

    # ----------------------------------------------------------------- private
    def _tick_cooldowns(self) -> None:
        if self._jump_cd  > 0: self._jump_cd  -= 1
        if self._kick_cd  > 0: self._kick_cd  -= 1
        if self._punch_cd > 0: self._punch_cd -= 1

    def _ok(self, kp_conf, *indices) -> bool:
        return all(kp_conf[i] >= CONF_MIN for i in indices)

    # ── jump ──────────────────────────────────────────────────────────────────
    def _check_jump(self, kp_norm, kp_conf) -> str | None:
        if not self._ok(kp_conf, NOSE, L_WRIST, R_WRIST):
            self._jump_hold = 0
            return None
        both_up = (kp_norm[L_WRIST, 1] < kp_norm[NOSE, 1] and
                   kp_norm[R_WRIST, 1] < kp_norm[NOSE, 1])
        if both_up:
            self._jump_hold += 1
        else:
            self._jump_hold = 0
        if self._jump_hold >= JUMP_HOLD_FRAMES and self._jump_cd == 0:
            self._jump_hold = 0
            self._jump_cd   = JUMP_COOLDOWN
            return "jump"
        return None

    # ── kick ──────────────────────────────────────────────────────────────────
    def _check_kick(self, kp_world, kp_conf) -> str | None:
        if self._kick_cd > 0:
            return None
        if not self._ok(kp_conf, L_HIP, R_HIP, L_KNEE, R_KNEE):
            self._kick_hold = 0
            self._kick_side = None
            return None

        hip_y = (kp_world[L_HIP, 1] + kp_world[R_HIP, 1]) / 2.0
        l_knee_rise = kp_world[L_KNEE, 1] - hip_y
        r_knee_rise = kp_world[R_KNEE, 1] - hip_y

        kicked_side = None
        if l_knee_rise > KICK_KNEE_Y and r_knee_rise < KICK_REST_MAX:
            kicked_side = "L"
        elif r_knee_rise > KICK_KNEE_Y and l_knee_rise < KICK_REST_MAX:
            kicked_side = "R"

        if kicked_side and kicked_side == self._kick_side:
            self._kick_hold += 1
        else:
            self._kick_hold = 1 if kicked_side else 0
            self._kick_side = kicked_side

        if self._kick_hold >= KICK_HOLD_FRAMES and kicked_side:
            self._kick_hold = 0
            self._kick_side = None
            self._kick_cd   = KICK_COOLDOWN
            return f"kick_{kicked_side}"
        return None

    # ── punch ─────────────────────────────────────────────────────────────────
    def _check_punch(self, kp_world, kp_conf) -> str | None:
        if self._punch_cd > 0:
            return None
        for side, wrist_i, elbow_i, shoulder_i in (
            ("L", L_WRIST, L_ELBOW, L_SHOULDER),
            ("R", R_WRIST, R_ELBOW, R_SHOULDER),
        ):
            if not self._ok(kp_conf, wrist_i, elbow_i, shoulder_i):
                continue
            wrist_dz  = kp_world[wrist_i,  2] - kp_world[shoulder_i, 2]
            elbow_dz  = kp_world[elbow_i,  2] - kp_world[shoulder_i, 2]
            if wrist_dz < PUNCH_WRIST_Z and elbow_dz < PUNCH_ELBOW_Z:
                self._punch_cd = PUNCH_COOLDOWN
                return f"punch_{side}"
        return None

    # ── move ──────────────────────────────────────────────────────────────────
    def _check_move(self, kp_norm, kp_conf) -> str | None:
        if not self._ok(kp_conf, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
            return None
        sh_x  = (kp_norm[L_SHOULDER, 0] + kp_norm[R_SHOULDER, 0]) / 2.0
        hip_x = (kp_norm[L_HIP, 0]      + kp_norm[R_HIP, 0])      / 2.0
        lean  = sh_x - hip_x
        if lean > LEAN_DEADZONE:
            return "move_R"
        if lean < -LEAN_DEADZONE:
            return "move_L"
        return None
