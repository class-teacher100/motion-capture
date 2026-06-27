"""Converts YOLO11 pose keypoints into Unity locomotion features.

Locomotion is decoupled from the faithful full-body mirroring (which is done
Unity-side from the raw keypoints). This mapper only derives *intentional*
movement controls so the avatar is easy to steer:

    forward : step-in-place cadence  -> walk forward            [0, 1]
    turn    : upper-body orientation -> rotate (yaw) left/right  [-1, 1]
    jump    : both wrists raised above the head                  bool

Also contains FightingGestureMapper for keyboard-driven fighting game control.
"""

import time
from collections import deque

import numpy as np

FRAME_W = 1280.0
FRAME_H = 720.0

# --- Tunable thresholds ---------------------------------------------------
KP_MIN_CONF = 0.40          # minimum keypoint confidence to treat as valid
EMA_ALPHA = 0.35            # exponential moving average weight (lower = smoother)

# forward / step-in-place detection
STEP_WINDOW_S = 1.0         # sliding window for cadence measurement (seconds)
STEP_MIN_AMP = 0.15         # min peak-to-peak foot oscillation (torso-height units)
STEP_HYSTERESIS = 0.03      # ignore tiny crossings as noise (torso-height units)
STEP_FULL_CROSS = 3.0       # zero-crossings/sec that maps to full forward (1.0)
STEP_IDLE_DECAY = 0.80      # forward multiplier per frame when no recent steps

# turn / upper-body orientation
TURN_DEADZONE = 0.08        # shoulder-vs-hip offset ignored near center
TURN_SCALE = 0.35           # offset (torso-height units) that maps to full turn

# jump
JUMP_HOLD_FRAMES = 3        # frames both wrists must stay above nose before jump fires
JUMP_COOLDOWN_S = 0.55      # seconds between jumps (matches ThirdPersonController.JumpTimeout)

# COCO keypoint indices
NOSE = 0
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# --- Fighting game thresholds ---------------------------------------------
FIGHT_LEAN_DEADZONE = 0.15  # shoulder-vs-hip offset / torso_h before move key triggers
FIGHT_CROUCH_ANGLE = 120.0  # both knee joint angles below this (degrees) → crouch
FIGHT_JUMP_HOLD = 4         # frames both wrists must stay above ears
FIGHT_JUMP_COOLDOWN = 0.80  # s

# Punch: XZ-plane distance from wrist to same-side shoulder
PUNCH_EXTEND_SPEED = 0.20   # m/s distance expansion → enter extending phase
PUNCH_RETRACT_SPEED = 0.15  # m/s distance retraction → trigger at reversal
PUNCH_MIN_EXT = 0.10        # m extension (peak - start) for weak punch
PUNCH_MED_EXT = 0.20        # m for medium punch
PUNCH_STRONG_EXT = 0.32     # m for strong punch
PUNCH_COOLDOWN = 0.45       # s between punch events

# Kick: XZ-plane distance from ankle to hip midpoint
KICK_EXTEND_SPEED = 0.20    # m/s
KICK_RETRACT_SPEED = 0.15   # m/s
KICK_MIN_EXT = 0.12         # m extension for weak kick
KICK_MED_EXT = 0.25         # m for medium kick
KICK_STRONG_EXT = 0.40      # m for strong kick
KICK_COOLDOWN = 0.50        # s between kick events


class GestureMapper:
    def __init__(self):
        self._prev_forward = 0.0
        self._prev_turn = 0.0
        self._step_hist = deque()  # (timestamp, signed foot-height difference)
        self._jump_hold_count = 0
        self._jump_cooldown_until = 0.0

    def compute(self, kps_xy_px: np.ndarray, kps_conf: np.ndarray) -> dict:
        """
        kps_xy_px : shape (17, 2), pixel coordinates
        kps_conf  : shape (17,),   confidence per keypoint [0, 1]
        Returns locomotion feature dict (forward / turn / jump / confidence).
        """
        kps = kps_xy_px.astype(float).copy()
        kps[:, 0] /= FRAME_W
        kps[:, 1] /= FRAME_H

        def ok(*indices):
            return all(kps_conf[i] >= KP_MIN_CONF for i in indices)

        # Mean confidence over the keypoints relevant to locomotion
        feat_kps = [NOSE, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST,
                    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]
        used_confs = [kps_conf[i] for i in feat_kps if kps_conf[i] >= KP_MIN_CONF]
        mean_conf = float(np.mean(used_confs)) if used_confs else 0.0

        # Body scale: vertical shoulder->hip distance (stable size proxy).
        torso_h = 0.0
        shoulder_mid_x = hip_mid_x = 0.0
        if ok(L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
            shoulder_mid_y = (kps[L_SHOULDER][1] + kps[R_SHOULDER][1]) / 2.0
            hip_mid_y = (kps[L_HIP][1] + kps[R_HIP][1]) / 2.0
            torso_h = abs(hip_mid_y - shoulder_mid_y)
            shoulder_mid_x = (kps[L_SHOULDER][0] + kps[R_SHOULDER][0]) / 2.0
            hip_mid_x = (kps[L_HIP][0] + kps[R_HIP][0]) / 2.0

        now = time.monotonic()

        # ── forward: step-in-place cadence ───────────────────────────────────
        forward = self._compute_forward(kps, kps_conf, torso_h, now)

        # ── turn: upper-body horizontal offset (shoulders vs hips) ────────────
        turn = 0.0
        if torso_h > 0.01:
            offset = (shoulder_mid_x - hip_mid_x) / torso_h
            if abs(offset) > TURN_DEADZONE:
                sign = 1.0 if offset > 0 else -1.0
                turn = sign * min(1.0, (abs(offset) - TURN_DEADZONE) / TURN_SCALE)

        # EMA smoothing on the analog channels
        forward = EMA_ALPHA * forward + (1.0 - EMA_ALPHA) * self._prev_forward
        turn = EMA_ALPHA * turn + (1.0 - EMA_ALPHA) * self._prev_turn
        self._prev_forward = forward
        self._prev_turn = turn

        # ── jump: both wrists above nose for JUMP_HOLD_FRAMES frames ──────────
        jump = False
        if ok(L_WRIST, R_WRIST, NOSE):
            both_up = kps[L_WRIST][1] < kps[NOSE][1] and kps[R_WRIST][1] < kps[NOSE][1]
            self._jump_hold_count = (self._jump_hold_count + 1) if both_up else 0
            if self._jump_hold_count >= JUMP_HOLD_FRAMES and now >= self._jump_cooldown_until:
                jump = True
                self._jump_cooldown_until = now + JUMP_COOLDOWN_S
                self._jump_hold_count = 0

        return {
            "forward": round(forward, 3),
            "turn": round(turn, 3),
            "jump": bool(jump),
            "confidence": round(mean_conf, 3),
        }

    def _compute_forward(self, kps, kps_conf, torso_h, now) -> float:
        """Detect marching-in-place via alternating foot-height oscillation."""
        if torso_h <= 0.01:
            self._step_hist.clear()
            return self._prev_forward * STEP_IDLE_DECAY

        # Per-leg vertical position: prefer ankle, fall back to knee.
        def foot_y(ankle, knee):
            if kps_conf[ankle] >= KP_MIN_CONF:
                return kps[ankle][1]
            if kps_conf[knee] >= KP_MIN_CONF:
                return kps[knee][1]
            return None

        ly = foot_y(L_ANKLE, L_KNEE)
        ry = foot_y(R_ANKLE, R_KNEE)
        if ly is None or ry is None:
            return self._prev_forward * STEP_IDLE_DECAY

        # Signed, body-scaled difference; oscillates as feet alternate.
        signal = (ry - ly) / torso_h
        self._step_hist.append((now, signal))
        while self._step_hist and now - self._step_hist[0][0] > STEP_WINDOW_S:
            self._step_hist.popleft()

        if len(self._step_hist) < 3:
            return self._prev_forward * STEP_IDLE_DECAY

        vals = [s for _, s in self._step_hist]
        amplitude = max(vals) - min(vals)
        if amplitude < STEP_MIN_AMP:
            return self._prev_forward * STEP_IDLE_DECAY

        # Count detrended sign changes (with hysteresis) = step crossings.
        mean = sum(vals) / len(vals)
        crossings = 0
        prev_sign = 0
        for v in vals:
            d = v - mean
            if abs(d) < STEP_HYSTERESIS:
                continue
            sign = 1 if d > 0 else -1
            if prev_sign != 0 and sign != prev_sign:
                crossings += 1
            prev_sign = sign

        window = self._step_hist[-1][0] - self._step_hist[0][0]
        if window <= 0:
            return self._prev_forward * STEP_IDLE_DECAY
        cross_per_sec = crossings / window
        return max(0.0, min(1.0, cross_per_sec / STEP_FULL_CROSS))

    def reset(self) -> None:
        self._prev_forward = 0.0
        self._prev_turn = 0.0
        self._step_hist.clear()
        self._jump_hold_count = 0


def _new_limb_state() -> dict:
    return {'phase': 'idle', 'start_dist': 0.0, 'peak_dist': 0.0, 'prev_dist': None}


def _update_strike(st: dict, dist: float, dt: float,
                   extend_spd: float, retract_spd: float,
                   min_ext: float, med_ext: float, strong_ext: float,
                   ) -> str | None:
    """Track extending/retracting phase; return 'weak'/'med'/'strong' or None.

    Works on any scalar distance regardless of coordinate space or units —
    used by both FightingGestureMapper (3-D metres) and
    FightingGestureMapper2D (normalised pixel coords).
    """
    result = None
    prev = st['prev_dist']
    if prev is not None:
        speed = (dist - prev) / dt          # positive = extending away from body
        if st['phase'] == 'idle':
            if speed > extend_spd:
                st['phase'] = 'extending'
                st['start_dist'] = prev
                st['peak_dist'] = dist
        elif st['phase'] == 'extending':
            if dist > st['peak_dist']:
                st['peak_dist'] = dist
            if speed < -retract_spd:
                ext = st['peak_dist'] - st['start_dist']
                if ext >= strong_ext:
                    result = 'strong'
                elif ext >= med_ext:
                    result = 'med'
                elif ext >= min_ext:
                    result = 'weak'
                st['phase'] = 'idle'
    st['prev_dist'] = dist
    return result


class FightingGestureMapper:
    """Maps MediaPipe 3-D pose keypoints to fighting-game button states.

    Punch/kick detection uses phase tracking on XZ-plane distance:
      - Wrist to same-side shoulder (XZ) for punches
      - Ankle to hip midpoint (XZ) for kicks
    Trigger fires when the limb reverses from extending to retracting.
    Extension amount (peak_dist - start_dist) determines strength.

    Output keys: move_left, move_right, crouch, jump,
                 weak_punch, med_punch, strong_punch,
                 weak_kick, med_kick, strong_kick
    """

    def __init__(self):
        self._prev_time: float | None = None
        self._punch_st = {L_WRIST: _new_limb_state(), R_WRIST: _new_limb_state()}
        self._kick_st  = {L_ANKLE: _new_limb_state(), R_ANKLE: _new_limb_state()}
        self._jump_hold = 0
        self._jump_cooldown = 0.0
        self._punch_cooldown = 0.0
        self._kick_cooldown = 0.0

    def compute(self, kp_px: np.ndarray, kp_conf: np.ndarray, kp_world: np.ndarray) -> dict:
        now = time.monotonic()
        dt = (now - self._prev_time) if self._prev_time is not None else 1 / 30.0
        self._prev_time = now
        dt = max(dt, 1e-3)

        out = dict(
            move_left=False, move_right=False,
            crouch=False, jump=False,
            weak_punch=False, med_punch=False, strong_punch=False,
            weak_kick=False, med_kick=False, strong_kick=False,
        )

        def ok(*idx):
            return all(kp_conf[i] >= KP_MIN_CONF for i in idx)

        kn = kp_px.astype(float).copy()
        kn[:, 0] /= FRAME_W
        kn[:, 1] /= FRAME_H

        # ── Move left / right: shoulder-vs-hip lateral lean ───────────────────
        if ok(L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
            sh_x = (kn[L_SHOULDER, 0] + kn[R_SHOULDER, 0]) / 2
            hi_x = (kn[L_HIP, 0] + kn[R_HIP, 0]) / 2
            sh_y = (kn[L_SHOULDER, 1] + kn[R_SHOULDER, 1]) / 2
            hi_y = (kn[L_HIP, 1] + kn[R_HIP, 1]) / 2
            torso_h = abs(hi_y - sh_y)
            if torso_h > 0.01:
                offset = (sh_x - hi_x) / torso_h
                if offset > FIGHT_LEAN_DEADZONE:
                    out['move_right'] = True
                elif offset < -FIGHT_LEAN_DEADZONE:
                    out['move_left'] = True

        # ── Crouch: both knee joint angles below threshold ─────────────────────
        if ok(L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE):
            def _angle(a, v, b):
                u = kn[a] - kn[v]
                w = kn[b] - kn[v]
                cos = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-9)
                return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

            if (_angle(L_HIP, L_KNEE, L_ANKLE) < FIGHT_CROUCH_ANGLE and
                    _angle(R_HIP, R_KNEE, R_ANKLE) < FIGHT_CROUCH_ANGLE):
                out['crouch'] = True

        # ── Jump: both wrists above ears for N consecutive frames ─────────────
        if ok(L_WRIST, R_WRIST, L_EAR, R_EAR):
            if kn[L_WRIST, 1] < kn[L_EAR, 1] and kn[R_WRIST, 1] < kn[R_EAR, 1]:
                self._jump_hold += 1
            else:
                self._jump_hold = 0
            if self._jump_hold >= FIGHT_JUMP_HOLD and now >= self._jump_cooldown:
                out['jump'] = True
                self._jump_cooldown = now + FIGHT_JUMP_COOLDOWN
                self._jump_hold = 0

        # ── Punch: XZ-plane distance from wrist to same-side shoulder ─────────
        # Trigger fires when wrist reverses from extending to retracting.
        for wrist_idx, shoulder_idx in ((L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)):
            if ok(wrist_idx, shoulder_idx):
                w = kp_world[wrist_idx]
                s = kp_world[shoulder_idx]
                dist = float(np.hypot(w[0] - s[0], w[2] - s[2]))
                result = _update_strike(
                    self._punch_st[wrist_idx], dist, dt,
                    PUNCH_EXTEND_SPEED, PUNCH_RETRACT_SPEED,
                    PUNCH_MIN_EXT, PUNCH_MED_EXT, PUNCH_STRONG_EXT,
                )
                if result and now >= self._punch_cooldown:
                    out[result + '_punch'] = True
                    self._punch_cooldown = now + PUNCH_COOLDOWN
            else:
                self._punch_st[wrist_idx]['prev_dist'] = None

        # ── Kick: XZ-plane distance from ankle to hip midpoint ────────────────
        if ok(L_HIP, R_HIP):
            hip_x = (kp_world[L_HIP][0] + kp_world[R_HIP][0]) / 2
            hip_z = (kp_world[L_HIP][2] + kp_world[R_HIP][2]) / 2
            for ankle_idx in (L_ANKLE, R_ANKLE):
                if ok(ankle_idx):
                    a = kp_world[ankle_idx]
                    dist = float(np.hypot(a[0] - hip_x, a[2] - hip_z))
                    result = _update_strike(
                        self._kick_st[ankle_idx], dist, dt,
                        KICK_EXTEND_SPEED, KICK_RETRACT_SPEED,
                        KICK_MIN_EXT, KICK_MED_EXT, KICK_STRONG_EXT,
                    )
                    if result and now >= self._kick_cooldown:
                        out[result + '_kick'] = True
                        self._kick_cooldown = now + KICK_COOLDOWN
                else:
                    self._kick_st[ankle_idx]['prev_dist'] = None

        return out

    def reset(self) -> None:
        self._prev_time = None
        for s in self._punch_st.values():
            s.update(_new_limb_state())
        for s in self._kick_st.values():
            s.update(_new_limb_state())
        self._jump_hold = 0
        self._jump_cooldown = 0.0
        self._punch_cooldown = 0.0
        self._kick_cooldown = 0.0


# --- 2-D fighting game thresholds (normalised pixel coords 0-1) -----------
# Distances are frame-size normalised: 0.10 ≈ 128 px at 1280 wide.
# Best results when punch/kick motion is perpendicular to the camera axis
# (side punches, roundhouse kicks); forward punches toward the camera are
# largely invisible in the image plane.
PUNCH_2D_EXTEND_SPEED = 0.30   # norm/s; extension speed to enter extending phase
PUNCH_2D_RETRACT_SPEED = 0.20  # norm/s; retraction speed to trigger
PUNCH_2D_MIN_EXT = 0.07        # norm; extension (peak-start) for weak punch
PUNCH_2D_MED_EXT = 0.13        # norm; medium punch
PUNCH_2D_STRONG_EXT = 0.21     # norm; strong punch
PUNCH_2D_COOLDOWN = 0.45       # s

KICK_2D_EXTEND_SPEED = 0.30
KICK_2D_RETRACT_SPEED = 0.20
KICK_2D_MIN_EXT = 0.08
KICK_2D_MED_EXT = 0.15
KICK_2D_STRONG_EXT = 0.24
KICK_2D_COOLDOWN = 0.50


class FightingGestureMapper2D:
    """2-D variant of FightingGestureMapper — uses only normalised pixel coords.

    Punch: 2-D Euclidean distance from wrist to same-side shoulder.
    Kick : 2-D Euclidean distance from ankle to hip midpoint.
    Move/crouch/jump: identical to the 3-D version (already pixel-based).

    Takes kp_px (17, 2) pixel coords and kp_conf (17,) — no kp_world needed.
    Compatible with any 2-D pose estimator that outputs COCO-17 keypoints
    (YOLO-pose, MediaPipe 2-D, etc.).
    """

    def __init__(self):
        self._prev_time: float | None = None
        self._punch_st = {L_WRIST: _new_limb_state(), R_WRIST: _new_limb_state()}
        self._kick_st  = {L_ANKLE: _new_limb_state(), R_ANKLE: _new_limb_state()}
        self._jump_hold = 0
        self._jump_cooldown = 0.0
        self._punch_cooldown = 0.0
        self._kick_cooldown = 0.0

    def compute(self, kp_px: np.ndarray, kp_conf: np.ndarray) -> dict:
        now = time.monotonic()
        dt = (now - self._prev_time) if self._prev_time is not None else 1 / 30.0
        self._prev_time = now
        dt = max(dt, 1e-3)

        out = dict(
            move_left=False, move_right=False,
            crouch=False, jump=False,
            weak_punch=False, med_punch=False, strong_punch=False,
            weak_kick=False, med_kick=False, strong_kick=False,
        )

        def ok(*idx):
            return all(kp_conf[i] >= KP_MIN_CONF for i in idx)

        kn = kp_px.astype(float).copy()
        kn[:, 0] /= FRAME_W
        kn[:, 1] /= FRAME_H

        # ── Move left / right ─────────────────────────────────────────────────
        if ok(L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
            sh_x = (kn[L_SHOULDER, 0] + kn[R_SHOULDER, 0]) / 2
            hi_x = (kn[L_HIP, 0] + kn[R_HIP, 0]) / 2
            sh_y = (kn[L_SHOULDER, 1] + kn[R_SHOULDER, 1]) / 2
            hi_y = (kn[L_HIP, 1] + kn[R_HIP, 1]) / 2
            torso_h = abs(hi_y - sh_y)
            if torso_h > 0.01:
                offset = (sh_x - hi_x) / torso_h
                if offset > FIGHT_LEAN_DEADZONE:
                    out['move_right'] = True
                elif offset < -FIGHT_LEAN_DEADZONE:
                    out['move_left'] = True

        # ── Crouch ────────────────────────────────────────────────────────────
        if ok(L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE):
            def _angle(a, v, b):
                u = kn[a] - kn[v]
                w = kn[b] - kn[v]
                cos = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-9)
                return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
            if (_angle(L_HIP, L_KNEE, L_ANKLE) < FIGHT_CROUCH_ANGLE and
                    _angle(R_HIP, R_KNEE, R_ANKLE) < FIGHT_CROUCH_ANGLE):
                out['crouch'] = True

        # ── Jump ──────────────────────────────────────────────────────────────
        if ok(L_WRIST, R_WRIST, L_EAR, R_EAR):
            if kn[L_WRIST, 1] < kn[L_EAR, 1] and kn[R_WRIST, 1] < kn[R_EAR, 1]:
                self._jump_hold += 1
            else:
                self._jump_hold = 0
            if self._jump_hold >= FIGHT_JUMP_HOLD and now >= self._jump_cooldown:
                out['jump'] = True
                self._jump_cooldown = now + FIGHT_JUMP_COOLDOWN
                self._jump_hold = 0

        # ── Punch: 2-D distance from wrist to same-side shoulder ──────────────
        for wrist_idx, shoulder_idx in ((L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)):
            if ok(wrist_idx, shoulder_idx):
                dist = float(np.linalg.norm(kn[wrist_idx] - kn[shoulder_idx]))
                result = _update_strike(
                    self._punch_st[wrist_idx], dist, dt,
                    PUNCH_2D_EXTEND_SPEED, PUNCH_2D_RETRACT_SPEED,
                    PUNCH_2D_MIN_EXT, PUNCH_2D_MED_EXT, PUNCH_2D_STRONG_EXT,
                )
                if result and now >= self._punch_cooldown:
                    out[result + '_punch'] = True
                    self._punch_cooldown = now + PUNCH_2D_COOLDOWN
            else:
                self._punch_st[wrist_idx]['prev_dist'] = None

        # ── Kick: 2-D distance from ankle to hip midpoint ─────────────────────
        if ok(L_HIP, R_HIP):
            hip = (kn[L_HIP] + kn[R_HIP]) / 2
            for ankle_idx in (L_ANKLE, R_ANKLE):
                if ok(ankle_idx):
                    dist = float(np.linalg.norm(kn[ankle_idx] - hip))
                    result = _update_strike(
                        self._kick_st[ankle_idx], dist, dt,
                        KICK_2D_EXTEND_SPEED, KICK_2D_RETRACT_SPEED,
                        KICK_2D_MIN_EXT, KICK_2D_MED_EXT, KICK_2D_STRONG_EXT,
                    )
                    if result and now >= self._kick_cooldown:
                        out[result + '_kick'] = True
                        self._kick_cooldown = now + KICK_2D_COOLDOWN
                else:
                    self._kick_st[ankle_idx]['prev_dist'] = None

        return out

    def reset(self) -> None:
        self._prev_time = None
        for s in self._punch_st.values():
            s.update(_new_limb_state())
        for s in self._kick_st.values():
            s.update(_new_limb_state())
        self._jump_hold = 0
        self._jump_cooldown = 0.0
        self._punch_cooldown = 0.0
        self._kick_cooldown = 0.0
