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

PUNCH_MIN_SPEED = 0.35      # m/s; any punch (3-D world velocity)
PUNCH_MED_SPEED = 0.70      # m/s; medium punch threshold
PUNCH_STRONG_SPEED = 1.10   # m/s; strong punch threshold
PUNCH_COOLDOWN = 0.45       # s between punch events

KICK_MIN_SPEED = 0.50       # m/s; any kick (ankle 3-D world velocity)
KICK_MED_SPEED = 0.90       # m/s
KICK_STRONG_SPEED = 1.40    # m/s
KICK_COOLDOWN = 0.50        # s between kick events
KICK_KNEE_RISE = 0.08       # knee must rise by this fraction of frame height above EMA


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


class FightingGestureMapper:
    """Maps MediaPipe 3-D pose keypoints to fighting-game button states.

    Returns a dict with bool values for:
        move_left / move_right  — body lean (held)
        crouch                  — knee-angle squat (held)
        jump                    — both wrists above ears (one-shot)
        weak_punch / med_punch / strong_punch  — wrist speed tiers (one-shot)
        weak_kick  / med_kick  / strong_kick   — ankle speed tiers (one-shot)

    Requires kp_world (17, 3) metric 3-D coords from extract_coco().
    """

    def __init__(self):
        self._prev_time: float | None = None
        self._prev_wrist = {L_WRIST: None, R_WRIST: None}
        self._prev_ankle = {L_ANKLE: None, R_ANKLE: None}
        self._jump_hold = 0
        self._jump_cooldown = 0.0
        self._punch_cooldown = 0.0
        self._kick_cooldown = 0.0
        self._knee_y_ema: np.ndarray | None = None  # (left_knee_y, right_knee_y) in norm coords

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

        # ── Punch: max wrist 3-D world speed ──────────────────────────────────
        if ok(L_WRIST, R_WRIST):
            speeds = []
            for idx in (L_WRIST, R_WRIST):
                prev = self._prev_wrist[idx]
                cur = kp_world[idx].copy()
                if prev is not None:
                    speeds.append(float(np.linalg.norm((cur - prev) / dt)))
                self._prev_wrist[idx] = cur
            if speeds and now >= self._punch_cooldown:
                spd = max(speeds)
                if spd >= PUNCH_MIN_SPEED:
                    if spd >= PUNCH_STRONG_SPEED:
                        out['strong_punch'] = True
                    elif spd >= PUNCH_MED_SPEED:
                        out['med_punch'] = True
                    else:
                        out['weak_punch'] = True
                    self._punch_cooldown = now + PUNCH_COOLDOWN
        else:
            self._prev_wrist[L_WRIST] = None
            self._prev_wrist[R_WRIST] = None

        # ── Kick: ankle 3-D world speed with knee-raised guard ────────────────
        if ok(L_KNEE, R_KNEE):
            kv = np.array([kn[L_KNEE, 1], kn[R_KNEE, 1]])
            if self._knee_y_ema is None:
                self._knee_y_ema = kv.copy()
            else:
                self._knee_y_ema = 0.05 * kv + 0.95 * self._knee_y_ema
            knee_raised = (kn[L_KNEE, 1] < self._knee_y_ema[0] - KICK_KNEE_RISE or
                           kn[R_KNEE, 1] < self._knee_y_ema[1] - KICK_KNEE_RISE)
        else:
            knee_raised = False

        if ok(L_ANKLE, R_ANKLE):
            speeds = []
            for idx in (L_ANKLE, R_ANKLE):
                prev = self._prev_ankle[idx]
                cur = kp_world[idx].copy()
                if prev is not None and knee_raised:
                    speeds.append(float(np.linalg.norm((cur - prev) / dt)))
                self._prev_ankle[idx] = cur
            if speeds and now >= self._kick_cooldown:
                spd = max(speeds)
                if spd >= KICK_MIN_SPEED:
                    if spd >= KICK_STRONG_SPEED:
                        out['strong_kick'] = True
                    elif spd >= KICK_MED_SPEED:
                        out['med_kick'] = True
                    else:
                        out['weak_kick'] = True
                    self._kick_cooldown = now + KICK_COOLDOWN
        else:
            self._prev_ankle[L_ANKLE] = None
            self._prev_ankle[R_ANKLE] = None

        return out

    def reset(self) -> None:
        self._prev_time = None
        self._prev_wrist = {L_WRIST: None, R_WRIST: None}
        self._prev_ankle = {L_ANKLE: None, R_ANKLE: None}
        self._jump_hold = 0
        self._jump_cooldown = 0.0
        self._punch_cooldown = 0.0
        self._kick_cooldown = 0.0
        self._knee_y_ema = None
