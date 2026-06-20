"""Fight2D — Motion-capture 2D fighting game (Human vs CPU).

Run: uv run python main.py
Controls (pose):
  Lean body L/R  → move
  Both hands up  → jump
  Thrust one arm toward camera → punch
  Raise one knee → kick
Keyboard: ESC = quit, R = restart (round_end only)
"""

import os
import random
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pygame
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from fighter import Fighter
from gesture import GestureDetector

# ── Window / arena ───────────────────────────────────────────────────────────
WIN_W, WIN_H = 960, 540
GROUND_Y     = 460
ARENA_L, ARENA_R = 60, 900
FPS          = 60

# Camera preview (bottom-right corner)
CAM_PW, CAM_PH = 240, 135

# Fighter spawn positions
PLAYER_SPAWN_X = ARENA_L + 200
CPU_SPAWN_X    = ARENA_R - 200

# Countdown
COUNTDOWN_S  = 3
ROUND_END_S  = 3

# ── MediaPipe ────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
CAM_W, CAM_H = 640, 480

COCO_FROM_MP = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
KP_CONF_MIN  = 0.5

MODEL_VARIANT = "full"
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_SIBLING      = os.path.join(_THIS_DIR, "..", "pose_estimation",
                              f"pose_landmarker_{MODEL_VARIANT}.task")
MODEL_PATH    = os.path.join(_THIS_DIR, f"pose_landmarker_{MODEL_VARIANT}.task")
MODEL_URL     = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    f"pose_landmarker_{MODEL_VARIANT}/float16/latest/"
    f"pose_landmarker_{MODEL_VARIANT}.task"
)

SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]


def ensure_model() -> str:
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    if os.path.exists(_SIBLING):
        return os.path.abspath(_SIBLING)
    print(f"Downloading {MODEL_PATH} …")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def extract_coco(image_lms, world_lms):
    kp_norm  = np.zeros((17, 2))
    kp_conf  = np.zeros(17)
    kp_world = np.zeros((17, 3))
    for ci, mi in enumerate(COCO_FROM_MP):
        lm = image_lms[mi]
        kp_norm[ci]  = (lm.x, lm.y)
        kp_conf[ci]  = lm.visibility
        wl = world_lms[mi]
        kp_world[ci] = (wl.x, wl.y, wl.z)
    return kp_norm, kp_conf, kp_world


def draw_skeleton(frame, kp_px, kp_conf):
    pts = {}
    h, w = frame.shape[:2]
    for i, (kp, c) in enumerate(zip(kp_px, kp_conf)):
        if c >= KP_CONF_MIN:
            x, y = int(kp[0]), int(kp[1])
            if 0 <= x < w and 0 <= y < h:
                pts[i] = (x, y)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    for a, b in SKELETON:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0, 200, 255), 1)


def frame_to_surface(bgr_frame) -> pygame.Surface:
    rgb   = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (CAM_PW, CAM_PH))
    return pygame.surfarray.make_surface(small.transpose(1, 0, 2))


# ── CPU AI ────────────────────────────────────────────────────────────────────
def run_cpu_ai(cpu: Fighter, player: Fighter) -> str | None:
    if cpu.state in ("punch", "kick", "hurt", "dead", "jump"):
        return None
    dist = abs(cpu.x - player.x)
    if cpu.is_grounded() and random.random() < 0.003:
        return "jump"
    if dist <= 80 and cpu.kick_cooldown == 0:
        return "kick_L" if player.x < cpu.x else "kick_R"
    if dist <= 130 and cpu.punch_cooldown == 0:
        return "punch_L" if player.x < cpu.x else "punch_R"
    if dist > 120:
        return "move_L" if player.x > cpu.x else "move_R"
    return None


# ── Main ──────────────────────────────────────────────────────────────────────
def make_fighters():
    player = Fighter(PLAYER_SPAWN_X, GROUND_Y, is_player=True)
    cpu    = Fighter(CPU_SPAWN_X,    GROUND_Y, is_player=False)
    return player, cpu


def main() -> None:
    pygame.init()
    screen  = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Fight2D — Pose Control")
    clock   = pygame.time.Clock()

    font_big = pygame.font.SysFont(None, 90)
    font_med = pygame.font.SysFont(None, 44)
    font_sm  = pygame.font.SysFont(None, 26)

    # Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    # MediaPipe
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=ensure_model()),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker  = mp_vision.PoseLandmarker.create_from_options(options)
    last_ts_ms  = -1
    detector    = GestureDetector()
    cam_surf    = None
    action      = None

    # Game state
    phase        = "countdown"
    phase_timer  = COUNTDOWN_S * FPS
    winner_text  = ""
    player, cpu  = make_fighters()

    running = True
    while running:
        # ── Events ───────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r and phase == "round_end":
                    phase       = "countdown"
                    phase_timer = COUNTDOWN_S * FPS
                    winner_text = ""
                    player, cpu = make_fighters()
                    detector.reset()

        # ── Pose inference ───────────────────────────────────────────────────
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # mirror: lean-right = move right
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms  = max(last_ts_ms + 1, int(time.perf_counter() * 1000))
            last_ts_ms = ts_ms
            result = landmarker.detect_for_video(mp_img, ts_ms)

            if result.pose_landmarks:
                kp_norm, kp_conf, kp_world = extract_coco(
                    result.pose_landmarks[0],
                    result.pose_world_landmarks[0],
                )
                action = detector.update(kp_norm, kp_world, kp_conf)
                h, w   = frame.shape[:2]
                kp_px  = kp_norm.copy()
                kp_px[:, 0] *= w
                kp_px[:, 1] *= h
                draw_skeleton(frame, kp_px, kp_conf)
            else:
                action = None
                detector.reset()

            cam_surf = frame_to_surface(frame)

        # ── Game logic ───────────────────────────────────────────────────────
        if phase == "countdown":
            phase_timer -= 1
            if phase_timer <= 0:
                phase = "fighting"

        elif phase == "fighting":
            player.apply_action(action)
            player.tick(cpu,    ARENA_L, ARENA_R)

            cpu_action = run_cpu_ai(cpu, player)
            cpu.apply_action(cpu_action)
            cpu.tick(player, ARENA_L, ARENA_R)

            if player.is_dead() or cpu.is_dead():
                phase       = "round_end"
                phase_timer = ROUND_END_S * FPS
                winner_text = "YOU WIN!" if cpu.is_dead() else "CPU WINS!"

        elif phase == "round_end":
            phase_timer -= 1

        # ── Draw background ───────────────────────────────────────────────────
        screen.fill((28, 28, 38))
        # Floor
        pygame.draw.rect(screen, (55, 45, 35),
                         pygame.Rect(ARENA_L, GROUND_Y, ARENA_R - ARENA_L,
                                     WIN_H - GROUND_Y))
        pygame.draw.line(screen, (110, 90, 70),
                         (ARENA_L, GROUND_Y), (ARENA_R, GROUND_Y), 3)

        # Fighters
        player.draw(screen)
        cpu.draw(screen)

        # HUD
        player.draw_hp_bar(screen, 20, 12, flip=False)
        cpu.draw_hp_bar(screen, WIN_W - 20, 12, flip=True)
        screen.blit(font_sm.render("YOU", True, (160, 210, 255)), (22, 38))
        screen.blit(font_sm.render("CPU", True, (255, 160, 160)), (WIN_W - 62, 38))

        # Camera preview
        if cam_surf:
            px = WIN_W - CAM_PW - 8
            py = WIN_H - CAM_PH - 8
            screen.blit(cam_surf, (px, py))
            pygame.draw.rect(screen, (180, 180, 180),
                             pygame.Rect(px, py, CAM_PW, CAM_PH), 2)
            # Current gesture label
            if action:
                g_surf = font_sm.render(action, True, (255, 220, 80))
                screen.blit(g_surf, (px, py - 22))

        # Countdown overlay
        if phase == "countdown":
            secs = max(1, (phase_timer + FPS - 1) // FPS)
            txt  = font_big.render(str(secs), True, (255, 220, 60))
            screen.blit(txt, txt.get_rect(center=(WIN_W // 2, WIN_H // 2)))

        # Round-end overlay
        if phase == "round_end":
            col  = (80, 255, 80) if "YOU" in winner_text else (255, 80, 80)
            txt  = font_big.render(winner_text, True, col)
            screen.blit(txt, txt.get_rect(center=(WIN_W // 2, WIN_H // 2 - 40)))
            if phase_timer < (ROUND_END_S - 1) * FPS:
                hint = font_med.render("Press R to play again", True, (200, 200, 200))
                screen.blit(hint, hint.get_rect(center=(WIN_W // 2, WIN_H // 2 + 40)))

        # Controls hint
        hint_text = ("Lean: move  |  Thrust arm forward: punch  |"
                     "  Raise knee: kick  |  Both hands up: jump  |  ESC: quit")
        screen.blit(font_sm.render(hint_text, True, (100, 100, 110)), (8, WIN_H - 22))

        pygame.display.flip()
        clock.tick(FPS)

    cap.release()
    landmarker.close()
    pygame.quit()


if __name__ == "__main__":
    main()
