"""Fighter entity: state machine, physics, hit detection, drawing."""

from __future__ import annotations
import math
import pygame

# ── Physics ───────────────────────────────────────────────────────────────────
GRAVITY    = 0.65
WALK_SPEED = 5
JUMP_VY    = -15

# ── Attack parameters ─────────────────────────────────────────────────────────
PUNCH_DURATION = 18
PUNCH_ACTIVE   = 8
PUNCH_REACH    = 80
PUNCH_HEIGHT   = 50
PUNCH_DAMAGE   = 12

KICK_DURATION  = 28
KICK_ACTIVE    = 10
KICK_REACH     = 100
KICK_HEIGHT    = 60
KICK_DAMAGE    = 20

HURT_DURATION  = 12
PUNCH_COOLDOWN = 25
KICK_COOLDOWN  = 40

# ── Hitbox geometry (unchanged) ───────────────────────────────────────────────
BODY_W, BODY_H = 50, 100
HEAD_R         = 18
HP_BAR_W, HP_BAR_H = 280, 22

# ── Colors ────────────────────────────────────────────────────────────────────
_C_PLAYER = (70,  150, 255)
_C_CPU    = (235,  70,  70)
_C_HURT   = (255, 230,  50)
_C_DEAD   = (65,   65,  65)

# ── Pose data ─────────────────────────────────────────────────────────────────
# Origin = (self.x, self.y) = feet on ground.  y negative = upward.
# All poses use "facing-right" convention. x is multiplied by self.facing on draw.
# Right limbs (r_*) = the "front" / attacking side.

def _p(**kw): return kw

_IDLE = _p(
    head=(0,-112), neck=(0,-93), hip_c=(0,-55),
    l_shoulder=(-14,-89), r_shoulder=(14,-89),
    l_elbow=(-22,-71),    r_elbow=(26,-71),
    l_wrist=(-20,-54),    r_wrist=(22,-53),
    l_hip=(-10,-55),      r_hip=(10,-55),
    l_knee=(-13,-27),     r_knee=(15,-27),
    l_ankle=(-12,0),      r_ankle=(14,0),
)
_WALK_A = _p(     # right leg forward, left arm forward
    head=(0,-112), neck=(0,-93), hip_c=(0,-55),
    l_shoulder=(-14,-89), r_shoulder=(14,-89),
    l_elbow=(-30,-75),    r_elbow=(20,-65),
    l_wrist=(-34,-59),    r_wrist=(17,-49),
    l_hip=(-10,-55),      r_hip=(10,-55),
    l_knee=(-16,-23),     r_knee=(3,-32),
    l_ankle=(-19,0),      r_ankle=(20,0),
)
_WALK_B = _p(     # left leg forward, right arm forward
    head=(0,-112), neck=(0,-93), hip_c=(0,-55),
    l_shoulder=(-14,-89), r_shoulder=(14,-89),
    l_elbow=(-20,-65),    r_elbow=(30,-75),
    l_wrist=(-17,-49),    r_wrist=(34,-59),
    l_hip=(-10,-55),      r_hip=(10,-55),
    l_knee=(3,-32),       r_knee=(-16,-23),
    l_ankle=(20,0),       r_ankle=(-19,0),
)
_PUNCH = _p(      # right arm thrust forward
    head=(4,-112), neck=(3,-93), hip_c=(0,-55),
    l_shoulder=(-16,-89), r_shoulder=(14,-89),
    l_elbow=(-25,-69),    r_elbow=(44,-88),
    l_wrist=(-23,-53),    r_wrist=(73,-88),
    l_hip=(-10,-55),      r_hip=(10,-55),
    l_knee=(-13,-27),     r_knee=(15,-27),
    l_ankle=(-12,0),      r_ankle=(14,0),
)
_KICK = _p(       # right leg raised and extended forward
    head=(0,-112), neck=(0,-93), hip_c=(0,-55),
    l_shoulder=(-18,-89), r_shoulder=(12,-89),
    l_elbow=(-30,-73),    r_elbow=(8,-67),
    l_wrist=(-36,-59),    r_wrist=(6,-51),
    l_hip=(-10,-55),      r_hip=(10,-55),
    l_knee=(-14,-28),     r_knee=(42,-51),
    l_ankle=(-14,0),      r_ankle=(70,-33),
)
_JUMP = _p(
    head=(0,-112), neck=(0,-92), hip_c=(0,-52),
    l_shoulder=(-14,-86), r_shoulder=(14,-86),
    l_elbow=(-26,-76),    r_elbow=(28,-76),
    l_wrist=(-30,-63),    r_wrist=(30,-63),
    l_hip=(-10,-52),      r_hip=(10,-52),
    l_knee=(-20,-32),     r_knee=(22,-32),
    l_ankle=(-22,-16),    r_ankle=(24,-16),
)
_HURT = _p(
    head=(-8,-108), neck=(-4,-90), hip_c=(-2,-54),
    l_shoulder=(-20,-86), r_shoulder=(8,-88),
    l_elbow=(-38,-69),    r_elbow=(18,-67),
    l_wrist=(-44,-53),    r_wrist=(22,-51),
    l_hip=(-12,-54),      r_hip=(8,-54),
    l_knee=(-15,-26),     r_knee=(12,-26),
    l_ankle=(-14,0),      r_ankle=(14,0),
)

POSES = {
    'idle':   _IDLE,
    'walk_a': _WALK_A,
    'walk_b': _WALK_B,
    'punch':  _PUNCH,
    'kick':   _KICK,
    'jump':   _JUMP,
    'hurt':   _HURT,
}


def _lerp(p1: dict, p2: dict, t: float) -> dict:
    t = max(0.0, min(1.0, t))
    return {k: (p1[k][0] + (p2[k][0] - p1[k][0]) * t,
                p1[k][1] + (p2[k][1] - p1[k][1]) * t) for k in p1}


# ── Drawing helpers ───────────────────────────────────────────────────────────
def _i(pt) -> tuple[int, int]:
    return (int(pt[0]), int(pt[1]))


def _line(surf, p1, p2, color, w):
    pygame.draw.line(surf, (0, 0, 0), _i(p1), _i(p2), w + 2)
    pygame.draw.line(surf, color,     _i(p1), _i(p2), w)


def _joint(surf, pt, color, r):
    pygame.draw.circle(surf, (0, 0, 0), _i(pt), r + 1)
    pygame.draw.circle(surf, color,     _i(pt), r)


def _impact(surf, pos, size: int, color):
    x, y = int(pos[0]), int(pos[1])
    for i in range(8):
        a  = i * math.pi / 4 + math.pi / 8
        ex = x + int(math.cos(a) * size)
        ey = y + int(math.sin(a) * size)
        pygame.draw.line(surf, color, (x, y), (ex, ey), 3)
    pygame.draw.circle(surf, color,         (x, y), size // 3)
    pygame.draw.circle(surf, (255, 255, 200), (x, y), size // 6)


# ── Fighter ───────────────────────────────────────────────────────────────────
class Fighter:
    def __init__(self, x: int, ground_y: int, is_player: bool) -> None:
        self.x         = float(x)
        self.y         = float(ground_y)
        self.vy        = 0.0
        self.ground_y  = ground_y
        self.is_player = is_player
        self.facing    = 1 if is_player else -1

        self.hp             = 100
        self.state          = "idle"
        self.state_timer    = 0
        self.punch_cooldown = 0
        self.kick_cooldown  = 0
        self.damage_mult    = 1.0
        self._hit_dealt     = False
        self._frame         = 0

    # ─── geometry ─────────────────────────────────────────────────────────────
    def body_rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x) - BODY_W // 2,
                           int(self.y) - BODY_H,
                           BODY_W, BODY_H)

    def attack_rect(self) -> pygame.Rect | None:
        if self.state == "punch" and self.state_timer > PUNCH_DURATION - PUNCH_ACTIVE:
            ax = int(self.x) + (BODY_W // 2 if self.facing == 1
                                else -BODY_W // 2 - PUNCH_REACH)
            return pygame.Rect(ax, int(self.y) - BODY_H * 3 // 4,
                               PUNCH_REACH, PUNCH_HEIGHT)
        if self.state == "kick" and self.state_timer > KICK_DURATION - KICK_ACTIVE:
            ax = int(self.x) + (BODY_W // 2 if self.facing == 1
                                else -BODY_W // 2 - KICK_REACH)
            return pygame.Rect(ax, int(self.y) - BODY_H // 2,
                               KICK_REACH, KICK_HEIGHT)
        return None

    def is_grounded(self) -> bool: return self.y >= self.ground_y
    def is_dead(self)    -> bool:  return self.state == "dead"

    # ─── input ────────────────────────────────────────────────────────────────
    def apply_action(self, action: str | None) -> None:
        if self.state in ("punch", "kick", "hurt", "dead"):
            return
        if action is None:
            if self.state == "walk": self.state = "idle"
            return
        if action == "jump" and self.is_grounded():
            self._enter("jump")
        elif action in ("punch_L", "punch_R"):
            if self.punch_cooldown == 0 and self.state != "jump":
                self._enter("punch")
        elif action in ("kick_L", "kick_R"):
            if self.kick_cooldown == 0 and self.state != "jump":
                self._enter("kick")
        elif action == "move_R":
            self.x += WALK_SPEED
            self.facing = 1
            if self.state == "idle": self.state = "walk"
        elif action == "move_L":
            self.x -= WALK_SPEED
            self.facing = -1
            if self.state == "idle": self.state = "walk"

    # ─── update ───────────────────────────────────────────────────────────────
    def tick(self, opponent: Fighter, arena_l: int, arena_r: int) -> None:
        if self.state == "dead": return

        self._frame += 1
        self.facing = 1 if opponent.x >= self.x else -1

        if self.state_timer > 0:
            self.state_timer -= 1
            if self.state_timer == 0 and self.state in ("punch", "kick", "hurt"):
                self.state = "idle"

        atk = self.attack_rect()
        if atk and not self._hit_dealt and atk.colliderect(opponent.body_rect()):
            dmg = PUNCH_DAMAGE if self.state == "punch" else KICK_DAMAGE
            opponent.take_hit(max(1, int(dmg * self.damage_mult)))
            self._hit_dealt = True

        if self.punch_cooldown > 0: self.punch_cooldown -= 1
        if self.kick_cooldown  > 0: self.kick_cooldown  -= 1

        self.vy += GRAVITY
        self.y  += self.vy
        if self.y >= self.ground_y:
            self.y  = self.ground_y
            self.vy = 0.0
            if self.state == "jump": self.state = "idle"

        half   = BODY_W // 2
        self.x = max(arena_l + half, min(arena_r - half, self.x))

        if self.hp <= 0:
            self.hp    = 0
            self.state = "dead"

    def take_hit(self, damage: int) -> None:
        if self.state in ("hurt", "dead"): return
        self.hp         -= damage
        self.state       = "hurt"
        self.state_timer = HURT_DURATION

    # ─── draw ─────────────────────────────────────────────────────────────────
    def draw(self, surface: pygame.Surface) -> None:
        pose  = self._current_pose()
        cx    = int(self.x)
        cy    = int(self.y)
        f     = self.facing
        color = self._body_color()
        dim   = tuple(max(0, c - 55) for c in color)

        def pt(name):
            dx, dy = pose[name]
            return (cx + f * dx, cy + dy)

        lw = 7   # base limb width

        # ── back limbs (dimmed, drawn first) ──────────────────────────────────
        pygame.draw.line(surface, dim, _i(pt('l_hip')),    _i(pt('l_knee')),   lw - 1)
        pygame.draw.line(surface, dim, _i(pt('l_knee')),   _i(pt('l_ankle')),  lw - 2)
        _joint(surface, pt('l_knee'), dim, 4)

        pygame.draw.line(surface, dim, _i(pt('l_shoulder')), _i(pt('l_elbow')), lw - 2)
        pygame.draw.line(surface, dim, _i(pt('l_elbow')),    _i(pt('l_wrist')), lw - 3)
        _joint(surface, pt('l_elbow'), dim, 3)

        # ── torso ─────────────────────────────────────────────────────────────
        _line(surface, pt('hip_c'),      pt('neck'),       color, lw + 1)
        _line(surface, pt('l_shoulder'), pt('r_shoulder'), color, lw - 1)
        _line(surface, pt('l_hip'),      pt('r_hip'),      color, lw - 2)

        # ── front limbs (outlined) ─────────────────────────────────────────────
        _line(surface, pt('r_hip'),   pt('r_knee'),   color, lw)
        _line(surface, pt('r_knee'),  pt('r_ankle'),  color, lw - 1)
        _joint(surface, pt('r_knee'), color, 5)

        _line(surface, pt('r_shoulder'), pt('r_elbow'), color, lw)
        _line(surface, pt('r_elbow'),    pt('r_wrist'), color, lw - 1)
        _joint(surface, pt('r_elbow'), color, 4)
        _joint(surface, pt('r_wrist'), color, 6)  # fist

        # ── head ──────────────────────────────────────────────────────────────
        _line(surface, pt('neck'), pt('head'), color, lw - 2)
        head = _i(pt('head'))
        pygame.draw.circle(surface, (0, 0, 0), head, HEAD_R + 2)
        pygame.draw.circle(surface, color,     head, HEAD_R)
        self._draw_face(surface, head, f)

        # ── attack effects ────────────────────────────────────────────────────
        if self.state == 'punch' and self.state_timer > PUNCH_DURATION - PUNCH_ACTIVE:
            _impact(surface, pt('r_wrist'), 20, (255, 240, 80))
        elif self.state == 'kick' and self.state_timer > KICK_DURATION - KICK_ACTIVE:
            _impact(surface, pt('r_ankle'), 26, (255, 200, 50))

    def draw_hp_bar(self, surface: pygame.Surface,
                    x: int, y: int, flip: bool) -> None:
        ratio     = max(0.0, self.hp / 100.0)
        fill      = int(HP_BAR_W * ratio)
        bg_rect   = pygame.Rect(x - HP_BAR_W if flip else x, y, HP_BAR_W, HP_BAR_H)
        fill_rect = pygame.Rect(x - fill     if flip else x, y, fill,     HP_BAR_H)
        hp_color = (80, 200, 80) if ratio > 0.5 else (240, 200, 60) if ratio > 0.25 else (220, 60, 60)
        pygame.draw.rect(surface, (50, 50, 50),    bg_rect)
        pygame.draw.rect(surface, hp_color,        fill_rect)
        pygame.draw.rect(surface, (200, 200, 200), bg_rect, 2)

    # ─── internal ─────────────────────────────────────────────────────────────
    def _enter(self, state: str) -> None:
        self.state      = state
        self._hit_dealt = False
        if state == "jump":
            self.vy = JUMP_VY
        elif state == "punch":
            self.state_timer    = PUNCH_DURATION
            self.punch_cooldown = PUNCH_COOLDOWN
        elif state == "kick":
            self.state_timer   = KICK_DURATION
            self.kick_cooldown = KICK_COOLDOWN

    def _current_pose(self) -> dict:
        s = self.state

        if s in ('hurt', 'dead'):
            return POSES['hurt']

        if s == 'punch':
            # Extend quickly, hold, then retract
            t = 1.0 - self.state_timer / PUNCH_DURATION
            if t < 0.30:
                return _lerp(_IDLE, _PUNCH, t / 0.30)
            elif t < 0.60:
                return _PUNCH
            else:
                return _lerp(_IDLE, _PUNCH, 1.0 - (t - 0.60) / 0.40)

        if s == 'kick':
            t = 1.0 - self.state_timer / KICK_DURATION
            if t < 0.38:
                return _lerp(_IDLE, _KICK, t / 0.38)
            elif t < 0.65:
                return _KICK
            else:
                return _lerp(_IDLE, _KICK, 1.0 - (t - 0.65) / 0.35)

        if s == 'jump':
            return _JUMP

        if s == 'walk':
            phase = (self._frame // 10) % 2
            return _WALK_A if phase == 0 else _WALK_B

        # idle: subtle breathing bob on upper body
        bob = math.sin(self._frame * 0.04) * 2.0
        p   = dict(_IDLE)
        for k in ('head', 'neck', 'l_shoulder', 'r_shoulder',
                  'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist'):
            p[k] = (p[k][0], p[k][1] + bob)
        return p

    def _body_color(self) -> tuple:
        if self.state == 'hurt': return _C_HURT
        if self.state == 'dead': return _C_DEAD
        return _C_PLAYER if self.is_player else _C_CPU

    def _draw_face(self, surface, head, facing):
        ex = head[0] + facing * 8
        ey = head[1] - 5

        if self.state in ('dead', 'hurt'):
            # ×× eyes
            for ox in (-2, 2):
                nx = ex + ox
                pygame.draw.line(surface, (30, 30, 30), (nx-4, ey-4), (nx+4, ey+4), 2)
                pygame.draw.line(surface, (30, 30, 30), (nx+4, ey-4), (nx-4, ey+4), 2)
            return

        # Normal eye (white + pupil)
        pygame.draw.circle(surface, (255, 255, 255), (ex, ey), 5)
        pygame.draw.circle(surface, (10,  10,  80),  (ex + facing, ey), 3)

        # Eyebrow
        brow_y  = ey - 10
        inner_x = ex - facing * 5
        outer_x = ex + facing * 5
        if self.state in ('punch', 'kick'):
            # Angry: inner end lower
            pygame.draw.line(surface, (30, 30, 30),
                             (inner_x, brow_y + 3), (outer_x, brow_y - 1), 2)
        else:
            pygame.draw.line(surface, (30, 30, 30),
                             (inner_x, brow_y), (outer_x, brow_y), 2)

        # Mouth
        mx = head[0] + facing * 5
        my = head[1] + 7
        if self.state in ('punch', 'kick'):
            pygame.draw.line(surface, (30, 30, 30), (mx - 4, my + 2), (mx + 4, my - 1), 2)
        else:
            pygame.draw.arc(surface, (30, 30, 30),
                            pygame.Rect(mx - 5, my - 3, 10, 6), 0, math.pi, 2)
