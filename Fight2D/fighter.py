"""Fighter entity: state machine, physics, hit detection, drawing."""

from __future__ import annotations
import pygame

# Physics
GRAVITY      = 0.65
WALK_SPEED   = 5
JUMP_VY      = -15

# Attack parameters
PUNCH_DURATION   = 18   # total frames in punch state
PUNCH_ACTIVE     = 8    # first N frames: hitbox is live
PUNCH_REACH      = 80   # px, horizontal extent
PUNCH_HEIGHT     = 50   # px, vertical extent
PUNCH_DAMAGE     = 12

KICK_DURATION    = 28
KICK_ACTIVE      = 10
KICK_REACH       = 100
KICK_HEIGHT      = 60
KICK_DAMAGE      = 20

HURT_DURATION    = 12   # frames, invincible while hurting

PUNCH_COOLDOWN   = 25   # frames before same attack can start again
KICK_COOLDOWN    = 40

# Body geometry
BODY_W, BODY_H = 50, 100
HEAD_R         = 18

# Colours
COLOR_IDLE   = {True:  (100, 180, 255),   # player (blue)
                False: (255, 100, 100)}    # cpu (red)
COLOR_HURT   = (255, 255, 80)
COLOR_ATTACK = (255, 160, 30)
COLOR_DEAD   = (80,  80,  80)

HP_BAR_W, HP_BAR_H = 280, 22


class Fighter:
    def __init__(self, x: int, ground_y: int, is_player: bool) -> None:
        self.x         = float(x)
        self.y         = float(ground_y)
        self.vy        = 0.0
        self.ground_y  = ground_y
        self.is_player = is_player
        self.facing    = 1 if is_player else -1   # +1 = right, -1 = left

        self.hp             = 100
        self.state          = "idle"
        self.state_timer    = 0
        self.punch_cooldown = 0
        self.kick_cooldown  = 0
        self.damage_mult    = 1.0     # scale outgoing attack damage
        self._hit_dealt     = False   # prevent double-hit per swing

    # ───────────────────────────────────────────── geometry ──────────────────
    def body_rect(self) -> pygame.Rect:
        return pygame.Rect(int(self.x) - BODY_W // 2,
                           int(self.y) - BODY_H,
                           BODY_W, BODY_H)

    def attack_rect(self) -> pygame.Rect | None:
        """Active hitbox during the early frames of punch / kick; else None."""
        if self.state == "punch" and self.state_timer > PUNCH_DURATION - PUNCH_ACTIVE:
            if self.facing == 1:
                ax = int(self.x) + BODY_W // 2
            else:
                ax = int(self.x) - BODY_W // 2 - PUNCH_REACH
            ay = int(self.y) - BODY_H * 3 // 4
            return pygame.Rect(ax, ay, PUNCH_REACH, PUNCH_HEIGHT)

        if self.state == "kick" and self.state_timer > KICK_DURATION - KICK_ACTIVE:
            if self.facing == 1:
                ax = int(self.x) + BODY_W // 2
            else:
                ax = int(self.x) - BODY_W // 2 - KICK_REACH
            ay = int(self.y) - BODY_H // 2
            return pygame.Rect(ax, ay, KICK_REACH, KICK_HEIGHT)

        return None

    def is_grounded(self) -> bool:
        return self.y >= self.ground_y

    def is_dead(self) -> bool:
        return self.state == "dead"

    # ───────────────────────────────────────────── input ─────────────────────
    def apply_action(self, action: str | None) -> None:
        """Accept an action string. Blocked when already attacking / hurt / dead."""
        if self.state in ("punch", "kick", "hurt", "dead"):
            return
        if action is None:
            if self.state == "walk":
                self.state = "idle"
            return

        if action == "jump" and self.is_grounded():
            self._enter("jump")
        elif action == "punch_L" or action == "punch_R":
            if self.punch_cooldown == 0 and self.state != "jump":
                self._enter("punch")
        elif action == "kick_L" or action == "kick_R":
            if self.kick_cooldown == 0 and self.state != "jump":
                self._enter("kick")
        elif action == "move_R":
            self.x += WALK_SPEED
            self.facing = 1
            if self.state == "idle":
                self.state = "walk"
        elif action == "move_L":
            self.x -= WALK_SPEED
            self.facing = -1
            if self.state == "idle":
                self.state = "walk"

    # ───────────────────────────────────────────── update ────────────────────
    def tick(self, opponent: Fighter, arena_l: int, arena_r: int) -> None:
        if self.state == "dead":
            return

        # Face opponent (always updated)
        self.facing = 1 if opponent.x >= self.x else -1

        # State timer
        if self.state_timer > 0:
            self.state_timer -= 1
            if self.state_timer == 0:
                if self.state in ("punch", "kick"):
                    self.state = "idle"
                elif self.state == "hurt":
                    self.state = "idle"

        # Hit detection during active attack frames
        atk = self.attack_rect()
        if atk and not self._hit_dealt:
            if atk.colliderect(opponent.body_rect()):
                dmg = PUNCH_DAMAGE if self.state == "punch" else KICK_DAMAGE
                opponent.take_hit(max(1, int(dmg * self.damage_mult)))
                self._hit_dealt = True

        # Cooldowns
        if self.punch_cooldown > 0: self.punch_cooldown -= 1
        if self.kick_cooldown  > 0: self.kick_cooldown  -= 1

        # Gravity
        self.vy += GRAVITY
        self.y  += self.vy
        if self.y >= self.ground_y:
            self.y  = self.ground_y
            self.vy = 0.0
            if self.state == "jump":
                self.state = "idle"

        # Arena bounds
        half = BODY_W // 2
        self.x = max(arena_l + half, min(arena_r - half, self.x))

        # Death
        if self.hp <= 0:
            self.hp    = 0
            self.state = "dead"

    def take_hit(self, damage: int) -> None:
        if self.state in ("hurt", "dead"):
            return
        self.hp -= damage
        self.state       = "hurt"
        self.state_timer = HURT_DURATION

    # ───────────────────────────────────────────── draw ──────────────────────
    def draw(self, surface: pygame.Surface) -> None:
        if self.state == "hurt":
            color = COLOR_HURT
        elif self.state in ("punch", "kick"):
            color = COLOR_ATTACK
        elif self.state == "dead":
            color = COLOR_DEAD
        else:
            color = COLOR_IDLE[self.is_player]

        pygame.draw.rect(surface, color, self.body_rect(), border_radius=6)
        head_pos = (int(self.x), int(self.y) - BODY_H - HEAD_R)
        pygame.draw.circle(surface, color, head_pos, HEAD_R)

        # Active hitbox outline
        atk = self.attack_rect()
        if atk:
            pygame.draw.rect(surface, (255, 255, 0), atk, 2)

    def draw_hp_bar(self, surface: pygame.Surface,
                    x: int, y: int, flip: bool) -> None:
        ratio     = max(0.0, self.hp / 100.0)
        fill      = int(HP_BAR_W * ratio)
        bg_rect   = pygame.Rect(x - HP_BAR_W if flip else x, y, HP_BAR_W, HP_BAR_H)
        fill_rect = pygame.Rect(x - fill if flip else x, y, fill, HP_BAR_H)
        if ratio > 0.5:
            hp_color = (80, 200, 80)
        elif ratio > 0.25:
            hp_color = (240, 200, 60)
        else:
            hp_color = (220, 60, 60)
        pygame.draw.rect(surface, (50, 50, 50), bg_rect)
        pygame.draw.rect(surface, hp_color,     fill_rect)
        pygame.draw.rect(surface, (200, 200, 200), bg_rect, 2)

    # ───────────────────────────────────────────── internal ──────────────────
    def _enter(self, state: str) -> None:
        self.state       = state
        self._hit_dealt  = False
        if state == "jump":
            self.vy = JUMP_VY
        elif state == "punch":
            self.state_timer    = PUNCH_DURATION
            self.punch_cooldown = PUNCH_COOLDOWN
        elif state == "kick":
            self.state_timer   = KICK_DURATION
            self.kick_cooldown = KICK_COOLDOWN
