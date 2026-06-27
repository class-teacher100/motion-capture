"""Translates FightingGestureMapper output into physical keyboard events.

Held gestures (move_left, move_right, crouch) keep the key pressed until the
gesture ends.  One-shot gestures (jump, punches, kicks) are held for
ONESHOT_HOLD_FRAMES frames so games polling at ~60 fps reliably detect them.
"""

from pynput.keyboard import Controller

ONESHOT_HOLD_FRAMES = 4  # frames to hold one-shot keys (~67 ms at 60 fps)

_KEY_MAP: dict[str, str] = {
    'move_left':    'a',
    'move_right':   'd',
    'crouch':       's',
    'jump':         'w',
    'weak_punch':   'i',
    'med_punch':    'o',
    'strong_punch': 'p',
    'weak_kick':    'j',
    'med_kick':     'k',
    'strong_kick':  'l',
}

_HELD = frozenset({'move_left', 'move_right', 'crouch'})
_ONESHOT = frozenset({'jump', 'weak_punch', 'med_punch', 'strong_punch',
                      'weak_kick', 'med_kick', 'strong_kick'})


class FightingKeyboardController:
    def __init__(self):
        self._kb = Controller()
        self._held_now: set[str] = set()
        # gesture -> remaining frames to keep key pressed
        self._oneshot_countdown: dict[str, int] = {}

    def update(self, state: dict) -> None:
        # --- Held keys (A/D/S) ---
        wanted = {g for g in _HELD if state.get(g)}
        for g in self._held_now - wanted:
            self._kb.release(_KEY_MAP[g])
        for g in wanted - self._held_now:
            self._kb.press(_KEY_MAP[g])
        self._held_now = wanted

        # --- One-shot keys: press on trigger, hold for N frames, then release ---
        for g in _ONESHOT:
            if state.get(g) and g not in self._oneshot_countdown:
                self._kb.press(_KEY_MAP[g])
                self._oneshot_countdown[g] = ONESHOT_HOLD_FRAMES

        # Tick down countdowns; release keys that have expired
        done = [g for g, n in self._oneshot_countdown.items() if n <= 1]
        for g in done:
            self._kb.release(_KEY_MAP[g])
            del self._oneshot_countdown[g]
        for g in self._oneshot_countdown:
            self._oneshot_countdown[g] -= 1

    def release_all(self) -> None:
        for g in list(self._held_now):
            self._kb.release(_KEY_MAP[g])
        self._held_now.clear()
        for g in list(self._oneshot_countdown):
            self._kb.release(_KEY_MAP[g])
        self._oneshot_countdown.clear()
