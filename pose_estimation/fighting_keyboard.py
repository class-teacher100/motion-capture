"""Translates FightingGestureMapper output into physical keyboard events.

Held gestures (move_left, move_right, crouch) keep the key pressed until the
gesture ends.  One-shot gestures (jump, punches, kicks) send a brief press/release.
"""

from pynput.keyboard import Controller

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

    def update(self, state: dict) -> None:
        wanted = {g for g in _HELD if state.get(g)}

        for g in self._held_now - wanted:
            self._kb.release(_KEY_MAP[g])
        for g in wanted - self._held_now:
            self._kb.press(_KEY_MAP[g])
        self._held_now = wanted

        for g in _ONESHOT:
            if state.get(g):
                k = _KEY_MAP[g]
                self._kb.press(k)
                self._kb.release(k)

    def release_all(self) -> None:
        for g in list(self._held_now):
            self._kb.release(_KEY_MAP[g])
        self._held_now.clear()

    def active_keys(self) -> list[str]:
        """Return uppercase key labels currently active (for on-screen display)."""
        keys = [_KEY_MAP[g].upper() for g in self._held_now]
        return sorted(keys)
