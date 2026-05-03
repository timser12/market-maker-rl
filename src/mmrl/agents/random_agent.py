from __future__ import annotations

import random


class RandomAgent:
    def __init__(self, n_actions: int, seed: int = 7):
        self.n_actions = n_actions
        self.rng = random.Random(seed)

    def act(self, _obs) -> int:
        return self.rng.randrange(self.n_actions)
