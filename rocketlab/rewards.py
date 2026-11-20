"""Inspectable reward rules; evaluation always uses the original environment."""

from dataclasses import asdict, dataclass
import math

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    mode: str = "baseline"
    time_cost: float = 0.05
    hover_bonus: float = 1.0
    potential_scale: float = 20.0
    gamma: float = 0.999

    def __post_init__(self):
        if self.mode not in {"baseline", "time", "hover", "potential"}:
            raise ValueError(f"Unknown reward mode: {self.mode}")
        for key in ("time_cost", "hover_bonus", "potential_scale"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if not math.isfinite(self.gamma) or not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")


def potential(obs, scale=20.0):
    """Higher is better: near the pad, slow, upright. Zero at the ideal state."""
    x, y, vx, vy, angle = (float(v) for v in obs[:5])
    return -scale * (math.hypot(x, y) + 0.5 * math.hypot(vx, vy) + abs(angle))


def reward_terms(previous, current, terminated, config):
    terms = {"time": 0.0, "hover": 0.0, "potential": 0.0}
    if config.mode in {"time", "potential"}:
        terms["time"] = -config.time_cost
    if config.mode == "hover":
        # Deliberately exploitable: upright + slow pays again on every frame.
        upright = abs(float(current[4])) < 0.2
        slow = np.linalg.norm(current[2:4]) < 0.3
        airborne = not (current[6] or current[7])
        if upright and slow and airborne and not terminated:
            terms["hover"] = config.hover_bonus
    if config.mode == "potential":
        # A true terminal state has zero continuation potential. Time limits
        # are truncations, so retain their potential for PPO bootstrapping.
        next_phi = 0.0 if terminated else potential(current, config.potential_scale)
        terms["potential"] = config.gamma * next_phi - potential(previous, config.potential_scale)
    return terms


class ShapedReward(gym.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.config = config
        self.previous = None
        self.base_return = 0.0
        self.shaped_return = 0.0
        self.length = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous = obs.copy()
        self.base_return = self.shaped_return = 0.0
        self.length = 0
        return obs, info

    def step(self, action):
        obs, base, terminated, truncated, info = self.env.step(action)
        terms = reward_terms(self.previous, obs, terminated, self.config)
        shaped = float(base + sum(terms.values()))
        self.previous = obs.copy()
        self.base_return += float(base)
        self.shaped_return += shaped
        self.length += 1
        info = dict(info)
        info["reward_terms"] = {"base": float(base), **terms}
        if terminated or truncated:
            info["lab_episode"] = {
                "base": self.base_return, "shaped": self.shaped_return,
                "length": self.length, "truncated": bool(truncated),
            }
        return obs, shaped, terminated, truncated, info


def make_env(config=None, render_mode=None):
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    return ShapedReward(env, config or RewardConfig())
