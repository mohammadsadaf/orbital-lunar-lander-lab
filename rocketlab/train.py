import csv
from dataclasses import asdict
from datetime import datetime, timezone
import signal
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from .evaluate import evaluate
from .rewards import RewardConfig, make_env
from .storage import write_json


class Progress(BaseCallback):
    def __init__(self, directory, total, config, eval_every=50000):
        super().__init__()
        self.directory, self.total, self.config = Path(directory), total, config
        self.eval_every = eval_every
        self.next_eval = eval_every
        self.started = time.monotonic()
        self.last_flush = 0.0
        self.rows, self.validations = [], []
        self.best = float("-inf")
        self.stop_requested = False
        self.state = "training"
        self.stream = None

    def _on_training_start(self):
        self.stream = (self.directory / "episodes.csv").open("w", newline="")
        self.writer = csv.DictWriter(self.stream, fieldnames=["step", "base", "shaped", "length", "truncated"])
        self.writer.writeheader()
        self.flush()

    def flush(self):
        elapsed = time.monotonic() - self.started
        recent = self.rows[-100:]
        write_json(self.directory / "progress.json", {
            "state": self.state, "step": self.num_timesteps, "total": self.total,
            "elapsed": elapsed, "fps": self.num_timesteps / max(elapsed, 0.001),
            "episodes": len(self.rows), "config": asdict(self.config),
            "mean_base": sum(r["base"] for r in recent) / len(recent) if recent else None,
            "mean_shaped": sum(r["shaped"] for r in recent) / len(recent) if recent else None,
            "history": self.rows[-3000:], "validations": self.validations,
        })
        if self.stream:
            self.stream.flush()
        self.last_flush = time.monotonic()

    def _on_step(self):
        for info in self.locals["infos"]:
            if "lab_episode" in info:
                row = {"step": self.num_timesteps, **info["lab_episode"]}
                self.rows.append(row)
                self.writer.writerow(row)
        if time.monotonic() - self.last_flush > 1:
            self.flush()
        if self.num_timesteps >= self.next_eval and not self.stop_requested:
            self.next_eval += self.eval_every
            self.state = "validating"
            self.flush()
            summary, _ = evaluate(self.model, episodes=5, seed=10000,
                                  output=self.directory / "validation", policy=self.config.mode)
            self.validations.append({"step": self.num_timesteps, "mean": summary["mean"],
                                     "landing_rate": summary["landing_rate"]})
            if summary["mean"] > self.best:
                self.best = summary["mean"]
                self.model.save(self.directory / "best")
            self.state = "training"
            self.flush()
        return not self.stop_requested


def train(directory, config=None, steps=1000000, seed=7, n_envs=4, eval_every=50000):
    config = config or RewardConfig()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", {"reward": asdict(config), "steps": steps,
               "seed": seed, "n_envs": n_envs, "created": datetime.now(timezone.utc).isoformat(),
               "algorithm": "PPO", "environment": "LunarLander-v3", "network": [64, 64]})
    # Tiny MLPs are faster on CPU with one BLAS thread than many competing threads.
    torch.set_num_threads(1)
    env = DummyVecEnv([lambda: Monitor(make_env(config)) for _ in range(n_envs)])
    callback = Progress(directory, steps, config, eval_every)
    old_handler = signal.signal(signal.SIGINT, lambda *_: setattr(callback, "stop_requested", True))
    try:
        model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=1,
                    n_steps=1024, batch_size=64, n_epochs=4, gamma=config.gamma,
                    gae_lambda=0.98, ent_coef=0.01, learning_rate=3e-4,
                    policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])})
        model.learn(total_timesteps=steps, callback=callback)
        model.save(directory / "final")
        callback.state = "stopped" if callback.stop_requested else "evaluating"
        callback.flush()
        if not callback.stop_requested:
            # Checkpoint selection used validation seeds; final testing uses new seeds.
            chosen = directory / "best.zip"
            if chosen.exists():
                model = PPO.load(chosen, device="cpu")
            evaluate(model, episodes=20, seed=20000, output=directory, policy=config.mode)
            callback.state = "complete"
            callback.flush()
        return directory
    except Exception as error:
        callback.state = "failed"
        callback.flush()
        write_json(directory / "error.json", {"error": str(error)})
        raise
    finally:
        signal.signal(signal.SIGINT, old_handler)
        if callback.stream:
            callback.stream.close()
        env.close()
