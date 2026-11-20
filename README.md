# Orbital · Lunar Lander Lab

A working version of the attached rocket reward-shaping project, with **real PPO training, Gymnasium Box2D physics, saved neural networks, an interactive local dashboard, honest evaluation, and MP4 recordings**. No API keys, paid services, or GPU required.

## Hosted browser view

The `docs/` directory is a self-contained browser archive of the five measured experiments. It includes the original training curves, Box2D replays, evaluations, CSV downloads, and recorded videos. It can be served by GitHub Pages or any static host, including from a repository subpath.

The hosted view reads saved experiment data. New training, new evaluations, and new recordings run in the Python app described below. Its training button opens setup instructions instead of pretending to run PPO in the browser.

After creating new local runs, rebuild the archive with `python scripts/export_site.py`. Preview it with `python -m http.server 8766 --directory docs`, then open `http://127.0.0.1:8766`. To publish with GitHub Pages, select **Deploy from a branch → main → /docs** in the repository's Pages settings.

## Open the lab

On this Mac, double-click **Launch Lab.command**, then visit **http://127.0.0.1:8765**. The dashboard is already running there for this session. Keep its terminal process running while using it. If that address is in use, use `rocketlab serve --port 8766` instead.

The launcher uses the environment prepared alongside this project when available. On another computer, it creates `.venv` and installs the dependencies on first launch; that step needs internet access. The dashboard itself runs offline and includes local fonts. The models, results, and clips in `runs/` are already included.

For a manual setup (Python 3.11 recommended):

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip swig
python -m pip install -e '.[test]'
rocketlab serve --runs runs
```

If Box2D must compile on your platform, ensure the virtual environment is activated so its `swig` executable is on `PATH`. Python 3.14 may have fewer compatible Box2D wheels; use Python 3.11 if installation fails. `requirements-tested.txt` records the exact packages tested on this Apple Silicon Mac. These wheels may differ across operating systems.

## Included measured results

Each trained policy uses a separate 64 × 64 policy/value MLP, four parallel environments, training seed **7**, and CPU PPO. The best model is chosen by **original environment score** on validation seeds 10000–10004. The following final results use **20 other seeds, 20000–20019**, with the shaping wrapper removed and deterministic actions.

| Policy | Training budget | Mean original score | Score standard deviation | Safe landings | Mean episode frames |
|---|---:|---:|---:|---:|---:|
| Random actions | 0 | -164.09 | 133.31 | 0 / 20 | 126.25 |
| Baseline | 1,000,000 | 274.66 | 20.79 | 20 / 20 | 314.30 |
| Time cost | 1,000,000 | 276.62 | 21.21 | 20 / 20 | 260.45 |
| Hover bonus | 2,000,000 | -20.27 | 17.16 | 0 / 20 | 1000.00 |
| Potential + time cost | 1,000,000 | 271.44 | 21.35 | 20 / 20 | 303.45 |

The deliberately broken hover policy timed out in every test episode. Its training shaped reward rose dramatically while its base score stayed poor. Compare its replay and clip with the repaired policy. These are **single-training-seed demonstrations**, not evidence that one reward design is statistically superior; the hover run also has a larger training budget. Repeat with multiple training seeds and matched budgets for a controlled comparison. PPO collects whole rollouts, so actual steps can slightly exceed the requested budget.

## What the dashboard does

- Replays actual body polygons and eight-number observations captured from a Gymnasium evaluation episode. The custom flight view is a rendering of recorded physics, not a live training feed.
- Starts a new training process with the selected reward, seed, and step budget; saves progress and episode CSVs continuously.
- Plots rolling base and shaped returns so reward hacking is visible.
- Stops training gracefully and saves a reloadable `final.zip` checkpoint.
- Evaluates saved models on 20 unshaped episodes and records them as MP4.
- Selects experiments from the log and exports their CSVs and videos.

The first replay of a new experiment appears at its first 50,000-step validation. Five validation episodes are a quick checkpoint-selection signal. The final test uses 20 different seeds and reports the mean, sample standard deviation, standard error, landing rate, and each individual result. This lab marks a policy solved when the mean original score is at least 200 over at least 10 episodes. Landings are counted from the environment's successful terminal reward, independently of that score criterion. Re-running evaluation repeats the same seeds for reproducibility; use the CLI's `--seed` option for a fresh test set.

## Reproduce and extend

Run commands from this project directory with its environment activated. Use a new output folder for each training run; existing runs are never overwritten by training.

```sh
# Watch random actions through a saved replay.
rocketlab random --episodes 20 --output runs/random-new

# Train the baseline.
rocketlab train --mode baseline --steps 1000000 --seed 7 --output runs/baseline-new

# Add a small frame cost.
rocketlab train --mode time --time-cost 0.05 --steps 1000000 --seed 7 --output runs/time-new

# Break the incentive: reward upright, slow, airborne frames.
rocketlab train --mode hover --hover-bonus 1 --steps 2000000 --seed 7 --output runs/hover-new

# Fix the shaping rule and retrain from scratch.
rocketlab train --mode potential --time-cost 0.05 --potential-scale 20 --steps 1000000 --seed 7 --output runs/fixed-new

# Evaluate without shaping, on a fresh set of seeds.
rocketlab evaluate --model runs/baseline-seed7/best.zip --episodes 100 --seed 30000 --output runs/baseline-fresh-test

# Save the actual Gymnasium render as a video.
rocketlab record --model runs/baseline-seed7/best.zip --seed 20000 --output runs/baseline-seed7/videos

# Run the checks.
python -m pytest -q
```

`best.zip` is selected by validation base score. `final.zip` stores the final training weights, which may behave differently. The dashboard evaluates and records `best.zip` when available. To inspect the exact final learner, supply `final.zip` to the CLI. Model loading uses Python serialization: load model files you trust.

## Reward rules

Edit `rocketlab/rewards.py` to add your own experiment. Every wrapper step reports a reward breakdown, and every completed episode stores both original and shaped returns.

- **Baseline:** original environment reward only.
- **Time cost:** `base - 0.05` each frame.
- **Hover / deliberately broken:** `base + 1` when airborne, `abs(angle) < 0.2`, and the norm of the two normalized velocity components is below `0.3`.
- **Potential / repair:** `base + gamma * Phi(next) - Phi(current) - 0.05`, with `gamma = 0.999` and `Phi(s) = -20 * (distance_to_pad + 0.5 * speed + abs(angle))`.

The PPO discount and shaping discount are the same. True terminal states have zero continuation potential; time-limit truncations retain their next-state potential for bootstrapping. Discounted potential terms telescope; they do not create an unlimited stream of fresh hover bonuses. Adding the frame cost intentionally changes the task's incentives, and its coefficient is an experimental choice rather than a universal fix.

The eight observations are normalized x/y position, normalized x/y velocity, angle, normalized angular velocity, and two leg-contact flags. The four actions are idle, left side engine, main engine, and right side engine. The dashboard keeps positions and velocities in observation units and converts tilt to degrees. Episodes are limited to 1,000 frames at 50 Hz.

## Project map

```text
rocketlab/rewards.py       Reward configuration, terms, and environment wrapper
rocketlab/train.py         PPO, CSV/progress logging, validation, checkpointing
rocketlab/evaluate.py      Unshaped tests, Box2D replay extraction, video capture
rocketlab/server.py        Loopback HTTP server and isolated training workers
rocketlab/web/             Dashboard, canvas rendering, and local fonts
rocketlab/cli.py           Command-line interface
tests/                    Reward math, reproducibility, API and path validation
runs/                     Saved models, configurations, measurements, and clips
```

The server binds to `127.0.0.1` and rejects cross-origin process-launch requests. Jobs started by the dashboard are serialized. Stopping its server also requests a graceful stop of its active worker. Training started separately in another terminal is observed in the log but controlled from that terminal.

## References

- [Gymnasium Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) — environment, observations, actions, reward, and termination.
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) — algorithm and implementation.
- [Gymnasium RecordVideo](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordVideo) — MP4 recording.
- [Policy invariance under reward transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) — potential-based reward shaping.

Inspired by the user-provided article, “Training an AI to land a rocket with reward shaping,” attributed there to Hari (@harixganesh). Bundled DM Sans and Space Grotesk fonts are distributed under their included SIL Open Font Licenses. Gymnasium, Stable-Baselines3, PyTorch, and other dependencies retain their own licenses.
