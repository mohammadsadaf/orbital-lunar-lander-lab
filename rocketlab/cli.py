import argparse
import json
from pathlib import Path


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main():
    parser = argparse.ArgumentParser(description="Train, inspect, and compare real LunarLander-v3 agents.")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Open the local mission control dashboard")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--runs", type=Path, default=Path("runs"))
    learn = sub.add_parser("train")
    learn.add_argument("--output", type=Path, required=True)
    learn.add_argument("--mode", choices=["baseline", "time", "hover", "potential"], default="baseline")
    learn.add_argument("--steps", type=positive, default=1000000)
    learn.add_argument("--seed", type=int, default=7)
    learn.add_argument("--envs", type=positive, default=4)
    learn.add_argument("--eval-every", type=positive, default=50000)
    learn.add_argument("--time-cost", type=float, default=0.05)
    learn.add_argument("--hover-bonus", type=float, default=1.0)
    learn.add_argument("--potential-scale", type=float, default=20.0)
    learn.add_argument("--gamma", type=float, default=0.999)
    for name in ["evaluate", "random", "record"]:
        cmd = sub.add_parser(name)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--seed", type=int, default=20000)
        if name != "random":
            cmd.add_argument("--model", type=Path, required=True)
        if name != "record":
            cmd.add_argument("--episodes", type=positive, default=20)
    args = parser.parse_args()
    if args.command == "serve":
        from .server import serve
        serve(args.runs, args.port)
    elif args.command == "train":
        from .rewards import RewardConfig
        from .train import train
        train(args.output, RewardConfig(args.mode, args.time_cost, args.hover_bonus,
                                       args.potential_scale, args.gamma),
              args.steps, args.seed, args.envs, args.eval_every)
    else:
        import torch
        from stable_baselines3 import PPO
        from .evaluate import evaluate, rollout
        torch.set_num_threads(1)
        model = None if args.command == "random" else PPO.load(args.model, device="cpu")
        if args.command == "record":
            row, _ = rollout(model, seed=args.seed, video_dir=args.output, policy="ppo")
            print(json.dumps(row, indent=2))
        else:
            result, _ = evaluate(model, args.episodes, args.seed, args.output,
                                 policy="random" if model is None else "ppo")
            print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
