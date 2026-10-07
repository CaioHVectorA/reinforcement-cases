"""Local PPO + self-play league training for HaxBall.

This is the machine-local counterpart of notebooks/train_haxball_rl_from_scratch_colab.ipynb.
Run from the repository root, for example:

    python train_haxball_rl_local.py --iterations 10 --steps 1024

The default settings mirror the notebook, but every expensive setting can be overridden
from the command line. Checkpoints and metrics are written under checkpoints/.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from haxball.bots import (
    HeuristicBot,
    MasterBot,
    PressBot,
    RLBot,
    StrikerBot,
    WallReboundBot,
)
from haxball.core.constants import Team
from haxball.core.game import HaxBallGame
from haxball.core.stadium import Stadium
from haxball.gym_env.haxball_env import HaxBallEnv
from haxball.rl.models.mlp_policy import ActorCriticMLP


ROOT = Path(__file__).resolve().parent
MAPS = ROOT / "haxball" / "maps"


@dataclass
class Config:
    map_name: str = "futsal_2v2"
    scale: float = 1.0
    policy: str = "mlp"
    iterations: int = 1000
    steps: int = 4096
    learning_rate: float = 3e-4
    entropy_coef: float = 0.012
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 128
    update_epochs: int = 8
    stage1_ratio: float = 0.20
    eval_frequency: int = 25
    eval_matches: int = 2
    max_steps: int = 1500
    checkpoint_dir: Path = ROOT / "checkpoints"
    warm_start: Optional[Path] = None
    device: str = "auto"
    seed: int = 42
    plot_mode: str = "save"
    stop_event: object = field(default=None, repr=False)


BOT_FACTORIES = {
    "heuristic": HeuristicBot,
    "striker": StrikerBot,
    "wall": WallReboundBot,
    "press": PressBot,
    "master": MasterBot,
}


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitado, mas não está disponível neste Python.")
    return torch.device(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_agent(config: Config, device: torch.device) -> nn.Module:
    if config.policy != "mlp":
        raise ValueError("A versão local suporta 'mlp'; a atenção pode ser adicionada sem mudar o treino.")

    agent = ActorCriticMLP(
        obs_dim=61,
        act_dim=18,
        is_discrete=True,
        hidden_dim=128,
    )
    if config.warm_start:
        payload = torch.load(config.warm_start, map_location=device, weights_only=True)
        state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
        agent.load_state_dict(state, strict=False)
        print(f"Warm-start carregado: {config.warm_start}")
    else:
        print("Agente inicializado do zero.")
    return agent.to(device)


def make_env(config: Config, opponent) -> HaxBallEnv:
    stadium_file = MAPS / f"{config.map_name}.hbs"
    if not stadium_file.exists():
        raise FileNotFoundError(f"Mapa não encontrado: {stadium_file}")
    return HaxBallEnv(
        stadium_file=str(stadium_file),
        stadium_scale=config.scale,
        opponent_bot=opponent,
        discrete_actions=True,
        max_steps=config.max_steps,
        score_limit=3,
    )


def evaluate_agent(agent: nn.Module, config: Config, device: torch.device) -> Dict[str, dict]:
    opponents = {
        "heuristic": HeuristicBot(name="eval_heuristic"),
        "striker": StrikerBot(name="eval_striker"),
        "wall": WallReboundBot(name="eval_wall"),
        "master": MasterBot(name="eval_master"),
    }
    results: Dict[str, dict] = {}
    agent.eval()

    for name, opponent in opponents.items():
        env = make_env(config, opponent)
        wins = goals_for = goals_against = 0
        for match in range(config.eval_matches):
            obs, _ = env.reset(seed=config.seed + match)
            done = False
            while not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(obs_tensor)
                obs, _, terminated, truncated, _ = env.step(int(action.item()))
                done = terminated or truncated
            wins += int(env.game.red_score > env.game.blue_score)
            goals_for += env.game.red_score
            goals_against += env.game.blue_score

        results[name] = {
            "win_rate": 100.0 * wins / max(1, config.eval_matches),
            "goals_for": goals_for,
            "goals_against": goals_against,
        }
    return results


def save_state(agent: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), path)


def save_training_checkpoint(
    agent: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    iteration: int,
    metrics: list[dict],
    path: Path,
) -> None:
    """Save enough state to continue training, not just to run inference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": agent.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "iteration": iteration,
            "metrics": metrics,
        },
        path,
    )


class TrainingDashboard:
    """Live notebook-style dashboard with customizable scale adjustments and PNG snapshot."""

    def __init__(self, output_dir: Path, mode: str, use_log_scale: bool = False, y_limit_margin: float = 1.0):
        self.output_dir = output_dir
        self.output_path = output_dir / "training_dashboard.png"
        self.use_log_scale = use_log_scale
        self.y_limit_margin = y_limit_margin
        self.plt = None
        self.figure = None
        self.axes = None
        if mode == "off":
            return
        try:
            import matplotlib.pyplot as plt

            self.plt = plt
            self.live = mode == "live"
            if self.live:
                self.plt.ion()
            self.figure, self.axes = self.plt.subplots(2, 3, figsize=(15, 8))
            self.figure.suptitle("HaxBall PPO Local | Treinamento em tempo real", fontsize=15)
            self.figure.tight_layout(rect=(0, 0, 1, 0.95))
        except Exception as error:
            print(f"Aviso: gráficos desativados ({error})")

    def update(self, metrics: list[dict], use_log_scale: Optional[bool] = None, custom_y_limits: Optional[Dict[str, tuple]] = None) -> None:
        if self.plt is None or self.figure is None or self.axes is None:
            return

        if use_log_scale is not None:
            self.use_log_scale = use_log_scale

        iterations = [row["iteration"] for row in metrics]
        reward = [row["mean_reward"] for row in metrics]
        win_rate = [row["train_win_rate"] for row in metrics]
        policy_loss = [row["policy_loss"] for row in metrics]
        value_loss = [row["value_loss"] for row in metrics]
        entropy = [row["entropy"] for row in metrics]
        gauntlet = [row.get("gauntlet_score", np.nan) for row in metrics]

        for axis in self.axes.flat:
            axis.clear()
            axis.grid(True, alpha=0.25, which="both" if self.use_log_scale else "major")

        # 1. Recompensa média
        self.axes[0, 0].plot(iterations, reward, color="#20a464", linewidth=2)
        self.axes[0, 0].set_title("Recompensa média")
        self.axes[0, 0].set_xlabel("Iteração")
        if custom_y_limits and "reward" in custom_y_limits:
            self.axes[0, 0].set_ylim(custom_y_limits["reward"])

        # 2. Win rate no treino
        self.axes[0, 1].plot(iterations, win_rate, color="#2878c8", linewidth=2)
        self.axes[0, 1].set_title("Win rate no treino (%)")
        self.axes[0, 1].set_ylim(0, 100)
        self.axes[0, 1].set_xlabel("Iteração")

        # 3. Losses PPO (com opção de Escala Logarítmica gigante p/ aproximar de reta/linha)
        self.axes[0, 2].plot(iterations, policy_loss, label="Policy", color="#d84a4a", linewidth=2)
        self.axes[0, 2].plot(iterations, value_loss, label="Value", color="#e39a25", linewidth=2)
        self.axes[0, 2].set_title("Losses PPO" + (" (Escala Log)" if self.use_log_scale else ""))
        self.axes[0, 2].set_xlabel("Iteração")
        self.axes[0, 2].legend()
        if self.use_log_scale:
            self.axes[0, 2].set_yscale("log")
        if custom_y_limits and "losses" in custom_y_limits:
            self.axes[0, 2].set_ylim(custom_y_limits["losses"])

        # 4. Entropia da política
        self.axes[1, 0].plot(iterations, entropy, color="#824caf", linewidth=2)
        self.axes[1, 0].set_title("Entropia da política")
        self.axes[1, 0].set_xlabel("Iteração")
        if custom_y_limits and "entropy" in custom_y_limits:
            self.axes[1, 0].set_ylim(custom_y_limits["entropy"])

        # 5. Gauntlet médio
        self.axes[1, 1].plot(iterations, gauntlet, marker="o", color="#dc6b28", linewidth=2)
        self.axes[1, 1].set_title("Gauntlet médio (%)")
        self.axes[1, 1].set_ylim(0, 100)
        self.axes[1, 1].set_xlabel("Iteração")

        # 6. Avaliação por bot
        latest_eval = metrics[-1].get("evaluation", {})
        names = list(latest_eval)
        values = [latest_eval[name]["win_rate"] for name in names]
        if names:
            self.axes[1, 2].bar(names, values, color=["#2878c8", "#d84a4a", "#e39a25", "#824caf"])
            self.axes[1, 2].set_ylim(0, 100)
        self.axes[1, 2].set_title("Última avaliação por bot")
        self.axes[1, 2].tick_params(axis="x", rotation=25)

        self.figure.suptitle(
            f"HaxBall PPO Local | Iteração {metrics[-1]['iteration']} | "
            f"Passos {metrics[-1]['steps']:,}",
            fontsize=15,
        )
        self.figure.tight_layout(rect=(0, 0, 1, 0.95))
        self.figure.savefig(self.output_path, dpi=120)
        if getattr(self, "live", False):
            self.plt.pause(0.001)

    def close(self) -> None:
        if self.plt is not None and self.figure is not None:
            if getattr(self, "live", False):
                self.plt.ioff()
            self.plt.close(self.figure)


def train(config: Config) -> None:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    opponent = HeuristicBot(name="league_heuristic")
    env = make_env(config, opponent)
    agent = make_agent(config, device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.iterations, eta_min=1e-5
    )

    stage1_iterations = max(1, int(config.iterations * config.stage1_ratio))
    league_pool: list[Path] = []
    best_score = -1.0
    metrics = []
    dashboard = TrainingDashboard(config.checkpoint_dir, config.plot_mode)
    latest_path = config.checkpoint_dir / "haxball_rl_latest.pt"
    latest_payload = None
    if config.warm_start:
        latest_payload = torch.load(config.warm_start, map_location=device, weights_only=True)
        if isinstance(latest_payload, dict) and "optimizer_state" in latest_payload:
            optimizer.load_state_dict(latest_payload["optimizer_state"])
            scheduler.load_state_dict(latest_payload.get("scheduler_state", {}))
            metrics = list(latest_payload.get("metrics", []))
            print(f"Estado do otimizador restaurado: iteração {latest_payload.get('iteration', 0)}")

    current_iteration = int(latest_payload.get("iteration", 0)) if isinstance(latest_payload, dict) else 0

    def save_on_exit() -> None:
        save_training_checkpoint(agent, optimizer, scheduler, current_iteration, metrics, latest_path)

    atexit.register(save_on_exit)
    start_time = time.time()

    print("=" * 72)
    print("HAXBALL PPO LOCAL")
    print(f"Mapa: {config.map_name} | Device: {device} | Política: {config.policy}")
    print(f"Iterações: {config.iterations:,} | Passos/iteração: {config.steps:,}")
    print(f"Fase 1 até a iteração {stage1_iterations}; depois liga de self-play")
    print("=" * 72)

    obs, _ = env.reset(seed=config.seed)
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)

    for iteration in range(current_iteration + 1, config.iterations + 1):
        if config.stop_event is not None and config.stop_event.is_set():
            print("Parada solicitada; salvando o último checkpoint concluído.")
            break
        current_iteration = iteration
        if iteration == stage1_iterations + 1:
            stage_path = config.checkpoint_dir / "gen_stage1.pt"
            save_state(agent, stage_path)
            league_pool.append(stage_path)
            print(f"Liga iniciada com {stage_path.name}")

        if iteration > stage1_iterations and iteration % 50 == 0:
            generation_path = config.checkpoint_dir / f"gen_iter_{iteration}.pt"
            save_state(agent, generation_path)
            league_pool.append(generation_path)

        if iteration > stage1_iterations and league_pool:
            choice = random.random()
            if choice < 0.70:
                opponent = RLBot(model_path=str(league_pool[-1]), name="recent_self")
            elif choice < 0.90:
                opponent = RLBot(model_path=str(random.choice(league_pool)), name="historic_self")
            else:
                opponent = PressBot(name="press_baseline")
            env.opponent_bot = opponent

        obs_buffer = torch.zeros((config.steps, 61), device=device)
        action_buffer = torch.zeros(config.steps, dtype=torch.long, device=device)
        logprob_buffer = torch.zeros(config.steps, device=device)
        reward_buffer = torch.zeros(config.steps, device=device)
        done_buffer = torch.zeros(config.steps, device=device)
        value_buffer = torch.zeros(config.steps, device=device)

        episode_rewards = []
        episode_wins = 0
        episode_count = 0
        current_reward = 0.0
        agent.train()

        for step in range(config.steps):
            obs_buffer[step] = obs_tensor
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_tensor.unsqueeze(0))
            action_buffer[step] = action.squeeze()
            logprob_buffer[step] = logprob.squeeze()
            value_buffer[step] = value.squeeze()

            next_obs, reward, terminated, truncated, info = env.step(int(action.item()))
            done = terminated or truncated
            reward_buffer[step] = reward
            done_buffer[step] = float(done)
            current_reward += reward

            if done:
                episode_rewards.append(current_reward)
                episode_count += 1
                episode_wins += int(info.get("winner") == Team.RED)
                current_reward = 0.0
                next_obs, _ = env.reset()
            obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            next_value = agent.get_value(obs_tensor.unsqueeze(0)).squeeze()
            advantages = torch.zeros_like(reward_buffer)
            last_advantage = torch.tensor(0.0, device=device)
            for step in reversed(range(config.steps)):
                next_nonterminal = 1.0 - done_buffer[step]
                next_values = next_value if step == config.steps - 1 else value_buffer[step + 1]
                delta = reward_buffer[step] + config.gamma * next_values * next_nonterminal - value_buffer[step]
                last_advantage = delta + config.gamma * config.gae_lambda * next_nonterminal * last_advantage
                advantages[step] = last_advantage
            returns = advantages + value_buffer

        indices = np.arange(config.steps)
        policy_loss_total = value_loss_total = entropy_total = 0.0
        update_count = 0
        for _ in range(config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, config.steps, config.batch_size):
                batch_indices = torch.as_tensor(
                    indices[start:start + config.batch_size], dtype=torch.long, device=device
                )
                _, new_logprob, entropy, new_value = agent.get_action_and_value(
                    obs_buffer[batch_indices], action_buffer[batch_indices]
                )
                ratio = (new_logprob - logprob_buffer[batch_indices]).exp()
                batch_advantage = advantages[batch_indices]
                batch_advantage = (batch_advantage - batch_advantage.mean()) / (batch_advantage.std() + 1e-8)
                policy_loss = torch.max(
                    -batch_advantage * ratio,
                    -batch_advantage * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef),
                ).mean()
                value_loss = 0.5 * (new_value.squeeze(-1) - returns[batch_indices]).pow(2).mean()
                entropy_value = entropy.mean()
                loss = policy_loss - config.entropy_coef * entropy_value + config.value_coef * value_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()
                entropy_total += entropy_value.item()
                update_count += 1

        scheduler.step()
        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        win_rate = 100.0 * episode_wins / max(1, episode_count)
        row = {
            "iteration": iteration,
            "steps": iteration * config.steps,
            "mean_reward": mean_reward,
            "train_win_rate": win_rate,
            "policy_loss": policy_loss_total / max(1, update_count),
            "value_loss": value_loss_total / max(1, update_count),
            "entropy": entropy_total / max(1, update_count),
        }

        if iteration % config.eval_frequency == 0 or iteration == 1 or iteration == config.iterations:
            evaluation = evaluate_agent(agent, config, device)
            row["evaluation"] = evaluation
            row["gauntlet_score"] = float(np.mean([item["win_rate"] for item in evaluation.values()]))
            if row["gauntlet_score"] >= best_score:
                best_score = row["gauntlet_score"]
                save_state(agent, config.checkpoint_dir / "haxball_rl_best.pt")

        metrics.append(row)
        with (config.checkpoint_dir / "training_metrics.json").open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2, ensure_ascii=False)
        dashboard.update(metrics)
        save_training_checkpoint(agent, optimizer, scheduler, iteration, metrics, latest_path)

        elapsed = time.time() - start_time
        speed = (iteration * config.steps) / max(elapsed, 1e-6)
        print(
            f"[Iter {iteration:04d}/{config.iterations}] "
            f"steps={iteration * config.steps:,} reward={mean_reward:+.3f} "
            f"train_win={win_rate:5.1f}% speed={speed:,.0f}/s"
        )
        if "evaluation" in row:
            summary = ", ".join(f"{name}={data['win_rate']:.0f}%" for name, data in row["evaluation"].items())
            print(f"  eval: {summary} | média={row['gauntlet_score']:.1f}%")

    final_path = config.checkpoint_dir / "haxball_rl_final.pt"
    save_state(agent, final_path)
    save_training_checkpoint(agent, optimizer, scheduler, current_iteration, metrics, latest_path)
    atexit.unregister(save_on_exit)
    dashboard.close()
    print(f"Treinamento concluído. Modelo final: {final_path}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", dest="map_name", default="futsal_2v2", choices=sorted(p.stem for p in MAPS.glob("*.hbs")))
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=8)
    parser.add_argument("--eval-frequency", type=int, default=25)
    parser.add_argument("--eval-matches", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.012)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--warm-start", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plot-mode",
        choices=["save", "live", "off"],
        default="save",
        help="save=PNG sem janela (padrão), live=janela atualizada, off=sem gráficos",
    )
    args = parser.parse_args()
    return Config(**vars(args))


if __name__ == "__main__":
    train(parse_args())
