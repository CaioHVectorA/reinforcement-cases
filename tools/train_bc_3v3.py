"""
Treinamento e Avaliação de Behavioral Cloning (BC) para Futsal 3v3.
Pipeline completo:
1. Geração de dataset multiagente 3v3 com 61 dimensões contínuas e simetria Y.
2. Treinamento supervisionado da arquitetura EntityAttentionPolicy (Multi-Head Attention).
3. Avaliação de acurácia de predição de ações humanas e táticas (Top-1 e Top-3).
4. Confronto em partida real 3v3: IA Treinada (BC) vs Equipe Futsal 3v3 Finetunada.
5. Medição de métricas de jogo: Placar, Posse de Bola, Chutes e Espaçamento Médio.
"""

from __future__ import annotations
import os
import sys
import time
import math
import random

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Maximize multi-core efficiency
torch.set_num_threads(min(14, os.cpu_count() or 4))


from haxball.core.vector import Vec2
from haxball.core.constants import Team, GameState
from haxball.core.stadium import Stadium
from haxball.core.game import HaxBallGame
from haxball.bots.futsal_3v3_team import Futsal3v3Bot, Futsal3v3Coordinator
from haxball.rl.observations.decoupled_obs import DecoupledObservationBuilder
from haxball.rl.actions.action_space import ActionHandler
from haxball.rl.models.entity_attention import EntityAttentionPolicy


def mirror_y_observation(obs: np.ndarray) -> np.ndarray:
    sym = obs.copy()
    sym[1] = -sym[1]  # b_y
    sym[3] = -sym[3]  # b_vy
    sym[9] = -sym[9]  # p_y
    sym[11] = -sym[11]  # p_vy
    sym[13] = -sym[13]  # rel_b_y
    for base in range(16, len(obs), 5):
        sym[base + 1] = -sym[base + 1]  # rel_y
        sym[base + 3] = -sym[base + 3]  # vy
    return sym


def generate_3v3_futsal_dataset(
    num_matches: int = 40,
    steps_per_match: int = 600,
    augment_y: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gera um dataset rico de 3v3 em campo oficial futsal_3v3.hbs.
    Em cada tick, extrai a perspectiva ego (61 dimensões) e a ação de TODOS os 6 atletas em campo.
    """
    print(f"=== [Dataset 3v3] Gerando demonstrações em {num_matches} partidas ({steps_per_match} ticks cada) ===")
    stad_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "haxball", "maps", "futsal_3v3.hbs")
    stadium = Stadium.load_from_file(stad_path)

    obs_builder = DecoupledObservationBuilder()
    action_handler = ActionHandler()

    obs_list = []
    act_list = []

    red_coord = Futsal3v3Coordinator(Team.RED)
    blue_coord = Futsal3v3Coordinator(Team.BLUE)

    red_bots = [Futsal3v3Bot(f"Red_{i}", red_coord) for i in range(3)]
    blue_bots = [Futsal3v3Bot(f"Blue_{i}", blue_coord) for i in range(3)]

    for m in range(num_matches):
        game = HaxBallGame(stadium=stadium, red_players_count=3, blue_players_count=3)
        # Variar os cenários para enriquecer a distribuição (aberturas, escanteios, contra-ataques)
        randomize_start = (m % 3 != 0)
        game.reset_round(kickoff_team=Team.RED if m % 2 == 0 else Team.BLUE, randomize_scenario=randomize_start)

        red_players = [p for p in game.players if p.team == Team.RED]
        blue_players = [p for p in game.players if p.team == Team.BLUE]

        for step in range(steps_per_match):
            inputs_dict = {}

            # Extrai decisões dos 3 jogadores vermelhos
            for b, p in zip(red_bots, red_players):
                mx, my, kick = b.act(game, p)
                inputs_dict[p.player_id] = (mx, my, kick)
                obs = obs_builder.build_observation(game, p)
                act_idx = action_handler.encode_discrete(mx, my, kick)
                obs_list.append(obs)
                act_list.append(act_idx)

                # Aumento de Simetria Y
                if augment_y:
                    obs_sym = mirror_y_observation(obs)
                    # Espelha componente Y da ação discreta
                    is_kicking = act_idx >= 9
                    dir_part = act_idx % 9
                    x_idx = dir_part % 3
                    y_idx = dir_part // 3
                    y_sym = 2 - y_idx
                    act_sym = (y_sym * 3 + x_idx) + (9 if is_kicking else 0)
                    obs_list.append(obs_sym)
                    act_list.append(act_sym)

            # Extrai decisões dos 3 jogadores azuis
            for b, p in zip(blue_bots, blue_players):
                mx, my, kick = b.act(game, p)
                inputs_dict[p.player_id] = (mx, my, kick)
                obs = obs_builder.build_observation(game, p)
                act_idx = action_handler.encode_discrete(mx, my, kick)
                obs_list.append(obs)
                act_list.append(act_idx)

                if augment_y:
                    obs_sym = mirror_y_observation(obs)
                    is_kicking = act_idx >= 9
                    dir_part = act_idx % 9
                    x_idx = dir_part % 3
                    y_idx = dir_part // 3
                    y_sym = 2 - y_idx
                    act_sym = (y_sym * 3 + x_idx) + (9 if is_kicking else 0)
                    obs_list.append(obs_sym)
                    act_list.append(act_sym)

            game.step(inputs_dict)


        if (m + 1) % 10 == 0 or (m + 1) == num_matches:
            print(f"  Partidas concluídas: {m+1}/{num_matches} | Amostras acumuladas: {len(obs_list):,}")

    obs_tensor = torch.from_numpy(np.array(obs_list, dtype=np.float32))
    act_tensor = torch.tensor(act_list, dtype=torch.long)
    return obs_tensor, act_tensor


def train_3v3_behavioral_cloning(
    obs_tensor: torch.Tensor,
    act_tensor: torch.Tensor,
    epochs: int = 35,
    batch_size: int = 1024,
    lr: float = 1e-3
) -> Tuple[EntityAttentionPolicy, Dict[str, Any]]:
    """
    Treina a EntityAttentionPolicy com Cross-Entropy Loss em PyTorch.
    Otimizado para throughput maximo em CPU multi-core com batch_size=1024.
    Salva periodicamente e armazena o melhor modelo conforme a perda de validacao.
    """
    print(f"\n=== [Treinamento BC 3v3] Arquitetura Entity-Attention (Multi-Head Attention) ===")
    num_samples = len(obs_tensor)
    indices = np.random.permutation(num_samples)
    split = int(0.85 * num_samples)
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = TensorDataset(obs_tensor[train_idx], act_tensor[train_idx])
    val_ds = TensorDataset(obs_tensor[val_idx], act_tensor[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = EntityAttentionPolicy(embed_dim=64, num_heads=4, act_dim=18, is_discrete=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "top1_acc": [], "top3_acc": []}

    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "bc_futsal_3v3.pt")

    print(f"Amostras de Treino: {len(train_ds):,} | Amostras de Validacao: {len(val_ds):,}")
    print(f"Total de Parametros Neurais: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Configuracao: {epochs} epocas | batch_size={batch_size} | threads={torch.get_num_threads()}\n")

    best_val_loss = float("inf")
    start_time = time.time()

    for ep in range(epochs):
        ep_start = time.time()
        model.train()
        total_loss = 0.0
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = model.actor(model.forward_repr(x_b))
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(x_b)

        scheduler.step()
        train_loss = total_loss / len(train_ds)

        # Validacao
        model.eval()
        val_loss = 0.0
        correct_top1 = 0
        correct_top3 = 0

        with torch.no_grad():
            for x_b, y_b in val_loader:
                logits = model.actor(model.forward_repr(x_b))
                loss = criterion(logits, y_b)
                val_loss += loss.item() * len(x_b)

                preds = torch.argmax(logits, dim=-1)
                correct_top1 += (preds == y_b).sum().item()

                _, top3 = torch.topk(logits, k=3, dim=-1)
                correct_top3 += (top3 == y_b.unsqueeze(1)).any(dim=-1).sum().item()

        val_loss /= len(val_ds)
        top1_acc = 100.0 * correct_top1 / len(val_ds)
        top3_acc = 100.0 * correct_top3 / len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["top1_acc"].append(top1_acc)
        history["top3_acc"].append(top3_acc)

        ep_duration = time.time() - ep_start
        total_elapsed = time.time() - start_time
        remaining_eps = epochs - (ep + 1)
        eta_secs = remaining_eps * ep_duration

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

        # Snapshot a cada 5 epocas
        if (ep + 1) % 5 == 0 or (ep + 1) == epochs:
            snap_path = os.path.join(save_dir, f"bc_futsal_3v3_ep{ep+1}.pt")
            torch.save(model.state_dict(), snap_path)

        star = "*" if is_best else " "
        print(f"[{ep+1:02d}/{epochs:02d}]{star} Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Top-1: {top1_acc:.2f}% | Top-3: {top3_acc:.2f}% | Ep: {ep_duration:.1f}s | Elapsed: {total_elapsed/60.0:.1f}m | ETA: {eta_secs/60.0:.1f}m")

    total_time = time.time() - start_time
    print(f"\nTreinamento concluido em {total_time/60.0:.2f} minutos ({total_time:.1f}s)!")
    print(f"Melhor modelo salvo em: {ckpt_path} (Best Val Loss: {best_val_loss:.4f})")

    # Garante que o modelo retornado tenha os melhores pesos
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model, history


def evaluate_3v3_match(
    model: EntityAttentionPolicy,
    duration_ticks: int = 3600 # 60 segundos de tempo real a 60 FPS
) -> Dict[str, Any]:
    """
    Executa uma partida completa 3v3 oficial:
    Time Vermelho: 3 instâncias controladas pela IA BC (EntityAttentionPolicy).
    Time Azul: 3 instâncias do time de bots analíticos coordenados (Futsal3v3Coordinator).
    """
    print(f"\n=== [Avaliação em Partida 3v3 Real] IA BC (Red) vs Bots Coordenados (Blue) ===")
    print(f"Duração da simulação: {duration_ticks} ticks ({duration_ticks/60.0:.1f} segundos)")

    stad_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "haxball", "maps", "futsal_3v3.hbs")
    stadium = Stadium.load_from_file(stad_path)
    game = HaxBallGame(stadium=stadium, score_limit=10, time_limit_secs=180, red_players_count=3, blue_players_count=3)

    obs_builder = DecoupledObservationBuilder()
    action_handler = ActionHandler()

    blue_coord = Futsal3v3Coordinator(Team.BLUE)
    blue_bots = [Futsal3v3Bot(f"Blue_{i}", blue_coord) for i in range(3)]

    red_players = [p for p in game.players if p.team == Team.RED]
    blue_players = [p for p in game.players if p.team == Team.BLUE]

    model.eval()

    red_possession_ticks = 0
    blue_possession_ticks = 0
    red_shots = 0
    blue_shots = 0
    team_spacings = []

    inference_times = []

    for tick in range(duration_ticks):
        inputs_dict = {}

        # 1. Decisão da IA BC para os 3 atletas do Time Vermelho
        t0 = time.perf_counter()
        for p in red_players:
            obs = obs_builder.build_observation(game, p)
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            with torch.no_grad():
                logits = model.actor(model.forward_repr(obs_t))
                act_idx = torch.argmax(logits, dim=-1).item()
            mx, my, kick = action_handler.decode_discrete(act_idx)
            inputs_dict[p.player_id] = (mx, my, kick)

            if kick and p.pos.distance_to(game.ball.pos) < 30.0:
                red_shots += 1
        t_infer = (time.perf_counter() - t0) * 1000.0 / 3.0
        inference_times.append(t_infer)

        # 2. Decisão dos Bots para os 3 atletas do Time Azul
        for b, p in zip(blue_bots, blue_players):
            mx, my, kick = b.act(game, p)
            inputs_dict[p.player_id] = (mx, my, kick)
            if kick and p.pos.distance_to(game.ball.pos) < 30.0:
                blue_shots += 1

        # Estatística de posse (jogador mais próximo da bola a < 80 px)
        closest_p = min(game.players, key=lambda p: p.pos.distance_to(game.ball.pos))
        if closest_p.pos.distance_to(game.ball.pos) < 80.0:
            if closest_p.team == Team.RED:
                red_possession_ticks += 1
            elif closest_p.team == Team.BLUE:
                blue_possession_ticks += 1

        # Medição de espaçamento mútuo entre companheiros (Red)
        d01 = red_players[0].pos.distance_to(red_players[1].pos)
        d02 = red_players[0].pos.distance_to(red_players[2].pos)
        d12 = red_players[1].pos.distance_to(red_players[2].pos)
        avg_spacing = (d01 + d02 + d12) / 3.0
        team_spacings.append(avg_spacing)

        # Avança a física
        game.step(inputs_dict)

    total_poss = max(1, red_possession_ticks + blue_possession_ticks)
    red_poss_pct = 100.0 * red_possession_ticks / total_poss
    blue_poss_pct = 100.0 * blue_possession_ticks / total_poss
    mean_spacing = float(np.mean(team_spacings))
    mean_latency = float(np.mean(inference_times))

    results = {
        "final_score": f"{game.red_score} x {game.blue_score}",
        "red_score": game.red_score,
        "blue_score": game.blue_score,
        "red_possession_pct": red_poss_pct,
        "blue_possession_pct": blue_poss_pct,
        "red_shots": red_shots,
        "blue_shots": blue_shots,
        "avg_teammate_spacing_px": mean_spacing,
        "mean_inference_latency_ms": mean_latency
    }

    print("\n" + "="*50)
    print("=== RELATÓRIO DO CONFRONTO 3v3 ===")
    print("="*50)
    print(f"Placar Final: IA BC (Red) {game.red_score} x {game.blue_score} Bots Coordenados (Blue)")
    print(f"Posse de Bola: Red {red_poss_pct:.1f}% | Blue {blue_poss_pct:.1f}%")
    print(f"Chutes Tentados: Red {red_shots} | Blue {blue_shots}")
    print(f"Espaçamento Médio do Time Red: {mean_spacing:.1f} px (Anti-clustering ativo!)")
    print(f"Latência Média de Inferência: {mean_latency:.2f} ms por jogador")
    print("="*50)

    return results


def main():
    print("=" * 60)
    print("=== INICIANDO TREINO EXPANDIDO DE BEHAVIORAL CLONING (3v3 FUTSAL) ===")
    print("Meta de duracao: ~40 minutos | Volume de dados: ~2.16 milhoes de amostras")
    print("=" * 60 + "\n")

    # 1. Gerar dataset 3v3 (300 partidas com cenarios ricos e simetria Y)
    obs_t, act_t = generate_3v3_futsal_dataset(num_matches=300, steps_per_match=600, augment_y=True)

    # 2. Treinar politica de Auto-Atencao com BC por 35 epocas em batch_size=1024
    model, history = train_3v3_behavioral_cloning(obs_t, act_t, epochs=35, batch_size=1024, lr=1e-3)

    # 3. Avaliar confronto 3v3 no motor de fisica
    eval_results = evaluate_3v3_match(model, duration_ticks=3600)
    print("\n=== TREINO E AVALIACAO CONCLUIDOS COM SUCESSO! ===")

if __name__ == "__main__":
    main()

