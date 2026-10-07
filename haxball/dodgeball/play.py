"""
HaxBall Dodgeball Oficial: Você vs 1 ou 2 Bots.

Como Jogar:
    python haxball-dodgeball/play.py         # Você vs 1 Bot (1v1)
    python haxball-dodgeball/play.py 2       # Você vs 2 Bots (1v2 Desafio)
    python haxball-dodgeball/play.py --mode 2v2 # Você + 1 Aliado vs 2 Bots

Regras:
    1. A BOLA NÃO MATA AO TOCAR: Ela empurra o jogador com grande força cinética (knockback)!
    2. A MORTE OCORRE NA PAREDE: Se você ou o bot encostar na parede externa do próprio campo,
       morre na hora e o time adversário ganha 1 ponto!
    3. A BOLA PERDE VELOCIDADE: A cada quique na parede ela dissipa energia e fica mais lenta.

Controles:
    - W, A, S, D ou Setas: Mover
    - Barra de Espaço, X, Shift ou Ctrl: Chutar
    - R: Reiniciar partida
    - ESC: Sair
"""

from __future__ import annotations
import os
import sys
import math
import time
import argparse
from pathlib import Path

# Add haxball-dodgeball directory to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import pygame
import torch
import numpy as np

from core.vector import Vec2
from core.constants import Team, GameState
from core.stadium import Stadium
from core.dodgeball_game import DodgeballGame
from bots.dodge_bot import DodgeBot
from rl.observations import DodgeballObservationBuilder
from rl.mlp_policy import ActorCriticMLP

def parse_args():
    parser = argparse.ArgumentParser(description="HaxBall Dodgeball (Humano vs Bots)")
    parser.add_argument("bots_arg", nargs="?", type=int, default=None, help="Quantidade de bots adversários (1 ou 2)")
    parser.add_argument("--bots", type=int, default=None, help="Quantidade de bots (1 ou 2)")
    parser.add_argument("--mode", type=str, default=None, choices=["1v1", "1v2", "2v2"], help="Modo: 1v1, 1v2 ou 2v2")
    parser.add_argument("--ai", type=str, default="heuristic", choices=["heuristic", "rl"], help="Tipo de IA: heuristic ou rl")
    parser.add_argument("--score_limit", type=int, default=5, help="Pontos para vencer")
    return parser.parse_args()

def main():
    args = parse_args()

    num_bots = args.bots_arg if args.bots_arg is not None else (args.bots if args.bots is not None else 1)
    if args.mode is not None:
        mode = args.mode
    elif num_bots == 2:
        mode = "1v2"
    else:
        mode = "1v1"

    if mode == "1v1":
        red_count = 1
        blue_count = 1
        mode_desc = "1v1: VOCÊ (Red) vs 1 BOT (Blue)"
    elif mode == "1v2":
        red_count = 1
        blue_count = 2
        mode_desc = "1v2 DESAFIO: VOCÊ (Red) vs 2 BOTS (Blue)"
    else:
        red_count = 2
        blue_count = 2
        mode_desc = "2v2 EQUIPE: VOCÊ + 1 ALIADO vs 2 BOTS (Blue)"

    map_path = os.path.join(SCRIPT_DIR, "maps", "dodgeball.hbs")
    stadium = Stadium.load_from_file(map_path)

    game = DodgeballGame(
        stadium=stadium,
        score_limit=args.score_limit,
        time_limit_secs=300,
        red_players_count=red_count,
        blue_players_count=blue_count
    )

    # Configuração de IA
    rl_policy = None
    rl_obs_builder = None
    ckpt_path = os.path.join(SCRIPT_DIR.parent, "checkpoints", "dodgeball_rl_best.pt")

    if args.ai == "rl" and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        rl_obs_builder = DodgeballObservationBuilder(
            max_teammates=1,
            max_opponents=2
        )
        expected_obs_dim = ckpt.get("obs_dim", 36)
        rl_policy = ActorCriticMLP(
            obs_dim=expected_obs_dim,
            act_dim=ckpt.get("act_dim", 3),
            hidden_dim=128
        )
        rl_policy.load_state_dict(ckpt["model_state_dict"])
        rl_policy.eval()
        ai_name = "IA PPO Treinada"
    else:
        ai_name = "DodgeBot Pro (Especialista)"

    dodge_bots = {p.player_id: DodgeBot(f"Bot_{p.player_id}") for p in game.players}

    # Inicialização Pygame
    pygame.init()
    pygame.font.init()
    screen_w, screen_h = 1100, 620
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption(f"HaxBall Dodgeball Arena - {mode_desc}")

    clock = pygame.time.Clock()
    running = True

    # Transformação de coordenadas
    stad = game.stadium
    margin = 50.0
    scale = min((screen_w - margin * 2) / (stad.width * 2.0), (screen_h - margin * 2 - 50.0) / (stad.height * 2.0))
    center_x = screen_w / 2.0
    center_y = (screen_h + 50.0) / 2.0

    def world_to_screen(v: Vec2):
        return (int(center_x + v.x * scale), int(center_y - v.y * scale))

    def world_len(l: float):
        return max(1, int(round(l * scale)))

    font_score = pygame.font.SysFont("Verdana", 28, bold=True)
    font_banner = pygame.font.SysFont("Verdana", 14, bold=True)
    font_alert = pygame.font.SysFont("Verdana", 22, bold=True)
    font_sub = pygame.font.SysFont("Verdana", 12)
    font_player = pygame.font.SysFont("Arial", 13, bold=True)

    print("\n" + "="*65)
    print("🔥 HAXBALL DODGEBALL INICIADO!")
    print(f"🎮 Modo: {mode_desc}")
    print(f"🤖 Oponente: {ai_name}")
    print("-----------------------------------------------------------------")
    print("💡 MECÂNICAS DA PARTIDA (HAXBALL DODGEBALL REAL):")
    print("   • BOLA ULTRA RÁPIDA: Disparos a laser e ricochetes ágeis!")
    print("   • O JOGADOR NÃO EMPURRA A BOLA: Andar na bola não a move; só se move no CHUTE!")
    print("   • REBOTE DE ALTA PRECISÃO: Chutar a bola que vem na sua direção lança um contra-ataque feroz!")
    print("   • MORTE NA PAREDE: Se tomar knockback ou recuar na parede vermelha = eliminado!")
    print("-----------------------------------------------------------------")
    print("⌨️  Controles: WASD/Setas = Mover | Espaço/X/Shift = Chutar | R = Reset | ESC = Sair")
    print("="*65 + "\n")

    alert_text = ""
    alert_color = (255, 255, 255)
    alert_timer = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset_match()
                    alert_text = "PARTIDA REINICIADA!"
                    alert_color = (255, 215, 0)
                    alert_timer = 90

        # Input Humano (Red 1)
        keys = pygame.key.get_pressed()
        mx, my = 0.0, 0.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            mx -= 1.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            mx += 1.0
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            my += 1.0
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            my -= 1.0

        if mx != 0.0 and my != 0.0:
            mx *= 0.7071
            my *= 0.7071

        kick = (keys[pygame.K_SPACE] or keys[pygame.K_x] or keys[pygame.K_c] or keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])

        human = game.players[0]
        inputs = {}

        if game.is_alive(human.player_id):
            inputs[human.player_id] = (mx, my, kick)
        else:
            inputs[human.player_id] = (0.0, 0.0, False)

        for p in game.players[1:]:
            if not game.is_alive(p.player_id):
                inputs[p.player_id] = (0.0, 0.0, False)
                continue

            if args.ai == "rl" and rl_policy is not None and rl_obs_builder is not None:
                obs = rl_obs_builder.build_observation(game, p)
                expected_dim = rl_policy.obs_dim
                if len(obs) < expected_dim:
                    obs = np.pad(obs, (0, expected_dim - len(obs)), mode='constant')
                elif len(obs) > expected_dim:
                    obs = obs[:expected_dim]
                with torch.no_grad():
                    act, _, _, _ = rl_policy.get_action_and_value(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                act_np = act.squeeze(0).numpy()
                ori = 1.0 if p.team == Team.RED else -1.0
                bot_mx = float(np.clip(act_np[0], -1.0, 1.0)) * ori
                bot_my = float(np.clip(act_np[1], -1.0, 1.0))
                bot_kick = bool(act_np[2] > 0.0)
                inputs[p.player_id] = (bot_mx, bot_my, bot_kick)
            else:
                inputs[p.player_id] = dodge_bots[p.player_id].act(game, p)

        step_info = game.step(inputs)

        # Tratar impactos da bola e mortes na parede
        for impact in step_info.get("events", {}).get("ball_player_impacts", []):
            if impact["player_id"] == human.player_id and impact["impact_speed"] > 2.0:
                alert_text = "VOCE SOFREU KNOCKBACK DA BOLA!"
                alert_color = (255, 200, 60)
                alert_timer = 40

        for elim in step_info.get("eliminations", []):
            p_id = elim["player_id"]
            reason = elim["reason"]
            is_you = (p_id == human.player_id)

            if is_you:
                if reason == "pushed_into_wall":
                    alert_text = "VOCE FOI EMPURRADO NA PAREDE! (+1 PONTO AZUL)"
                else:
                    alert_text = "VOCE SE MATOU NA PAREDE! (+1 PONTO AZUL)"
                alert_color = (255, 60, 60)
            else:
                if reason == "pushed_into_wall":
                    alert_text = f"BOT {p_id} FOI EMPURRADO NA PAREDE! (+1 PONTO RED)"
                else:
                    alert_text = f"BOT {p_id} SE MATOU NA PAREDE! (+1 PONTO RED)"
                alert_color = (80, 255, 120)
            alert_timer = 90

        # RENDERIZAÇÃO
        screen.fill((16, 20, 26))

        # Quadra de fundo
        bg_w_s = world_len(stad.bg_width * 2)
        bg_h_s = world_len(stad.bg_height * 2)
        bg_rect = pygame.Rect(center_x - bg_w_s / 2, center_y - bg_h_s / 2, bg_w_s, bg_h_s)
        pygame.draw.rect(screen, (30, 36, 46), bg_rect)

        # Paredes Letais (Vermelhas) e Divisória (Amarela)
        for seg in stad.segments:
            p0_s = world_to_screen(seg.p0)
            p1_s = world_to_screen(seg.p1)
            col = (255, 60, 70) if seg.trait == "deadlyWall" else (255, 215, 60)
            thick = 4 if seg.trait == "deadlyWall" else 3
            pygame.draw.line(screen, col, p0_s, p1_s, width=thick)

        # Bola (com rastro e sombra)
        b_pos = world_to_screen(game.ball.pos)
        b_rad = world_len(game.ball.radius)
        pygame.draw.circle(screen, (10, 10, 15), (b_pos[0]+2, b_pos[1]+2), b_rad)
        pygame.draw.circle(screen, (255, 60, 90), b_pos, b_rad)
        pygame.draw.circle(screen, (20, 20, 20), b_pos, b_rad, width=2)
        pygame.draw.circle(screen, (255, 255, 255), b_pos, max(1, b_rad // 3))

        # Jogadores
        for p in game.players:
            p_pos = world_to_screen(p.pos)
            p_rad = world_len(p.radius)
            is_dead = not game.is_alive(p.player_id)

            if is_dead:
                p_col = (60, 65, 75)
                pygame.draw.circle(screen, p_col, p_pos, p_rad)
                pygame.draw.circle(screen, (20, 20, 20), p_pos, p_rad, width=2)
                pygame.draw.line(screen, (255, 40, 40), (p_pos[0]-p_rad, p_pos[1]-p_rad), (p_pos[0]+p_rad, p_pos[1]+p_rad), width=2)
                pygame.draw.line(screen, (255, 40, 40), (p_pos[0]-p_rad, p_pos[1]+p_rad), (p_pos[0]+p_rad, p_pos[1]-p_rad), width=2)
            else:
                p_col = (229, 110, 86) if p.team == Team.RED else (86, 137, 229)
                if p.kick_flash > 0:
                    pygame.draw.circle(screen, (255, 255, 255), p_pos, p_rad + 4, width=3)
                pygame.draw.circle(screen, p_col, p_pos, p_rad)
                pygame.draw.circle(screen, (20, 20, 20), p_pos, p_rad, width=2)
                pygame.draw.circle(screen, (255, 255, 255), p_pos, max(1, p_rad - 5), width=1)

                num_surf = font_player.render(str(p.player_number), True, (255, 255, 255))
                screen.blit(num_surf, num_surf.get_rect(center=p_pos))

            # Destaque no Humano (Aura Dourada)
            if p.player_id == human.player_id and not is_dead:
                pulse = int(abs(math.sin(time.time() * 6.0)) * 3)
                pygame.draw.circle(screen, (255, 215, 0), p_pos, p_rad + 4 + pulse, width=2)

        # Barra Superior: Placar e Timer
        pygame.draw.rect(screen, (12, 15, 20), (0, 0, screen_w, 48))
        pygame.draw.line(screen, (30, 36, 46), (0, 48), (screen_w, 48), width=2)

        r_text = font_score.render(f"RED (VOCÊ)  {game.red_score}", True, (229, 110, 86))
        b_text = font_score.render(f"{game.blue_score}  BLUE (BOT)", True, (86, 137, 229))
        vs_text = font_score.render("-", True, (160, 160, 160))
        t_text = font_score.render(game.time_string, True, (220, 220, 220))

        screen.blit(r_text, (center_x - 220, 8))
        screen.blit(vs_text, (center_x - 10, 8))
        screen.blit(b_text, (center_x + 30, 8))
        screen.blit(t_text, (screen_w - 110, 8))

        # Avisos de Regras no topo
        rule_surf = font_banner.render(
            "[!] PAREDES VERMELHAS MATAM | A BOLA EMPURRA COM FORCA", True, (255, 160, 80)
        )
        screen.blit(rule_surf, rule_surf.get_rect(center=(center_x, 62)))

        ctrl_surf = font_sub.render(
            "WASD / Setas: Mover  |  Espaco / X: Chutar  |  R: Resetar  |  ESC: Sair", True, (150, 150, 160)
        )
        screen.blit(ctrl_surf, ctrl_surf.get_rect(center=(center_x, 80)))

        # Alertas de Ação (Knockback ou Morte na Parede)
        if alert_timer > 0:
            alert_timer -= 1
            a_surf = font_alert.render(alert_text, True, alert_color)
            a_rect = a_surf.get_rect(center=(center_x, center_y - 40))
            bg_a = a_rect.inflate(26, 14)
            pygame.draw.rect(screen, (10, 12, 16), bg_a, border_radius=6)
            pygame.draw.rect(screen, alert_color, bg_a, width=2, border_radius=6)
            screen.blit(a_surf, a_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("\nPartida encerrada!\n")

if __name__ == "__main__":
    main()
