"""
Partida Interativa 3v3 Futsal: Você + 2 IAs Behavioral Cloning vs 3 Bots Coordenados.

Configuração:
🔴 TIME VERMELHO:
   - Jogador 1 (Red 1): VOCÊ (Humano via Teclado: WASD / Setas + Espaço / X / Shift)
   - Jogador 2 (Red 2): IA Behavioral Cloning (EntityAttentionPolicy)
   - Jogador 3 (Red 3): IA Behavioral Cloning (EntityAttentionPolicy)

🔵 TIME AZUL:
   - Jogadores 1, 2 e 3: Trio de Bots Táticos Coordenados (Fixo, Ala, Pressionador)

Estádio: Futsal 3x3 Oficial (futsal_3v3.hbs)
Controles:
   - W, A, S, D ou Setas: Mover
   - Barra de Espaço, X, C ou Shift: Chutar
   - R: Reiniciar rodada
   - ESC: Sair
"""

from __future__ import annotations
import os
import sys
import math
import time

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pygame
import torch
import numpy as np

from haxball.core.vector import Vec2
from haxball.core.constants import Team, GameState
from haxball.core.stadium import Stadium
from haxball.core.game import HaxBallGame
from haxball.bots.futsal_3v3_team import Futsal3v3Bot, Futsal3v3Coordinator
from haxball.rl.observations.decoupled_obs import DecoupledObservationBuilder
from haxball.rl.actions.action_space import ActionHandler
from haxball.rl.models.entity_attention import EntityAttentionPolicy
from haxball.renderer.pygame_renderer import PygameRenderer


def run_3v3_human_match():
    pygame.init()
    pygame.font.init()

    # 1. Carregar estádio oficial Futsal 3v3
    stad_path = os.path.join(project_root, "haxball", "maps", "futsal_3v3.hbs")
    stadium = Stadium.load_from_file(stad_path)

    # 2. Criar jogo com 3 Red e 3 Blue
    game = HaxBallGame(
        stadium=stadium,
        score_limit=5,
        time_limit_secs=300,
        red_players_count=3,
        blue_players_count=3
    )

    # 3. Carregar modelo treinado de BC para os colegas de time Red
    ckpt_path = os.path.join(project_root, "models", "checkpoints", "bc_futsal_3v3.pt")
    bc_model = EntityAttentionPolicy(embed_dim=64, num_heads=4, act_dim=18, is_discrete=True)

    if os.path.exists(ckpt_path):
        bc_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"[IA 3v3] Modelo Behavioral Cloning carregado com sucesso de: {ckpt_path}")
    else:
        print(f"[Aviso] Checkpoint {ckpt_path} não encontrado, usando pesos padrão.")

    bc_model.eval()

    obs_builder = DecoupledObservationBuilder()
    action_handler = ActionHandler()

    # 4. Configurar time adversário (Azul) com Bots Coordenados
    blue_coord = Futsal3v3Coordinator(Team.BLUE)
    blue_bots = [Futsal3v3Bot(f"Bot_Blue_{i+1}", blue_coord) for i in range(3)]

    # 5. Inicializar Renderer Pygame
    renderer = PygameRenderer(game, width=1100, height=620)
    pygame.display.set_caption("HaxBall 3v3 Futsal - Você + 2 IAs (BC) vs 3 Bots Táticos")

    clock = pygame.time.Clock()
    running = True

    # Fontes extras para HUD
    font_hud = pygame.font.SysFont("Arial", 15, bold=True)
    font_badge = pygame.font.SysFont("Arial", 12, bold=True)

    print("\n" + "="*60)
    print("=== PARTIDA 3v3 INICIADA ===")
    print("[RED 1] Voce controla o Red 1 (Jogador Vermelho com anel de destaque amarelo)")
    print("[RED 2, 3] Suas 2 IAs treinadas por Behavioral Cloning (Multi-Head Attention)")
    print("[BLUE 1, 2, 3] Time adversario de bots coordenados (Fixo + Ala + Press)")
    print("Controles: WASD ou Setas = Mover | Espaco/Shift/X = Chutar | R = Reset | ESC = Sair")
    print("="*60 + "\n")

    while running:
        # 1. Tratar eventos de janela
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset_match()

        # 2. Capturar inputs do jogador humano (Red 0 / Red 1)
        keys = pygame.key.get_pressed()
        mx = 0.0
        my = 0.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            mx -= 1.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            mx += 1.0
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            my += 1.0  # +Y é para cima no plano de física
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            my -= 1.0  # -Y é para baixo

        if mx != 0.0 and my != 0.0:
            mx *= 0.7071
            my *= 0.7071

        kick = (
            keys[pygame.K_SPACE] or
            keys[pygame.K_x] or
            keys[pygame.K_c] or
            keys[pygame.K_LSHIFT] or
            keys[pygame.K_RSHIFT]
        )

        red_players = [p for p in game.players if p.team == Team.RED]
        blue_players = [p for p in game.players if p.team == Team.BLUE]

        inputs_dict = {}

        # Humano controla red_players[0]
        if red_players:
            inputs_dict[red_players[0].player_id] = (mx, my, kick)

        # IAs BC controlam red_players[1] e red_players[2]
        with torch.no_grad():
            for p in red_players[1:]:
                obs = obs_builder.build_observation(game, p)
                obs_t = torch.from_numpy(obs).unsqueeze(0)
                logits = bc_model.actor(bc_model.forward_repr(obs_t))
                act_idx = torch.argmax(logits, dim=-1).item()
                ai_mx, ai_my, ai_kick = action_handler.decode_discrete(act_idx)
                inputs_dict[p.player_id] = (ai_mx, ai_my, ai_kick)

        # Bots Coordenados controlam o Time Azul
        for b, p in zip(blue_bots, blue_players):
            inputs_dict[p.player_id] = b.act(game, p)

        # 3. Avançar um passo de física
        game.step(inputs_dict)

        # 4. Renderizar campo e entidades
        renderer.render()

        # 5. Desenhar HUD extra e indicadores sobre as cabeças dos atletas
        # Anel de destaque sobre o jogador humano
        if red_players:
            hx, hy = renderer.world_to_screen(red_players[0].pos)
            pygame.draw.circle(renderer.screen, (255, 235, 50), (hx, hy), int(red_players[0].radius * renderer.scale + 6), width=2)
            lbl = font_badge.render("VOCE", True, (255, 235, 50))
            renderer.screen.blit(lbl, (hx - lbl.get_width() // 2, hy - 28))

        # Indicador sobre as IAs aliadas
        for i, p in enumerate(red_players[1:]):
            ix, iy = renderer.world_to_screen(p.pos)
            lbl = font_badge.render(f"IA {i+1}", True, (255, 120, 120))
            renderer.screen.blit(lbl, (ix - lbl.get_width() // 2, iy - 26))

        # Indicador sobre os bots adversários
        for i, p in enumerate(blue_players):
            bx, by = renderer.world_to_screen(p.pos)
            lbl = font_badge.render(f"BOT {i+1}", True, (120, 180, 255))
            renderer.screen.blit(lbl, (bx - lbl.get_width() // 2, by - 26))

        # Painel Superior Informativo
        hud_bar = pygame.Surface((renderer.width, 32), pygame.SRCALPHA)
        hud_bar.fill((10, 12, 16, 220))
        renderer.screen.blit(hud_bar, (0, 0))

        t_red = font_hud.render("[RED] VOCE + 2 IAs (BC)", True, (255, 100, 100))
        t_vs = font_hud.render("VS", True, (200, 200, 200))
        t_blue = font_hud.render("[BLUE] 3 BOTS COORDENADOS", True, (100, 170, 255))
        t_ctrl = font_badge.render("WASD/Setas: Mover | Espaco/Shift: Chutar | R: Reset | ESC: Sair", True, (180, 180, 180))

        renderer.screen.blit(t_red, (20, 7))
        renderer.screen.blit(t_vs, (300, 7))
        renderer.screen.blit(t_blue, (340, 7))
        renderer.screen.blit(t_ctrl, (renderer.width - t_ctrl.get_width() - 20, 9))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("\nPartida encerrada. Obrigado por avaliar o agente!")



if __name__ == "__main__":
    run_3v3_human_match()
