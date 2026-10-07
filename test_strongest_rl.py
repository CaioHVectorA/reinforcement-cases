"""
Teste Imediato do Modelo RL Mais Forte em Treinamento.

Permite testar o melhor checkpoint de Aprendizado por Reforço ('models/checkpoints/haxball_rl_best.pt')
em tempo real contra você e/ou contra bots táticos.

Modos Disponíveis (selecione ao rodar ou passe como argumento):
1. Duelo 1x1: Você vs IA RL Mais Forte
2. Trio 3x3: Você + 2 IAs RL Mais Fortes vs 3 Bots Táticos
3. Duelo IA vs IA: IA RL Mais Forte vs IA Behavioral Cloning (BC)

Controles:
- WASD ou Setas: Mover
- Barra de Espaço, X, C ou Shift: Chutar
- R: Reiniciar rodada (recarrega os pesos mais recentes salvos pelo treino em background!)
- ESC: Sair
"""

from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from haxball.ui.haxball_gui import HaxBallStudioApp
from haxball.core.constants import Team


def run_strongest_test(mode: str = "duelo"):
    app = HaxBallStudioApp()

    if mode == "1v1" or mode == "duelo":
        # Formato 1v1
        app.set_format(1)
        # Red 1: Humano, Blue 1: IA RL
        app.slot_controllers[Team.RED][0] = "human"
        app.slot_controllers[Team.BLUE][0] = "ppo_rl"
        print("\n" + "="*60)
        print("=== DUELO 1v1 CONTRA A IA RL MAIS FORTE ===")
        print("🔴 Vermelho: VOCÊ (WASD / Setas + Espaço/Shift/X)")
        print("🔵 Azul:     IA RL Mais Forte (Treinamento PPO)")
        print("Pressione 'R' para recarregar pesos mais novos a qualquer momento!")
        print("="*60 + "\n")

    elif mode == "3v3_trio":
        app.set_format(3)
        # Red 1: Humano, Red 2, 3: IA RL
        app.slot_controllers[Team.RED] = ["human", "ppo_rl", "ppo_rl"]
        # Blue: Bots táticos
        app.slot_controllers[Team.BLUE] = ["bot_fixo", "bot_ala", "bot_press"]
        print("\n" + "="*60)
        print("=== PARTIDA 3v3 COM SUAS IAs RL MAIS FORTES ===")
        print("🔴 Vermelho: Você + 2 IAs RL Potentes")
        print("🔵 Azul:     Trio de Bots Táticos (Fixo, Ala, Press)")
        print("Pressione 'R' para recarregar pesos mais novos a qualquer momento!")
        print("="*60 + "\n")

    elif mode == "rl_vs_bc":
        app.set_format(1)
        # Red: IA RL, Blue: IA BC
        app.slot_controllers[Team.RED][0] = "ppo_rl"
        app.slot_controllers[Team.BLUE][0] = "bc_ai"
        print("\n" + "="*60)
        print("=== DUELO: IA RL MAIS FORTE vs IA BEHAVIORAL CLONING (BC) ===")
        print("🔴 Vermelho: IA RL Mais Forte (Treinamento PPO)")
        print("🔵 Azul:     IA BC (Behavioral Cloning Replays)")
        print("Pressione 'R' para recarregar pesos mais novos a qualquer momento!")
        print("="*60 + "\n")

    app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testar o agente RL mais forte atual")
    parser.add_argument(
        "--mode",
        choices=["1v1", "duelo", "3v3_trio", "rl_vs_bc"],
        default="1v1",
        help="Modo de teste: 1v1 (duelo direto), 3v3_trio (você + 2 IAs RL), ou rl_vs_bc (IA vs IA)"
    )
    args = parser.parse_args()
    run_strongest_test(args.mode)
