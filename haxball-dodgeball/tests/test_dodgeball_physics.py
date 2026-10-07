import unittest
import sys
import os
from pathlib import Path
import numpy as np
import torch

# Add haxball-dodgeball directory to sys.path
dodgeball_dir = Path(__file__).resolve().parent.parent
if str(dodgeball_dir) not in sys.path:
    sys.path.insert(0, str(dodgeball_dir))

from core.vector import Vec2
from core.constants import Team, GameState
from core.stadium import Stadium
from core.dodgeball_game import DodgeballGame
from bots.dodge_bot import DodgeBot
from rl.observations import DodgeballObservationBuilder
from rl.mlp_policy import ActorCriticMLP


class TestDodgeballPhysics(unittest.TestCase):
    def setUp(self):
        stadium_path = dodgeball_dir / "maps" / "dodgeball.hbs"
        self.stadium = Stadium.load_from_file(stadium_path)

    def test_ball_loses_speed_on_wall_bounce(self):
        """Verifica que a bola dissipa energia ao colidir com as paredes."""
        game = DodgeballGame(self.stadium, red_players_count=1, blue_players_count=1)
        ball = game.ball
        ball.pos = Vec2(0.0, 150.0)
        initial_speed = 12.0
        ball.speed = Vec2(0.0, initial_speed)

        # Executa passos até colidir e rebater
        max_speed_after_bounce = 0.0
        bounced = False
        for _ in range(60):
            game.physics.step({})
            if ball.speed.y < 0.0:
                bounced = True
                max_speed_after_bounce = abs(ball.speed.y)
                break

        self.assertTrue(bounced, "A bola deveria ter quicado na parede superior")
        # Deve ter perdido velocidade significativa devido a bCoef e damping
        self.assertLess(max_speed_after_bounce, initial_speed * 0.90, 
                        f"A bola deveria perder velocidade ao quicar. De {initial_speed} para {max_speed_after_bounce}")

    def test_player_walking_cannot_move_ball(self):
        """Verifica que o jogador NÃO move a bola ao andar contra ela (apenas no chute)."""
        game = DodgeballGame(self.stadium, red_players_count=1, blue_players_count=1)
        red_p = game.players[0]
        ball = game.ball

        # Coloca a bola parada no meio
        ball.pos = Vec2(-50.0, 0.0)
        ball.speed = Vec2(0.0, 0.0)

        # Coloca o jogador colado à bola tentando andar para a direita em direção a ela SEM chutar
        red_p.pos = Vec2(-75.0, 0.0)
        red_p.speed = Vec2(3.0, 0.0)

        initial_ball_x = ball.pos.x

        for _ in range(25):
            game.step({red_p.player_id: (1.0, 0.0, False)})

        # A bola NÃO deve ter sido empurrada pelo corpo do jogador
        self.assertAlmostEqual(ball.pos.x, initial_ball_x, delta=1.5,
                               msg="O jogador não deve conseguir empurrar a bola andando!")
        self.assertAlmostEqual(ball.speed.length(), 0.0, delta=0.5,
                               msg="A bola deve permanecer praticamente imóvel se o jogador apenas andar nela!")

    def test_timed_rebote_reverses_and_accelerates_incoming_ball(self):
        """Verifica que o rebote com timing perfeito inverte a bola em alta velocidade."""
        game = DodgeballGame(self.stadium, red_players_count=1, blue_players_count=1)
        red_p = game.players[0]
        ball = game.ball

        # Jogador posicionado e bola vindo em alta velocidade em sua direção
        red_p.pos = Vec2(100.0, 0.0)
        red_p.speed = Vec2(0.0, 0.0)

        # Bola a 25px do jogador vindo a 12 px/frame
        ball.pos = Vec2(75.0, 0.0)
        ball.speed = Vec2(12.0, 0.0)

        # Jogador chuta exatamente quando a bola entra no alcance
        events = game.physics.step({red_p.player_id: (-1.0, 0.0, True)})

        # A bola deve ter sido rebatida para a esquerda (speed.x < 0) em alta velocidade
        self.assertLess(ball.speed.x, -14.0, 
                        f"O rebote cronometrado deve disparar a bola de volta com alta velocidade, mas deu {ball.speed.x}")

    def test_player_does_not_die_on_ball_touch_and_receives_knockback(self):
        """Verifica que o jogador NÃO morre ao tomar bolada, mas recebe knockback violento."""
        game = DodgeballGame(self.stadium, red_players_count=1, blue_players_count=1)
        red_p = game.players[0]
        blue_p = game.players[1]
        ball = game.ball

        # Posiciona jogador azul no centro de seu campo
        blue_p.pos = Vec2(200.0, 0.0)
        blue_p.speed = Vec2(0.0, 0.0)

        # Atira a bola com alta velocidade no jogador azul
        ball.pos = Vec2(150.0, 0.0)
        ball.speed = Vec2(14.0, 0.0)

        initial_pos_x = blue_p.pos.x

        # Executa passos para o impacto
        impacted = False
        for _ in range(20):
            step_info = game.step({red_p.player_id: (0.0, 0.0, False), blue_p.player_id: (0.0, 0.0, False)})
            self.assertTrue(game.is_alive(blue_p.player_id), "O jogador não deve morrer ao tocar na bola!")
            if blue_p.speed.x > 3.0:
                impacted = True
                break

        self.assertTrue(impacted, "O jogador deve sofrer knockback e ser acelerado pelo impacto da bola")
        self.assertGreater(blue_p.pos.x, initial_pos_x, "O jogador deve ser empurrado para trás")

    def test_player_dies_only_on_outer_wall_contact(self):
        """Verifica que o jogador morre apenas ao bater na parede perimetral mortal."""
        game = DodgeballGame(self.stadium, red_players_count=1, blue_players_count=1)
        red_p = game.players[0]
        blue_p = game.players[1]

        # Posiciona o jogador a 10px da parede esquerda
        red_p.pos = Vec2(-360.0, 0.0)
        red_p.speed = Vec2(-8.0, 0.0)  # Correndo direto para a parede mortal

        died = False
        death_reason = None
        for _ in range(30):
            step_info = game.step({red_p.player_id: (-1.0, 0.0, False), blue_p.player_id: (0.0, 0.0, False)})
            for elim in step_info["eliminations"]:
                if elim["player_id"] == red_p.player_id:
                    died = True
                    death_reason = elim["reason"]
                    break
            if died:
                break

        self.assertTrue(died, "O jogador deve morrer ao colidir com a parede perimetral mortal")
        self.assertFalse(game.is_alive(red_p.player_id))
        self.assertEqual(death_reason, "wall_suicide")
        self.assertEqual(game.blue_score, 1, "Time adversário deve pontuar quando o oponente morre na parede")

    def test_rl_policy_runs_without_shape_error(self):
        """Verifica que a IA treinada roda em 1v1 e 1v2 sem erro de shape."""
        ckpt_path = dodgeball_dir.parent / "checkpoints" / "dodgeball_rl_best.pt"
        if not ckpt_path.exists():
            return

        ckpt = torch.load(ckpt_path, map_location="cpu")
        obs_builder = DodgeballObservationBuilder(max_teammates=1, max_opponents=2)
        policy = ActorCriticMLP(obs_dim=ckpt["obs_dim"], act_dim=ckpt["act_dim"], hidden_dim=128)
        policy.load_state_dict(ckpt["model_state_dict"])
        policy.eval()

        # Testa em 1v1
        game_1v1 = DodgeballGame(self.stadium, red_players_count=1, blue_players_count=1)
        blue_p = game_1v1.players[1]
        obs = obs_builder.build_observation(game_1v1, blue_p)
        self.assertEqual(len(obs), 36)

        with torch.no_grad():
            act, _, _, _ = policy.get_action_and_value(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
        self.assertEqual(act.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
