"""
Unit and integration tests for the HaxBall simulation engine,
official maps, physics, decoupled observations, attention policy, DQN, and Gymnasium environment.
"""

import os
import unittest
import numpy as np
import torch

from haxball.core.vector import Vec2
from haxball.core.constants import Team, GameState, CollisionMask
from haxball.core.disc import Disc
from haxball.core.segment import Vertex, Segment
from haxball.core.stadium import Stadium
from haxball.core.physics_engine import PhysicsEngine
from haxball.core.game import HaxBallGame
from haxball.bots import HeuristicBot, WallReboundBot, GoalieBot
from haxball.gym_env import HaxBallEnv
from haxball.rl.observations.decoupled_obs import DecoupledObservationBuilder
from haxball.rl.actions.action_space import ActionHandler
from haxball.rl.models.entity_attention import EntityAttentionPolicy
from haxball.rl.algorithms.standard_rl.dqn_trainer import DQNTrainer

MAP_DIR = os.path.join(os.path.dirname(__file__), "..", "haxball", "maps")

class TestVector(unittest.TestCase):
    def test_vector_operations(self):
        v1 = Vec2(3.0, 4.0)
        self.assertAlmostEqual(v1.length(), 5.0)
        self.assertAlmostEqual(v1.length_sq(), 25.0)

        norm = v1.normalized()
        self.assertAlmostEqual(norm.length(), 1.0)
        self.assertAlmostEqual(norm.x, 0.6)
        self.assertAlmostEqual(norm.y, 0.8)

        v2 = Vec2(1.0, 2.0)
        v_sum = v1 + v2
        self.assertEqual(v_sum, Vec2(4.0, 6.0))

        dot = v1.dot(v2)
        self.assertAlmostEqual(dot, 11.0)

        cross = v1.cross(v2)
        self.assertAlmostEqual(cross, 2.0)

class TestOfficialMaps(unittest.TestCase):
    def test_load_futsal_3v3_official(self):
        path = os.path.join(MAP_DIR, "futsal_3v3.hbs")
        stad = Stadium.load_from_file(path)
        self.assertIn("Futsal", stad.name)
        self.assertEqual(len(stad.goals), 2)
        self.assertAlmostEqual(stad.ball_physics["radius"], 6.3)
        self.assertAlmostEqual(stad.ball_physics["invMass"], 1.5)

    def test_load_futsal_5v5_official(self):
        path = os.path.join(MAP_DIR, "futsal_5v5.hbs")
        stad = Stadium.load_from_file(path)
        self.assertIn("Futsal", stad.name)
        self.assertEqual(stad.width, 1080)
        self.assertEqual(stad.height, 532)

class TestDecoupledObservation(unittest.TestCase):
    def test_decoupled_obs_dimensions(self):
        stad = Stadium.load_from_file(os.path.join(MAP_DIR, "futsal_3v3.hbs"))
        game = HaxBallGame(stadium=stad, red_players_count=3, blue_players_count=3)
        builder = DecoupledObservationBuilder()

        red_player = next(p for p in game.players if p.team == Team.RED)
        obs = builder.build_observation(game, red_player)

        self.assertEqual(obs.shape, (builder.obs_dim,))
        self.assertEqual(builder.obs_dim, 61)
        self.assertIsInstance(obs, np.ndarray)

    def test_entity_attention_forward(self):
        builder = DecoupledObservationBuilder()
        policy = EntityAttentionPolicy(embed_dim=32, num_heads=2, act_dim=3, is_discrete=False)
        dummy_obs = torch.randn(2, builder.obs_dim)

        act, logprob, entropy, val = policy.get_action_and_value(dummy_obs)
        self.assertEqual(act.shape, (2, 3))
        self.assertEqual(val.shape, (2, 1))

class TestActionHandler(unittest.TestCase):
    def test_action_conversion(self):
        handler = ActionHandler()
        # Test Direita (mx=1.0, my=0.0) -> index 5
        act_idx = handler.encode_discrete(1.0, 0.0, False)
        self.assertEqual(act_idx, 5)
        mx, my, kick = handler.decode_discrete(5)  # Direita
        self.assertAlmostEqual(mx, 1.0)
        self.assertAlmostEqual(my, 0.0)
        self.assertFalse(kick)

        # Test Esquerda (mx=-1.0, my=0.0) -> index 3
        act_left = handler.encode_discrete(-1.0, 0.0, False)
        self.assertEqual(act_left, 3)
        mx_l, my_l, _ = handler.decode_discrete(3)
        self.assertAlmostEqual(mx_l, -1.0)

        headless_cmd = handler.to_haxball_headless_format(1.0, 0.0, True)
        self.assertEqual(headless_cmd["xdir"], 1)
        self.assertTrue(headless_cmd["kick"])

class TestPhysicsAndKicking(unittest.TestCase):
    def test_kicking_mechanic(self):
        path = os.path.join(MAP_DIR, "futsal_3v3.hbs")
        stad = Stadium.load_from_file(path)
        game = HaxBallGame(stadium=stad)

        red = next(p for p in game.players if p.team == Team.RED)
        red.pos = Vec2(-20.0, 0.0)
        game.ball.pos = Vec2(0.0, 0.0)
        game.ball.speed = Vec2(0.0, 0.0)

        # Red kicks
        inputs = {red.player_id: (0.0, 0.0, True)}
        game.step(inputs)

        self.assertGreater(game.ball.speed.x, 2.0)
        self.assertTrue(red.kick_flash > 0)

    def test_goal_scoring_official(self):
        path = os.path.join(MAP_DIR, "futsal_3v3.hbs")
        stad = Stadium.load_from_file(path)
        game = HaxBallGame(stadium=stad)

        # In Haxball: goal at +556.3 is defended by Blue, so entering it gives a point to RED!
        game.ball.pos = Vec2(550.0, 0.0)
        game.ball.speed = Vec2(15.0, 0.0)
        initial_score = game.red_score

        step_info = game.step({})
        self.assertTrue(step_info["goal_scored"])
        self.assertEqual(step_info["scoring_team"], Team.RED)
        self.assertEqual(game.red_score, initial_score + 1)

class TestTeamPlayAndSelfPlay(unittest.TestCase):
    def test_expanded_maps_load(self):
        maps_to_test = ["futsal_2v2.hbs", "micro_1v1.hbs", "big_stadium.hbs", "small_classic.hbs"]
        for m in maps_to_test:
            path = os.path.join(MAP_DIR, m)
            stad = Stadium.load_from_file(path)
            self.assertGreater(stad.width, 0)
            self.assertGreater(stad.height, 0)
            self.assertEqual(len(stad.goals), 2)

    def test_team_play_reward_engine(self):
        from haxball.rl.rewards.reward_engine import TeamPlayRewardEngine
        stad = Stadium.load_from_file(os.path.join(MAP_DIR, "futsal_2v2.hbs"))
        game = HaxBallGame(stadium=stad, red_players_count=2, blue_players_count=2)
        engine = TeamPlayRewardEngine()
        engine.reset(game)

        # Place 2 red players very close to trigger anti-clustering spacing penalty
        p1, p2 = [p for p in game.players if p.team == Team.RED]
        p1.pos = Vec2(-50.0, 0.0)
        p2.pos = Vec2(-40.0, 0.0)  # Dist = 10 px (< 75 px threshold)
        game.ball.pos = Vec2(0.0, 0.0)

        step_info = game.step({})
        rewards = engine.compute_team_rewards(game, step_info)
        # Should have crowding penalty
        self.assertIn(p1.player_id, rewards)
        self.assertIn(p2.player_id, rewards)

    def test_self_play_trainer_steps_and_reset(self):
        from haxball.rl.algorithms.self_play.self_play_trainer import SelfPlay2v2Trainer
        stad = Stadium.load_from_file(os.path.join(MAP_DIR, "futsal_2v2.hbs"))
        game = HaxBallGame(stadium=stad, red_players_count=2, blue_players_count=2)
        trainer = SelfPlay2v2Trainer(game=game, rollout_steps=32)

        # Multi-step acceleration test
        stats = trainer.step_multistep(10)
        self.assertEqual(stats["total_steps"], 10)

        # Reset to dumb
        trainer.reset_policy_to_random()
        self.assertEqual(trainer.total_env_steps, 0)
        self.assertEqual(trainer.last_stats["total_steps"], 0)

    def test_rl_bot_team_symmetry(self):
        from haxball.bots.rl_bot import RLBot
        stad = Stadium.load_from_file(os.path.join(MAP_DIR, "futsal_2v2.hbs"))
        game = HaxBallGame(stadium=stad, red_players_count=1, blue_players_count=1)

        bot = RLBot()
        # Mock policy actor to always output forward action (+1, 0, 0) in ego frame
        class MockActor(torch.nn.Module):
            def forward(self, x):
                return torch.tensor([[1.0, 0.0, 0.0]])
        bot.policy.actor = MockActor()

        red_player = next(p for p in game.players if p.team == Team.RED)
        blue_player = next(p for p in game.players if p.team == Team.BLUE)

        red_mx, red_my, red_kick = bot.act(game, red_player)
        blue_mx, blue_my, blue_kick = bot.act(game, blue_player)

        # Red moving forward should have global mx > 0 (+X, towards Blue goal)
        self.assertAlmostEqual(red_mx, 1.0)
        # Blue moving forward should have global mx < 0 (-X, towards Red goal)
        self.assertAlmostEqual(blue_mx, -1.0)

if __name__ == "__main__":
    unittest.main()
