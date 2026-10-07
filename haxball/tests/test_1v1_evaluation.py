import unittest
import os
import torch
import math
from haxball.core.vector import Vec2
from haxball.core.constants import Team, GameState
from haxball.core.stadium import Stadium
from haxball.core.game import HaxBallGame
from haxball.bots import RLBot, HeuristicBot, WallReboundBot, GoalieBot

class Test1v1Evaluation(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        haxball_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.haxball_dir = haxball_dir
        candidates = [
            os.path.join(haxball_dir, "checkpoints", "meu_haxball_bot.pt"),
            os.path.join(haxball_dir, "checkpoints", "fase5_pro_master_1M.pt"),
            os.path.join(haxball_dir, "models", "checkpoints", "haxball_rl_best.pt"),
            "checkpoints/meu_haxball_bot.pt",
            "checkpoints/fase5_pro_master_1M.pt",
        ]
        self.model_path = next((c for c in candidates if os.path.exists(c)), candidates[0])
        self.assertTrue(os.path.exists(self.model_path), f"Checkpoint {self.model_path} not found")

    def _load_stadium(self, rel_path: str):
        full_path = os.path.join(self.haxball_dir, "maps", os.path.basename(rel_path))
        if not os.path.exists(full_path):
            full_path = rel_path
        return Stadium.load_from_file(full_path)

    def test_1v1_mechanics_and_pursuit_futsal(self):
        """Test 1v1 on futsal stadium for 3000 frames vs HeuristicBot."""
        stad = self._load_stadium("futsal_2v2.hbs")
        game = HaxBallGame(stadium=stad, red_players_count=1, blue_players_count=1, score_limit=10, time_limit_secs=300)

        
        red_bot = HeuristicBot("Red_Heuristic")
        blue_bot = RLBot(model_path=self.model_path, name="Blue_RL")
        
        pursuit_alignments = []
        contacts = 0
        kicks = 0
        dists = []
        
        for frame in range(3000):
            p_red = game.players[0]
            p_blue = game.players[1]
            
            inp_red = red_bot.act(game, p_red)
            inp_blue = blue_bot.act(game, p_blue)
            
            game.step({p_red.player_id: inp_red, p_blue.player_id: inp_blue})
            
            # Telemetry for Blue (RL Bot)
            ball = game.ball
            dist = p_blue.pos.distance_to(ball.pos)
            dists.append(dist)
            
            if dist < (p_blue.radius + ball.radius + 5.0):
                contacts += 1
            if p_blue.is_kicking:
                kicks += 1
                
            if dist >= (p_blue.radius + ball.radius + 5.0) and p_blue.speed.length_sq() > 0.05:
                to_ball = (ball.pos - p_blue.pos).normalized()
                dir_vel = p_blue.speed.normalized()
                cos_sim = dir_vel.dot(to_ball)
                pursuit_alignments.append(max(0.0, float(cos_sim)))

        avg_pursuit = sum(pursuit_alignments) / max(1, len(pursuit_alignments))
        avg_dist = sum(dists) / max(1, len(dists))
        
        print(f"\n[1v1 Futsal vs Heuristic] Steps: 3000 | Pursuit (on approach): {avg_pursuit*100:.1f}% | Avg Dist: {avg_dist:.1f}px | Contacts: {contacts} | Kicks: {kicks} | Score: Red {game.red_score} x {game.blue_score} Blue")
        
        self.assertGreater(avg_pursuit, 0.70, "RL bot pursuit index should be > 70% in 1v1 approach")
        self.assertLess(avg_dist, 180.0, "RL bot should stay close to the ball in 1v1")
        self.assertGreater(contacts, 30, "RL bot should have active ball contacts in 1v1")

    def test_1v1_micro_arena(self):
        """Test 1v1 on fast Micro 1v1 arena."""
        stad = self._load_stadium("micro_1v1.hbs")
        game = HaxBallGame(stadium=stad, red_players_count=1, blue_players_count=1, score_limit=10, time_limit_secs=300)
        
        red_bot = HeuristicBot("Red_Heuristic")
        blue_bot = RLBot(model_path=self.model_path, name="Blue_RL")
        
        pursuit_alignments = []
        dists = []
        for _ in range(2000):
            p_red = game.players[0]
            p_blue = game.players[1]
            inp_red = red_bot.act(game, p_red)
            inp_blue = blue_bot.act(game, p_blue)
            game.step({p_red.player_id: inp_red, p_blue.player_id: inp_blue})
            
            ball = game.ball
            dist = p_blue.pos.distance_to(ball.pos)
            dists.append(dist)
            if dist >= (p_blue.radius + ball.radius + 5.0) and p_blue.speed.length_sq() > 0.05:
                to_ball = (ball.pos - p_blue.pos).normalized()
                pursuit_alignments.append(max(0.0, float(p_blue.speed.normalized().dot(to_ball))))
                
        avg_pursuit = sum(pursuit_alignments) / max(1, len(pursuit_alignments))
        avg_dist = sum(dists) / max(1, len(dists))
        print(f"\n[1v1 Micro Arena] Steps: 2000 | Pursuit: {avg_pursuit*100:.1f}% | Avg Dist: {avg_dist:.1f}px | Score: Red {game.red_score} x {game.blue_score} Blue")
        self.assertGreater(avg_pursuit, 0.70)
        self.assertLess(avg_dist, 200.0)

    def test_1v1_rl_as_red_and_blue(self):
        """Verify that RLBot plays symmetrically as Red and as Blue."""
        stad = self._load_stadium("futsal_2v2.hbs")
        
        # Test RL as Red
        game_red = HaxBallGame(stadium=stad, red_players_count=1, blue_players_count=1)
        rl_red = RLBot(model_path=self.model_path, name="Red_RL")
        opp_blue = HeuristicBot("Opp_Blue")
        
        red_dists = []
        for _ in range(1500):
            p_red = game_red.players[0]
            p_blue = game_red.players[1]
            game_red.step({p_red.player_id: rl_red.act(game_red, p_red), p_blue.player_id: opp_blue.act(game_red, p_blue)})
            red_dists.append(p_red.pos.distance_to(game_red.ball.pos))
            
        avg_red_dist = sum(red_dists) / len(red_dists)
        print(f"\n[1v1 Symmetrical Test] RL as Red Avg Dist: {avg_red_dist:.1f}px | Score: Red {game_red.red_score} x {game_red.blue_score} Blue")
        self.assertLess(avg_red_dist, 220.0)

    def test_1v1_vs_wall_rebound_bot(self):
        """Test 1v1 vs WallReboundBot on classic map."""
        stad = self._load_stadium("classic.hbs")
        game = HaxBallGame(stadium=stad, red_players_count=1, blue_players_count=1, score_limit=10, time_limit_secs=300)
        red_bot = WallReboundBot("Red_Wall")
        blue_bot = RLBot(model_path=self.model_path, name="Blue_RL")
        
        pursuits = []
        for _ in range(2000):
            p_red = game.players[0]
            p_blue = game.players[1]
            game.step({p_red.player_id: red_bot.act(game, p_red), p_blue.player_id: blue_bot.act(game, p_blue)})
            dist = p_blue.pos.distance_to(game.ball.pos)
            if dist >= (p_blue.radius + game.ball.radius + 5.0) and p_blue.speed.length_sq() > 0.05:
                to_ball = (game.ball.pos - p_blue.pos).normalized()
                pursuits.append(max(0.0, float(p_blue.speed.normalized().dot(to_ball))))
        
        avg_p = sum(pursuits) / max(1, len(pursuits))
        print(f"\n[1v1 Classic vs WallRebound] Steps: 2000 | Pursuit: {avg_p*100:.1f}% | Score: Red {game.red_score} x {game.blue_score} Blue")
        self.assertGreater(avg_p, 0.60)

    def test_1v1_mirror_match(self):
        """Test RL vs RL mirror 1v1 match."""
        stad = self._load_stadium("futsal_2v2.hbs")
        game = HaxBallGame(stadium=stad, red_players_count=1, blue_players_count=1, score_limit=10, time_limit_secs=300)
        red_bot = RLBot(model_path=self.model_path, name="Red_RL")
        blue_bot = RLBot(model_path=self.model_path, name="Blue_RL")
        
        contacts = 0
        for _ in range(2000):
            p_red = game.players[0]
            p_blue = game.players[1]
            game.step({p_red.player_id: red_bot.act(game, p_red), p_blue.player_id: blue_bot.act(game, p_blue)})
            if p_red.pos.distance_to(game.ball.pos) < 30 or p_blue.pos.distance_to(game.ball.pos) < 30:
                contacts += 1
                
        print(f"\n[1v1 Mirror RL vs RL] Steps: 2000 | Active Contested Frames: {contacts} | Score: Red {game.red_score} x {game.blue_score} Blue")
        self.assertGreater(contacts, 50)

    def test_kickoff_barrier_mechanics(self):
        """Verify that kickoff barriers block opponent from crossing half and center circle until touch."""
        stad = self._load_stadium("futsal_2v2.hbs")

        game = HaxBallGame(stadium=stad, red_players_count=1, blue_players_count=1)
        
        # Initial state should be KICKOFF_RED
        self.assertEqual(game.state, GameState.KICKOFF_RED)
        
        # Blue tries to rush into Red half (-1.0, 0.0)
        p_blue = game.players[1]
        for _ in range(60):
            game.step({p_blue.player_id: (-1.0, 0.0, False)})
            
        # Blue must NOT be able to cross into Red half (x >= 0) or center circle (dist >= kickOffRadius)
        self.assertGreaterEqual(p_blue.pos.x, 0.0, "Blue player must not cross into Red half during Red kickoff")
        self.assertGreaterEqual(p_blue.pos.length(), stad.bg_kickoff_radius, "Blue must stay outside center circle")
        
        # Now Red moves all the way to the ball at (0,0) and kicks
        p_red = game.players[0]
        for _ in range(150):
            to_ball = (game.ball.pos - p_red.pos).normalized()
            game.step({p_red.player_id: (to_ball.x, to_ball.y, True)})
            if game.state == GameState.PLAYING:
                break
                
        self.assertEqual(game.state, GameState.PLAYING, "Match must transition to PLAYING upon kickoff touch")
        print("\n[Kickoff Mechanics] Barrier successfully held opponent outside center circle & transitioned upon kick!")

if __name__ == "__main__":
    unittest.main()
