"""
Game manager for HaxBall.
Handles match lifecycle, kickoff rules, goal detection, score tracking,
timer, and kickoff barrier collision mask transitions.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any
from haxball.core.vector import Vec2
from haxball.core.constants import Team, GameState, CollisionMask, FPS
from haxball.core.disc import Disc
from haxball.core.stadium import Stadium
from haxball.core.physics_engine import PhysicsEngine

class HaxBallGame:
    def __init__(
        self,
        stadium: Stadium,
        score_limit: int = 3,
        time_limit_secs: int = 180,
        red_players_count: int = 1,
        blue_players_count: int = 1,
    ):
        self.stadium = stadium
        self.score_limit = score_limit
        self.time_limit_secs = time_limit_secs
        self.red_players_count = red_players_count
        self.blue_players_count = blue_players_count

        self.physics = PhysicsEngine(stadium)

        self.state: GameState = GameState.KICKOFF_RED
        self.red_score: int = 0
        self.blue_score: int = 0
        self.time_ticks: int = 0
        self.celebration_timer: int = 0
        self.last_goal_team: Team = Team.NONE

        self.ball: Disc = self.stadium.create_ball()
        self.players: List[Disc] = []
        self._init_players()

        self.prev_ball_pos = self.ball.pos.copy()
        self.reset_match()

    def _init_players(self):
        self.players.clear()
        for i in range(self.red_players_count):
            p = self.stadium.create_player(Team.RED, player_id=1 + i, player_number=1 + i)
            self.players.append(p)
        for i in range(self.blue_players_count):
            p = self.stadium.create_player(Team.BLUE, player_id=101 + i, player_number=1 + i)
            self.players.append(p)

    def reset_match(self, starting_kickoff: Team = Team.RED):
        self.red_score = 0
        self.blue_score = 0
        self.time_ticks = 0
        self.celebration_timer = 0
        self.last_goal_team = Team.NONE
        self.reset_round(kickoff_team=starting_kickoff)

    def reset_round(self, kickoff_team: Team = Team.RED):
        self.state = GameState.KICKOFF_RED if kickoff_team == Team.RED else GameState.KICKOFF_BLUE
        self.celebration_timer = 0

        # Reset ball to center
        self.ball.pos = Vec2(0.0, 0.0)
        self.ball.speed = Vec2(0.0, 0.0)
        self.prev_ball_pos = self.ball.pos.copy()

        # Spawn players
        spawn_dist = self.stadium.spawn_distance
        h = self.stadium.bg_height

        red_players = [p for p in self.players if p.team == Team.RED]
        blue_players = [p for p in self.players if p.team == Team.BLUE]

        for i, p in enumerate(red_players):
            p.speed = Vec2(0.0, 0.0)
            p.is_kicking = False
            p.kick_flash = 0
            n = len(red_players)
            spacing = min(70.0, (h * 1.4) / max(1, n))
            offset_y = (i - (n - 1) / 2.0) * spacing
            p.pos = Vec2(-spawn_dist, offset_y)

        for i, p in enumerate(blue_players):
            p.speed = Vec2(0.0, 0.0)
            p.is_kicking = False
            p.kick_flash = 0
            n = len(blue_players)
            spacing = min(70.0, (h * 1.4) / max(1, n))
            offset_y = (i - (n - 1) / 2.0) * spacing
            p.pos = Vec2(spawn_dist, offset_y)

        self._update_kickoff_masks()
        self.physics.reset_with_entities(self.ball, self.players)

    def _update_kickoff_masks(self):
        """
        Manages kickOffBarrier collision flags faithful to HaxBall:
        - Base player mask excludes redKO and blueKO.
        - During Red kickoff: Blue players receive blueKO mask (blocked from center circle and red half).
        - During Blue kickoff: Red players receive redKO mask (blocked from center circle and blue half).
        - During normal PLAYING: Kickoff barriers are completely disabled for all players.
        """
        base_mask = CollisionMask.ALL & ~(CollisionMask.RED_KO | CollisionMask.BLUE_KO)

        if self.state == GameState.KICKOFF_RED:
            for p in self.players:
                if p.team == Team.BLUE:
                    p.cMask = base_mask | CollisionMask.BLUE_KO
                else:
                    p.cMask = base_mask
        elif self.state == GameState.KICKOFF_BLUE:
            for p in self.players:
                if p.team == Team.RED:
                    p.cMask = base_mask | CollisionMask.RED_KO
                else:
                    p.cMask = base_mask
        else:
            for p in self.players:
                p.cMask = base_mask

    @property
    def time_seconds(self) -> float:
        return self.time_ticks / FPS

    @property
    def time_string(self) -> str:
        secs = int(self.time_seconds)
        mins = secs // 60
        rem_s = secs % 60
        return f"{mins:02d}:{rem_s:02d}"

    def step(self, inputs: Dict[int, Tuple[float, float, bool]]) -> Dict[str, Any]:
        step_info = {
            "state": self.state,
            "goal_scored": False,
            "scoring_team": Team.NONE,
            "game_over": False,
            "winner": Team.NONE,
            "events": {}
        }

        if self.state == GameState.GAME_OVER:
            step_info["game_over"] = True
            step_info["winner"] = Team.RED if self.red_score > self.blue_score else Team.BLUE
            return step_info

        if self.state == GameState.GOAL_CELEBRATION:
            self.celebration_timer -= 1
            self.ball.speed = self.ball.speed * 0.95
            self.ball.pos = self.ball.pos + self.ball.speed
            for p in self.players:
                p.speed = p.speed * 0.95
                p.pos = p.pos + p.speed

            if self.celebration_timer <= 0:
                next_kickoff = Team.BLUE if self.last_goal_team == Team.RED else Team.RED
                self.reset_round(kickoff_team=next_kickoff)
            return step_info

        self.prev_ball_pos = self.ball.pos.copy()
        phys_events = self.physics.step(inputs)
        step_info["events"] = phys_events

        # Transition from Kickoff to Playing as soon as ball is touched or moved
        if self.state in (GameState.KICKOFF_RED, GameState.KICKOFF_BLUE):
            if self.ball.speed.length_sq() > 0.01 or len(phys_events["kicks"]) > 0:
                self.state = GameState.PLAYING
                self._update_kickoff_masks()

        # Goal detection
        goal_team = self._check_goals()
        if goal_team != Team.NONE:
            step_info["goal_scored"] = True
            step_info["scoring_team"] = goal_team
            self.last_goal_team = goal_team

            if goal_team == Team.RED:
                self.red_score += 1
            else:
                self.blue_score += 1

            if (self.score_limit > 0 and (self.red_score >= self.score_limit or self.blue_score >= self.score_limit)):
                self.state = GameState.GAME_OVER
                step_info["game_over"] = True
                step_info["winner"] = Team.RED if self.red_score > self.blue_score else Team.BLUE
            else:
                self.state = GameState.GOAL_CELEBRATION
                self.celebration_timer = int(FPS * 2.0)
            return step_info

        self.time_ticks += 1
        if self.time_limit_secs > 0 and self.time_seconds >= self.time_limit_secs:
            if self.red_score != self.blue_score:
                self.state = GameState.GAME_OVER
                step_info["game_over"] = True
                step_info["winner"] = Team.RED if self.red_score > self.blue_score else Team.BLUE

        return step_info

    def _check_goals(self) -> Team:
        for goal in self.stadium.goals:
            if goal.check_crossed(self.prev_ball_pos, self.ball.pos):
                return goal.scoring_team
        return Team.NONE
