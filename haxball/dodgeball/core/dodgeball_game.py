"""
Dodgeball Game Manager for HaxBall Dodgeball.
Core Game Rules:
1. The ball DOES NOT KILL instantly upon contact. It applies a strong kinetic knockback force.
2. The player is ELIMINATED ONLY when they contact the outer perimeter wall of their court:
   - Either by being blasted/pushed into the wall by the ball's force ("pushed_into_wall")
   - Or by accidentally walking into/sticking to the wall ("wall_suicide")
3. When eliminated, the opposing team gains +1 point.
4. When all players of one team are eliminated, the round ends.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Set, Any
from .vector import Vec2
from .constants import Team, GameState, CollisionMask, FPS
from .disc import Disc
from .stadium import Stadium
from .physics_engine import PhysicsEngine

class DodgeballGame:
    def __init__(
        self,
        stadium: Stadium,
        score_limit: int = 5,
        time_limit_secs: int = 300,
        red_players_count: int = 1,
        blue_players_count: int = 1,
    ):
        self.stadium = stadium
        self.score_limit = score_limit
        self.time_limit_secs = time_limit_secs
        self.red_players_count = red_players_count
        self.blue_players_count = blue_players_count

        self.physics = PhysicsEngine(stadium)

        self.state: GameState = GameState.PLAYING
        self.red_score: int = 0
        self.blue_score: int = 0
        self.time_ticks: int = 0
        self.celebration_timer: int = 0
        self.last_round_winner: Team = Team.NONE
        self.last_goal_team: Team = Team.NONE

        self.ball: Disc = self.stadium.create_ball()
        self.players: List[Disc] = []
        self._init_players()

        self.alive_players: Set[int] = set()
        self.eliminations: List[Dict[str, Any]] = []

        # Tracking who last struck the ball to credit knockback wall kills
        self.last_kicker_id: Optional[int] = None
        self.last_kicker_team: Team = Team.NONE
        self.last_kick_tick: int = -100

        # Tracking who recently suffered a ball impact
        self.last_impact_tick: Dict[int, int] = {}
        self.last_impact_from_team: Dict[int, Team] = {}

        self.reset_match()

    def _init_players(self):
        self.players.clear()
        for i in range(self.red_players_count):
            p = self.stadium.create_player(Team.RED, player_id=1 + i, player_number=1 + i)
            self.players.append(p)
        for i in range(self.blue_players_count):
            p = self.stadium.create_player(Team.BLUE, player_id=101 + i, player_number=1 + i)
            self.players.append(p)

    def reset_match(self):
        self.red_score = 0
        self.blue_score = 0
        self.time_ticks = 0
        self.celebration_timer = 0
        self.last_round_winner = Team.NONE
        self.last_goal_team = Team.NONE
        self.reset_round()

    def reset_round(self, starting_ball_team: Team = Team.NONE):
        self.celebration_timer = 0
        self.state = GameState.PLAYING
        self.eliminations.clear()
        self.alive_players = {p.player_id for p in self.players}

        bg_h = self.stadium.bg_height
        spawn_dist = self.stadium.spawn_distance

        red_players = [p for p in self.players if p.team == Team.RED]
        blue_players = [p for p in self.players if p.team == Team.BLUE]

        for p in self.players:
            p.speed = Vec2(0.0, 0.0)
            p.is_kicking = False
            p.kick_flash = 0
            p.cMask = CollisionMask.ALL

        for i, p in enumerate(red_players):
            n = len(red_players)
            spacing = min(70.0, (bg_h * 1.2) / max(1, n))
            offset_y = (i - (n - 1) / 2.0) * spacing
            p.pos = Vec2(-spawn_dist, offset_y)

        for i, p in enumerate(blue_players):
            n = len(blue_players)
            spacing = min(70.0, (bg_h * 1.2) / max(1, n))
            offset_y = (i - (n - 1) / 2.0) * spacing
            p.pos = Vec2(spawn_dist, offset_y)

        # Place ball
        self.ball.speed = Vec2(0.0, 0.0)
        if starting_ball_team == Team.RED:
            self.ball.pos = Vec2(-spawn_dist * 0.4, 0.0)
        elif starting_ball_team == Team.BLUE:
            self.ball.pos = Vec2(spawn_dist * 0.4, 0.0)
        else:
            self.ball.pos = Vec2(0.0, 0.0)

        self.last_kicker_id = None
        self.last_kicker_team = Team.NONE
        self.last_kick_tick = -100
        self.last_impact_tick.clear()
        self.last_impact_from_team.clear()

        self.physics.reset_with_entities(self.ball, self.players)

    @property
    def time_seconds(self) -> float:
        return self.time_ticks / FPS

    @property
    def time_string(self) -> str:
        secs = int(self.time_seconds)
        mins = secs // 60
        rem_s = secs % 60
        return f"{mins:02d}:{rem_s:02d}"

    @property
    def red_alive_count(self) -> int:
        return sum(1 for p in self.players if p.team == Team.RED and p.player_id in self.alive_players)

    @property
    def blue_alive_count(self) -> int:
        return sum(1 for p in self.players if p.team == Team.BLUE and p.player_id in self.alive_players)

    def is_alive(self, player_id: int) -> bool:
        return player_id in self.alive_players

    def eliminate_player(self, player: Disc, reason: str, credited_team: Team) -> Dict[str, Any]:
        """Eliminates a player who hit the deadly perimeter wall."""
        if player.player_id not in self.alive_players:
            return {}

        self.alive_players.remove(player.player_id)
        player.cMask = CollisionMask.NONE
        player.speed = Vec2(0.0, 0.0)
        # Move to neutral sideline
        side_x = -395.0 if player.team == Team.RED else 395.0
        player.pos = Vec2(side_x, player.pos.y)

        elim_event = {
            "player_id": player.player_id,
            "team": player.team,
            "reason": reason,
            "credited_team": credited_team,
            "tick": self.time_ticks,
        }
        self.eliminations.append(elim_event)

        # Opponent team scores
        if credited_team == Team.RED:
            self.red_score += 1
        elif credited_team == Team.BLUE:
            self.blue_score += 1

        return elim_event

    def check_wall_eliminations(self) -> List[Dict[str, Any]]:
        """
        The Deadly Wall Rule:
        A player is eliminated ONLY when they touch the outer perimeter wall of their half!
        - If they were recently struck by an opponent's shot: 'pushed_into_wall'
        - If they hit the wall on their own: 'wall_suicide'
        """
        eliminations = []
        bg_w = self.stadium.bg_width
        bg_h = self.stadium.bg_height

        for p in self.players:
            if p.player_id not in self.alive_players:
                continue

            wall_touched = False
            r = p.radius

            if p.team == Team.RED:
                # Red half: x is between -bg_w and 0
                # Outer walls: Top (y >= bg_h), Bottom (y <= -bg_h), Back (x <= -bg_w)
                if p.pos.y >= (bg_h - r - 0.5) or p.pos.y <= (-bg_h + r + 0.5) or p.pos.x <= (-bg_w + r + 0.5):
                    wall_touched = True
            elif p.team == Team.BLUE:
                # Blue half: x is between 0 and bg_w
                # Outer walls: Top (y >= bg_h), Bottom (y <= -bg_h), Back (x >= bg_w)
                if p.pos.y >= (bg_h - r - 0.5) or p.pos.y <= (-bg_h + r + 0.5) or p.pos.x >= (bg_w - r - 0.5):
                    wall_touched = True

            if wall_touched:
                opp_team = Team.BLUE if p.team == Team.RED else Team.RED
                recent_impact_tick = self.last_impact_tick.get(p.player_id, -100)
                was_pushed = (self.time_ticks - recent_impact_tick <= 75) and (self.last_impact_from_team.get(p.player_id) == opp_team)

                reason = "pushed_into_wall" if was_pushed else "wall_suicide"
                ev = self.eliminate_player(p, reason=reason, credited_team=opp_team)
                if ev:
                    eliminations.append(ev)

        return eliminations

    def step(self, inputs: Dict[int, Tuple[float, float, bool]]) -> Dict[str, Any]:
        step_info = {
            "state": self.state,
            "round_over": False,
            "round_winner": Team.NONE,
            "game_over": False,
            "winner": Team.NONE,
            "eliminations": [],
            "events": {},
        }

        if self.state == GameState.GAME_OVER:
            step_info["game_over"] = True
            step_info["winner"] = Team.RED if self.red_score > self.blue_score else Team.BLUE
            return step_info

        if self.state == GameState.ROUND_CELEBRATION:
            self.celebration_timer -= 1
            if self.celebration_timer <= 0:
                next_ball = Team.BLUE if self.last_round_winner == Team.RED else Team.RED
                self.reset_round(starting_ball_team=next_ball)
            return step_info

        # Filter active inputs
        active_inputs = {}
        for p_id, cmd in inputs.items():
            if p_id in self.alive_players:
                active_inputs[p_id] = cmd
            else:
                active_inputs[p_id] = (0.0, 0.0, False)

        phys_events = self.physics.step(active_inputs)
        step_info["events"] = phys_events

        # Track kicks
        for kick_ev in phys_events.get("kicks", []):
            self.last_kicker_id = kick_ev["player_id"]
            self.last_kicker_team = kick_ev["team"]
            self.last_kick_tick = self.time_ticks

        # Track ball knockback impacts on players (PLAYER DOES NOT DIE, BUT IS PUSHED!)
        for impact in phys_events.get("ball_player_impacts", []):
            p_id = impact["player_id"]
            self.last_impact_tick[p_id] = self.time_ticks
            self.last_impact_from_team[p_id] = self.last_kicker_team

        # Check Deadly Wall Eliminations (ONLY DEADLY ON WALL CONTACT!)
        wall_elims = self.check_wall_eliminations()
        step_info["eliminations"].extend(wall_elims)

        self.time_ticks += 1

        # Check Round Over (all players of a team eliminated)
        red_alive = self.red_alive_count
        blue_alive = self.blue_alive_count

        if red_alive == 0 or blue_alive == 0:
            step_info["round_over"] = True
            if red_alive > 0:
                round_winner = Team.RED
            elif blue_alive > 0:
                round_winner = Team.BLUE
            else:
                round_winner = Team.NONE

            self.last_round_winner = round_winner
            self.last_goal_team = round_winner
            step_info["round_winner"] = round_winner

            if self.score_limit > 0 and (self.red_score >= self.score_limit or self.blue_score >= self.score_limit):
                self.state = GameState.GAME_OVER
                step_info["game_over"] = True
                step_info["winner"] = Team.RED if self.red_score > self.blue_score else Team.BLUE
            else:
                self.state = GameState.ROUND_CELEBRATION
                self.celebration_timer = int(FPS * 1.5)

            return step_info

        if self.time_limit_secs > 0 and self.time_seconds >= self.time_limit_secs:
            if self.red_score != self.blue_score:
                self.state = GameState.GAME_OVER
                step_info["game_over"] = True
                step_info["winner"] = Team.RED if self.red_score > self.blue_score else Team.BLUE

        return step_info
