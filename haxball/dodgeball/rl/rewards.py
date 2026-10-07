"""
Aggressive Hunter Reward Engine for HaxBall Dodgeball RL.
Designed to eradicate cowardly / passive behavior:
- Continuous negative penalty for being far from the ball (Forces constant pursuit)
- Massive reward for closing distance, kicking, and launching projectiles at opponents
- High reward for forceful knockback impacts and eliminations
- Heavy anti-camping penalty for staying in the back
- Zero fear of death (low death penalty) so agent prioritizes aggressive attacks over survival
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional
try:
    from core.vector import Vec2
    from core.constants import Team
    from core.dodgeball_game import DodgeballGame
except (ImportError, ValueError):
    from ..core.vector import Vec2
    from ..core.constants import Team
    from ..core.dodgeball_game import DodgeballGame


class DodgeballRewardEngine:
    def __init__(
        self,
        kill_reward: float = 30.0,
        death_penalty: float = 0.5,
        wall_suicide_penalty: float = 1.0,
        kick_shot_bonus: float = 3.0,
        shot_towards_opp_bonus: float = 5.0,
        knockback_hit_bonus: float = 15.0,
        rebote_bonus: float = 8.0,
        approach_ball_weight: float = 0.25,
        distance_penalty_weight: float = 0.06,
    ):
        self.kill_reward = kill_reward
        self.death_penalty = death_penalty
        self.wall_suicide_penalty = wall_suicide_penalty
        self.kick_shot_bonus = kick_shot_bonus
        self.shot_towards_opp_bonus = shot_towards_opp_bonus
        self.knockback_hit_bonus = knockback_hit_bonus
        self.rebote_bonus = rebote_bonus
        self.approach_ball_weight = approach_ball_weight
        self.distance_penalty_weight = distance_penalty_weight
        self._prev_dist_to_ball: Dict[int, float] = {}

    def reset(self):
        self._prev_dist_to_ball.clear()

    def compute_rewards(self, game: DodgeballGame, step_info: Dict[str, Any]) -> Dict[int, float]:
        rewards: Dict[int, float] = {p.player_id: 0.0 for p in game.players}
        bg_w = game.stadium.bg_width
        bg_h = game.stadium.bg_height
        ball = game.ball

        # 1. Eliminações (Vitória / Ponto para o atacante)
        for elim in step_info.get("eliminations", []):
            p_id = elim["player_id"]
            reason = elim["reason"]
            credited_team = elim["credited_team"]

            if reason == "wall_suicide":
                rewards[p_id] -= self.wall_suicide_penalty
            else:
                rewards[p_id] -= self.death_penalty

            for opp in game.players:
                if opp.team == credited_team and game.is_alive(opp.player_id):
                    rewards[opp.player_id] += self.kill_reward

        # 2. Recompensas de Chute e Projétil Ativo (OFENSIVIDADE TOTAL)
        for kick_ev in step_info.get("events", {}).get("kicks", []):
            kicker_id = kick_ev["player_id"]
            if kicker_id in rewards:
                rewards[kicker_id] += self.kick_shot_bonus

                # Bônus por rebote de alta velocidade
                if kick_ev.get("is_rebote", False):
                    rewards[kicker_id] += self.rebote_bonus

                # Bônus adicional se o chute mandou a bola em direção ao campo adversário
                kicker = next((p for p in game.players if p.player_id == kicker_id), None)
                if kicker:
                    is_red = (kicker.team == Team.RED)
                    shot_forward = (ball.speed.x > 0) if is_red else (ball.speed.x < 0)
                    if shot_forward:
                        rewards[kicker_id] += self.shot_towards_opp_bonus

        # 3. Impacto de Knockback no Oponente (Acertou o tiro com força)
        for impact in step_info.get("events", {}).get("ball_player_impacts", []):
            p_id = impact["player_id"]
            team = impact["team"]
            speed = impact["impact_speed"]

            if speed > 2.0:
                for opp in game.players:
                    if opp.team != team and opp.player_id == game.last_kicker_id and opp.player_id in rewards:
                        rewards[opp.player_id] += self.knockback_hit_bonus

        # 4. PUNIÇÃO CONTÍNUA POR DISTÂNCIA DA BOLA (ELE TEM QUE JOGAR!)
        for p in game.players:
            if not game.is_alive(p.player_id):
                continue

            curr_dist = p.pos.distance_to(ball.pos)
            prev_dist = self._prev_dist_to_ball.get(p.player_id, curr_dist)

            is_red = (p.team == Team.RED)
            ball_on_own_half = (ball.pos.x < 0) if is_red else (ball.pos.x > 0)
            ball_near_divider = abs(ball.pos.x) < 60.0

            if ball_on_own_half or ball_near_divider:
                # RECOMPENSA NEGATIVA SE NÃO ESTIVER PERTO DA BOLA:
                # Cada tick longe da bola drena recompensa!
                rewards[p.player_id] -= self.distance_penalty_weight * (curr_dist / 80.0)

                # Recompensa positiva forte ao fechar distância
                delta = prev_dist - curr_dist
                rewards[p.player_id] += delta * self.approach_ball_weight
            else:
                # Mesmo com a bola no outro campo, se ficar acampado no fundo perde pontos
                pass

            # Anti-camping severo: Proibido ficar acovardado no fundo da quadra
            back_dist = (bg_w - p.radius + p.pos.x) if is_red else (bg_w - p.radius - p.pos.x)
            if back_dist < 60.0:
                rewards[p.player_id] -= 0.08  # Sai do fundo!

            # Bônus por pressão na rede / linha divisória
            dist_to_net = abs(p.pos.x)
            if dist_to_net < 80.0:
                rewards[p.player_id] += 0.04

            self._prev_dist_to_ball[p.player_id] = curr_dist

        return rewards
