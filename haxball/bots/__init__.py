"""
Bots package for HaxBall agents and baselines.

`NPC_BOTS` is the registry of scripted bots (same brain, different profiles). The GUI and the
launcher read it, so adding a bot here makes it available everywhere.
"""

from haxball.bots.base_bot import BaseBot
from haxball.bots.npc import NPCBot, NPCProfile
from haxball.bots.archetypes import (
    PressBot, StrikerBot, BankBot, DribblerBot, CounterBot, MasterBot,
    HeuristicBot, WallReboundBot,
)
from haxball.bots.goalie_bot import GoalieBot


def __getattr__(name):
    # RLBot pulls in torch (slow import); load it only when actually requested.
    if name == "RLBot":
        from haxball.bots.rl_bot import RLBot
        return RLBot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# key -> (class, short label, title, description, color)
NPC_BOTS = {
    "heuristic": (HeuristicBot, "Clássico", "HeuristicBot (Clássico)",
                  "Pressiona e chuta com lance livre. Sem tabelas.", (58, 142, 230)),
    "wall": (WallReboundBot, "Tabelas", "WallReboundBot (Tabelas)",
             "Bate na parede quando o gol está bloqueado.", (230, 140, 40)),
    "striker": (StrikerBot, "Artilheiro", "StrikerBot (Artilheiro)",
                "Chuta cedo e de longe nos cantos do gol.", (235, 90, 90)),
    "dribbler": (DribblerBot, "Fintador", "DribblerBot (Fintador)",
                 "Segura a bola, desvia de você e só chuta livre.", (220, 200, 50)),
    "bank": (BankBot, "Tabelador", "BankBot (Tabelador)",
             "Especialista em tabelas: chutes e passes na parede.", (255, 160, 70)),
    "counter": (CounterBot, "Muralha", "CounterBot (Muralha)",
                "Defende na linha da bola e sai em contra-ataque.", (120, 190, 255)),
    "press": (PressBot, "Pressão Total", "PressBot (Pressão Total)",
              "Marca a bola o tempo todo, chuta livre e finta com tabela.", (60, 210, 120)),
    "master": (MasterBot, "Mestre", "MasterBot (Mestre)",
               "Tudo ligado: pressão, mira, tabelas, fintas e jogo em equipe.", (200, 120, 255)),
    "goalie": (GoalieBot, "Goleiro", "GoalieBot (Goleiro)",
               "Fecha o ângulo do gol e rebate.", (160, 120, 255)),
}

# Order used by the "play against every bot" gauntlet (roughly easiest -> hardest)
GAUNTLET_ORDER = ["heuristic", "wall", "striker", "dribbler", "bank", "counter", "press", "master", "rl"]

# Aliases for compatibility
PressingBot = PressBot
MasterProBot = MasterBot

__all__ = [
    "BaseBot", "NPCBot", "NPCProfile", "HeuristicBot", "WallReboundBot", "GoalieBot", "RLBot",
    "PressBot", "StrikerBot", "BankBot", "DribblerBot", "CounterBot", "MasterBot",
    "PressingBot", "MasterProBot",
    "NPC_BOTS", "GAUNTLET_ORDER",
]
