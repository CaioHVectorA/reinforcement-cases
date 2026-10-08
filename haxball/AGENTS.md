# 🤖 AGENTS.md — Guia de Agentes de IA, Bots e Modelos

Este documento detalha todos os tipos de agentes implementados na codebase `haxball`, suas arquiteturas neurais, heurísticas táticas, convenções de observação/ação e como instanciá-los ou treiná-los.

---

## 1. Categorias de Agentes no Projeto

A codebase divide seus agentes em três grandes classes:

```
                  ┌────────────────────────┐
                  │      BaseBot API       │
                  └───────────┬────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
┌────────▼─────────┐ ┌────────▼─────────┐ ┌────────▼─────────┐
│ Bots Heurísticos │ │  NPCs Paramétricos│ │   Modelos de RL  │
│  & Especialistas │ │   (Arquetípicos)  │ │   & Imitação     │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

1. **Bots Heurísticos & Regras Rígidas:** Focados em comportamentos básicos determinísticos (ex.: goleiro puro, perseguição simples de bola).
2. **NPCs Arquetípicos Parametrizados:** Bots baseados no motor geométrico [`NPCBot`](file:///home/usuario/develop/reinforcement-cases/haxball/bots/npc.py), configurados com diferentes perfis de jogo competitivo (*tryhard*).
3. **Agentes Neurais (RLBot / Policy Networks):** Redes treinadas via Behavioral Cloning (BC) ou Reinforcement Learning (PPO/DQN), capazes de inferência a partir de observações em vetor.

---

## 2. Bots Heurísticos e NPCs Arquetípicos

Localizados no diretório [`bots/`](file:///home/usuario/develop/reinforcement-cases/haxball/bots/):

### 2.1. [`NPCBot`](file:///home/usuario/develop/reinforcement-cases/haxball/bots/npc.py) e Perfis em [`bots/archetypes.py`](file:///home/usuario/develop/reinforcement-cases/haxball/bots/archetypes.py)
Todos compartilham a mesma inteligência geométrica avançada (cálculo de trajetória de bola, reflexão em tabelas de parede, fintas e dribles curtos), variando apenas os hiperparâmetros de `NPCProfile`:

| Arquétipo | Apelido | Características Principais |
|---|---|---|
| `PressBot` | *Pressão Total* | Pressiona incessantemente o portador da bola; finaliza em qualquer brecha; usa tabelas ofensivas. |
| `StrikerBot` | *Artilheiro* | Especialista em chutes de longa distância, visando os cantos da trave (*far post bias*). |
| `BankBot` | *Tabelador* | Especialista em rebotes de parede; prefere tabelar do que chutar direto. |
| `DribblerBot` | *Fintador* | Retém a bola colada ao corpo (*dribble swerving*); só finaliza com gol completamente aberto. |
| `CounterBot` | *Muralha* | Fica na linha de contenção defensiva; limpa a bola pelas paredes e sai no contra-ataque. |
| `MasterBot` | *Mestre* | Combina pressão máxima, fintas dinâmicas, tabelas na parede e visão coletiva. |
| `HeuristicBot` | *Clássico* | Baseline simples de perseguição de bola e chute direto sem cálculo de parede. |
| `GoalieBot` | *Goleiro* | Patrulha a linha de gol mantendo o ângulo bissetor entre a bola e as traves. |
| `WallReboundBot` | *Tabelas* | Heurística focada em predição de reflexão de bolas nas paredes. |

### 2.2. Coordenação de Equipe 3v3 ([`bots/futsal_3v3_team.py`](file:///home/usuario/develop/reinforcement-cases/haxball/bots/futsal_3v3_team.py))
Coordena três jogadores de forma síncrona com papéis dinâmicos:
- **Zagueiro / Fixa:** Mantém cobertura defensiva atrás da linha da bola.
- **Ala:** Dá opção de passe aberto pelas laterais do campo.
- **Pivô / Atacante:** Pressiona a saída de bola adversária e finaliza jogadas.

---

## 3. Agentes Neurais e Aprendizado por Reforço

### 3.1. [`RLBot`](file:///home/usuario/develop/reinforcement-cases/haxball/bots/rl_bot.py)
Classe unificada que encapsula modelos PyTorch (`.pt`). Ela detecta automaticamente o formato dos pesos carregados:
- **Behavioral Cloning (`HaxBallExpertPolicy`):** MLP de 3 camadas com LayerNorm e saída de 18 ações discretas.
- **Entity Attention (`EntityAttentionPolicy`):** Modelo transformer/self-attention que trata cada disco (jogadores e bola) como um token com invariância de ordem.
- **Actor-Critic MLP (`ActorCriticMLP`):** Arquitetura padrão para PPO com cabeças de política e valor.

### 3.2. Espaço de Observação (61 Dimensões)
Construído por [`rl/observations/decoupled_obs.py`](file:///home/usuario/develop/reinforcement-cases/haxball/rl/observations/decoupled_obs.py):
- Posição normalizada do agente $(x, y) \in [-1, 1]$.
- Velocidade normalizada $(\dot{x}, \dot{y})$.
- Posição e velocidade da bola relativas e absolutas.
- Distância e ângulos para ambas as traves de gol.
- Vetor de kick em cooldown / disponibilidade de chute.
- Posições e velocidades relativas de todos os aliados e adversários (com padding para diferentes formatos de jogo).

### 3.3. Espaço de Ações (18 Classes Discretas)
Mapeadas em [`rl/actions/action_space.py`](file:///home/usuario/develop/reinforcement-cases/haxball/rl/actions/action_space.py):
- $0 \dots 8$: 8 direções de movimento (Neutro, Cima, Cima-Direita, Direita, Baixo-Direita, Baixo, Baixo-Esquerda, Esquerda, Cima-Esquerda) com `kick = False`.
- $9 \dots 17$: As mesmas 8 direções de movimento com `kick = True`.

---

## 4. Como Executar e Testar os Agentes

### 4.1. Jogar contra um Bot no Modo Gráfico
```bash
# Partida 1v1 contra o MasterBot na arena Pygame
python play.py --bot MasterBot

# Partida contra o RLBot carregando um checkpoint
python play.py --bot RLBot --model checkpoints/haxball_rl_best.pt
```

### 4.2. Executar Benchmarks Automatizados
```bash
# Roda torneio em lote entre diferentes arquétipos para extrair winrate
python benchmarks/run_benchmarks.py
```

### 4.3. Treinar Novo Agente por Reforço
```bash
# Inicia treinamento PPO com self-play local acelerado
python train_rl.py --episodes 10000 --save-interval 500
```

### 4.4. Pré-treinar por Imitação (Behavioral Cloning)
```bash
# Treina com replays salvos em data/filtered_replays_1v1
python tools/train_bc_3v3.py --data-dir data/filtered_replays_1v1 --epochs 50
```

---

## 5. Diretrizes para Adicionar Novos Agentes

1. **Novos Bots Heurísticos:** Herde de `BaseBot` em [`bots/base_bot.py`](file:///home/usuario/develop/reinforcement-cases/haxball/bots/base_bot.py) e implemente o método `act(game_state: HaxBallGame, player_disc: Disc) -> Tuple[Vec2, bool]`.
2. **Novos Perfis de NPC:** Crie uma nova classe em [`bots/archetypes.py`](file:///home/usuario/develop/reinforcement-cases/haxball/bots/archetypes.py) ajustando os parâmetros de `NPCProfile`.
3. **Novas Redes Neurais:** Adicione os módulos em [`rl/models/`](file:///home/usuario/develop/reinforcement-cases/haxball/rl/models/) e certifique-se de registrar a lógica de desempacotamento no [`RLBot`](file:///home/usuario/develop/reinforcement-cases/haxball/bots/rl_bot.py).
