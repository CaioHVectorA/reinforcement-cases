# 🧠 CONTEXT.md — HaxBall AI Studio & Reinforcement Learning Suite

> **Repositório:** `reinforcement-cases/haxball`  
> **Domínio:** Aprendizado por Reforço (RL), Aprendizado por Imitação (Behavioral Cloning), Simulação de Física 2D e Controle Multiagente Contínuo.  
> **Objetivo:** Desenvolver, simular, treinar e avaliar agentes de IA autônomos capazes de jogar HaxBall em nível competitivo (*tryhard*), desde 1v1 até 3v3 / 5v5 e variantes como Dodgeball, com suporte a inferência em tempo real e integração headless com salas online.

---

## 1. Visão Geral do Projeto

O **HaxBall AI Studio** é um ecossistema completo e modular escrito em Python (com suporte a PyTorch e Gym/Gymnasium) voltado à pesquisa de IA em futebol de física 2D em tempo real (60 FPS determinístico).

O sistema cobre todo o ciclo de vida de inteligência artificial em jogos:
1. **Engine de Física Fiel:** Implementação pura em Python com equações newtonianas, amortecimento viscoso (*damping*), reflexões e colisões elásticas/inelásticas com paredes normais e curvas de estádios `.hbs`.
2. **Coleta e Parsing de Dados:** Decodificador de alta performance do formato binário oficial `.hbr2` (comprimido em deflate/zlib) e scrapers para replays de partidas competitivas (Discord / TheHax).
3. **Imitação & Pré-treino (BC):** Pipeline de Behavioral Cloning usando centenas a milhares de partidas de humanos de alto nível (40k+ replays catalogados) com augmentação por simetria Y e X.
4. **Aprendizado por Reforço (RL):** PPO (Proximal Policy Optimization), DQN e Ligas de Self-Play Simétrico com *warm-start* a partir de pesos de imitação, modelagem de recompensa contra reward-hacking e observações desacopladas via Entity Attention.
5. **Execução & Avaliação:** Interface gráfica interativa (Pygame / Custom Tkinter Lobby Hub), modo Headless de alto throughput (> 10.000 passos/s para treino), suíte de benchmarks e bots heurísticos/especialistas.

---

## 2. Arquitetura da Codebase

```
haxball/
├── core/                  # Motor de física 2D e regras do jogo HaxBall
│   ├── physics_engine.py  # Integração Euleriana (60 FPS), colisões disco-disco e disco-segmento
│   ├── game.py            # Máquina de estados da partida (KickOff, Play, Goal, Placar, Timer)
│   ├── stadium.py         # Parser e geometria de mapas .hbs (segmentos, vértices, gols, planos)
│   ├── disc.py            # Entidade física de discos (jogadores, bola) com raio, massa, amortecimento
│   ├── segment.py         # Paredes lineares e curvas com coeficientes de restituição (bCoef)
│   ├── vector.py          # Vetor 2D (Vec2) otimizado com operações vetoriais
│   └── constants.py       # Constantes globais (Equipes, Dimensões Futsal, Taxa de Chute)
│
├── bots/                  # Coleção de agentes autônomos e heurísticas
│   ├── base_bot.py        # Classe base abstrata para bots
│   ├── archetypes.py      # Perfis táticos especializados (PressBot, StrikerBot, BankBot, etc.)
│   ├── npc.py             # Lógica geométrica de decisão de NPCs de alto nível
│   ├── heuristic_bot.py   # Heurística clássica baseada em regras de perseguição e chute
│   ├── goalie_bot.py      # Agente focado em posicionamento defensivo sob a baliza
│   ├── wall_rebound_bot.py# Bot especialista em tabelas e reflexões nas paredes
│   ├── futsal_3v3_team.py # Orquestração tática de times 3v3 (Zagueiro, Ala, Pivô)
│   └── rl_bot.py          # Wrapper de inferência neural PyTorch (.pt) para modelos treinados
│
├── rl/                    # Módulos de Aprendizado por Reforço e Modelos
│   ├── actions/           # Espaço de ações (18 ações discretas com kick ou contínuas)
│   ├── observations/      # Construtores de observações (Full 61-dim, desacoplado, ego-cêntrico)
│   ├── models/            # Redes neurais (MLP Policy, Entity Attention Transformer)
│   ├── algorithms/        # PPO Trainer, DQN Trainer, Self-Play League
│   ├── rewards/           # Funções de recompensa desenhadas para evitar reward-hacking
│   └── online_bridge/     # Integração com API Headless do HaxBall (node-haxball)
│
├── gym_env/               # Ambientes compatíveis com Gymnasium / OpenAI Gym
│   ├── haxball_env.py     # HaxballEnv (1v1, 3v3, multiagente, step(), reset(), render())
│   └── rewards.py         # Cálculo de recompensas por toque, avanço e gol
│
├── data/                  # Processamento de replays e datasets
│   ├── hbr2_parser.py     # Leitor do protocolo binário .hbr2 do HaxBall
│   ├── dataset_builder.py # Construção de datasets PyTorch a partir de replays
│   └── filtered_replays_1v1/ # Repositório de replays competitivos para treinamento
│
├── dodgeball/             # Variante Queimada (Dodgeball) com mapa especial e física modificada
│   ├── core/              # Regras de eliminação por impacto de bola
│   ├── gym_env/           # Ambiente Gym para Dodgeball
│   ├── bots/              # Bots de esquiva e arremesso
│   └── rl/                # Treinamento específico para Dodgeball
│
├── renderer/              # Renderização e efeitos
│   ├── pygame_renderer.py # Renderizador gráfico de 60 FPS com campo, jogadores e rastro
│   └── sound_effects.py   # Síntese procedural de áudio para chutes e gols
│
├── ui/                    # Interface gráfica do usuário
│   ├── haxball_gui.py     # Hub de gerenciamento, seleção de mapas, bots e checkpoints
│   └── widgets.py         # Componentes customizados da UI
│
├── tools/                 # Ferramentas auxiliares, scrapers e pipelines
│   ├── train_bc_3v3.py    # Script de treinamento de Behavioral Cloning
│   ├── cluster_replays.py # Agrupamento e análise estatística de replays
│   ├── discord_replay_scraper.py # Download automático de partidas em canais do Discord
│   └── thehax_replay_scraper.py  # Coleta de replays do TheHax
│
├── benchmarks/            # Suíte de avaliação quantitativa e logs de performance
│   └── run_benchmarks.py  # Confronto automatizado entre bots com cálculo de taxa de vitória
│
├── docs/                  # Documentação científica, arquitetura e especificações de pesquisa
└── train_rl.py / play.py  # Pontos de entrada para treino de RL e jogo interativo
```

---

## 3. Dinâmica de Física e Regras

1. **Ciclo de Atualização:** 60 passos por segundo ($\Delta t = 1/60\text{ s}$).
2. **Equações Fundamentais:**
   $$\vec{v}_{t+1} = (\vec{v}_t + \vec{u} \cdot a) \cdot d$$
   $$\vec{p}_{t+1} = \vec{p}_t + \vec{v}_{t+1}$$
   - $d_{\text{player}} = 0.96$, $d_{\text{ball}} = 0.99$.
   - $a = 0.1$ ($0.083$ no chute com aceleração reduzida para precisão).
3. **Mecânica de Chute:**
   - Alcance: $\|\vec{p}_{\text{bola}} - \vec{p}_{\text{jogador}}\| \le r_p + r_b + \text{kickMargin}$ ($\approx 4\text{ px}$).
   - Direção radial estrita do centro do jogador para a bola.
   - Força de impulso instantânea $v_{\text{kick}} \approx 5.0\text{ a }5.5\text{ px/frame}$.
   - Cooldown de chute (12-15 ticks) com efeito visual (*kick flash*).
4. **Condução Futsal:** $bCoef_{\text{player}} = 0.0$ garante retenção e controle inelástico rente ao corpo. Paredes com $bCoef \ge 1.25$ geram ricochete tático veloz.

---

## 4. Pipeline de Inteligência Artificial

### A. Behavioral Cloning (BC)
- Utiliza replays `.hbr2` parseados por [`data/hbr2_parser.py`](file:///home/usuario/develop/reinforcement-cases/haxball/data/hbr2_parser.py).
- Espaço de estados com 61 dimensões normalizadas cobrindo posições, velocidades relativas, distâncias a traves e alinhamentos geométricos.
- Ações mapeadas em 18 classes discretas (8 direções + neutro $\times$ [sem chute, com chute]).
- Augmentação de dados com inversão horizontal/vertical para generalização de lados (Vermelho vs Azul).

### B. PPO & Self-Play
- Treinamento no ambiente [`gym_env/haxball_env.py`](file:///home/usuario/develop/reinforcement-cases/haxball/gym_env/haxball_env.py) em modo acelerado sem GUI.
- Ligas de auto-confronto onde o agente enfrenta cópias passadas congeladas do seu próprio modelo para evitar ciclos de esquecimento (*catastrophic forgetting*).
- Função de recompensa balanceada: bônus principal pelo gol, penalidade por sofrer gol, e conformação sutil de recompensa (*potential-based reward shaping*) em direção ao gol adversário sem gerar comportamentos obsessivos de pinball.
