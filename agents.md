# Arquitetura e Especificação de Agentes: HaxBall RL

> Este documento apresenta a especificação técnica completa dos agentes, o espaço de observações desacoplado de dimensões de mapa e número de jogadores, e um **estudo comparativo aprofundado entre a abordagem PPO e o Aprendizado por Reforço Padrão (DQN/Value-Based)**.

---

## 1. Separação Modular da Arquitetura

O ecossistema de aprendizado foi modularizado para desacoplar algoritmos, projeto de ação, observações e modelos:

```
haxball/rl/
├── observations/
│   ├── base.py                 # Interface abstrata BaseObservationBuilder
│   └── decoupled_obs.py        # Observador universal desacoplado (1v1, 3v3, 5v5)
├── actions/
│   └── action_space.py         # Handlers: Contínuo, Discreto, MultiDiscreto e Headless
├── rewards/
│   └── reward_engine.py        # Modelagem de recompensas (potencial, gols, tabelas)
├── models/
│   ├── mlp_policy.py           # Rede Ator-Crítico densa (MLP)
│   └── entity_attention.py     # Rede baseada em Auto-Atenção (Transformer)
├── algorithms/
│   ├── ppo/
│   │   └── ppo_trainer.py      # Proximal Policy Optimization com GAE-lambda
│   └── standard_rl/
│       └── dqn_trainer.py      # Deep Q-Network com Replay Buffer e Target Net
└── online_bridge/
    └── headless_agent.py       # Ponte de integração com o jogo online real (Headless Host)
```

---

## 2. Espaço de Observação Desacoplado de Escala e Jogadores

Para permitir que a mesma inteligência atue em mapas pequenos ou gigantes (ex.: Futsal 3v3 [mapa 7899] ou Futsal 5v5 [mapa 9362]) e em formatos de **1x1, 2x2, 3x3 ou 5x5**, criamos o `DecoupledObservationBuilder`:

### 2.1. Invariância de Dimensão de Mapa
Todas as coordenadas cartesianas $(x, y)$ são divididas pelas semi-dimensões úteis da quadra $(W, H)$ onde $W = \text{stadium.bg\_width}$ e $H = \text{stadium.bg\_height}$.
* Velocidades são divididas por $v_{\text{max}} = 15.0$.
* Distâncias são divididas pela diagonal da quadra $D = \sqrt{W^2 + H^2}$.
* Todas as grandezas pertencem ao intervalo contínuo $[-1.0, 1.0]$.

### 2.2. Invariância do Número de Jogadores (Estrutura de Entidades com Máscara)
O vetor de estado totaliza **61 dimensões contínuas**:

| Bloco | Dimensão | Descrição |
|---|---|---|
| **Global & Bola** | 8 | $x, y, vx, vy$ da bola (orientados ao ataque), distâncias relativas aos gols, diferença de placar e tempo decorrido. |
| **Ego (Agente)** | 8 | $x, y, vx, vy$ do próprio jogador, vetor relativo à bola, distância e flag binária de alcance de chute (`can_kick`). |
| **Colegas de Time** | 20 (4 slots $\times$ 5) | Até 4 companheiros mais próximos: $[rel_x, rel_y, vx, vy, \text{active}]$. Slots não utilizados (ex.: em 1v1 ou 3v3) recebem $\text{active} = 0.0$. |
| **Adversários** | 25 (5 slots $\times$ 5) | Até 5 oponentes mais próximos: $[rel_x, rel_y, vx, vy, \text{active}]$. Slots não utilizados recebem $\text{active} = 0.0$. |

### 2.3. Arquitetura com Auto-Atenção (`EntityAttentionPolicy`)
Para alcançar invariância por permutação perfeita entre jogadores, implementamos uma rede que trata cada jogador e a bola como tokens vetoriais independentes passados por uma camada de **Multi-Head Self-Attention**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
Isso permite que a atenção da rede se concentre dinamicamente no adversário que bloqueia a linha de chute ou no companheiro livre para passe.

---

## 3. Estudo Comparativo: PPO vs. Aprendizado por Reforço Padrão (DQN/Value-Based)

Para o problema do HaxBall, comparamos rigorosamente duas linhagens de RL:

```
                    LINHAGENS DE APRENDIZADO
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
      ┌─────────────────────┐     ┌─────────────────────┐
      │         PPO         │     │     DQN Padrão      │
      │  (Policy-Gradient)  │     │    (Value-Based)    │
      └─────────────────────┘     └─────────────────────┘
      • On-Policy Ator-Crítico    • Off-Policy Q-Learning
      • Controle Contínuo         • Espaço Discreto (18)
      • Estável em Multiagente    • Replay Buffer Clássico
      • Auto-Confronto Robusto    • Instável com Não-Estacionaridade
```

### 3.1. Estudo da Linha PPO (*Proximal Policy Optimization*)

* **Natureza Algorítmica:** On-policy, Ator-Crítico.
* **Objetivo Sub-Rogrado Cortado (*Clipped Surrogate Objective*):**
  $$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$
  onde $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ é a razão de probabilidades.
* **Estimativa de Vantagem com GAE ($\lambda$):**
  $$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V, \quad \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$
* **Vantagens para o HaxBall:**
  1. **Controle Contínuo Natural:** Permite saídas suaves de aceleração analógica $(a_x, a_y) \in [-1, 1]^2$ e disparo probabilístico.
  2. **Robustez no Auto-Confronto (*Self-Play*):** Em cenários multiagente onde o adversário muda sua política a cada rodada, o ambiente torna-se não-estacionário. Algoritmos on-policy com updates conservadores como o PPO não sofrem com dados históricos desatualizados.
  3. **Estabilidade de Gradiente:** O corte $\epsilon=0.20$ impede que um episódio atípico destrua os pesos aprendidos.

### 3.2. Estudo da Linha de RL Padrão (*Deep Q-Network - DQN*)

* **Natureza Algorítmica:** Off-policy, puramente baseado em valor (*Value-Based*).
* **Equação de Bellman para $Q^*(s, a)$:**
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
* **Perda com Rede Alvo (*Target Network*):**
  $$L(\theta) = \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}} \left[ \left( r + \gamma (1-d) \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a) \right)^2 \right]$$
* **Discretização das Ações:**
  Requer um espaço discreto de 18 ações (9 orientações direcionais $\times$ 2 estados de chute).
* **Vantagens e Desafios no HaxBall:**
  1. **Eficiência Amostral Teórica:** Como é *off-policy*, pode reutilizar transições antigas armazenadas no *Replay Buffer*.
  2. **Problema da Não-Estacionaridade em Self-Play:** Quando o adversário evolui, as transições antigas no buffer pertencem a uma distribuição de jogo ultrapassada, gerando superestimação severa dos valores de $Q$.
  3. **Discretização do Controle:** Dribles refinados e micro-ajustes angulares tornam-se truncados pelas 9 direções discretas.

### 3.3. Tabela Comparativa de Desempenho no HaxBall

| Critério | PPO (Ator-Crítico) | DQN (Padrão) |
|---|---|---|
| **Espaço de Ação** | Contínuo $\mathcal{A} \in [-1, 1]^3$ ou Discreto | Estritamente Discreto (18 ações) |
| **Estabilidade em Self-Play** | **Altíssima** (updates conservadores) | Moderada / Baixa (buffer corrompido) |
| **Controle de Micro-Ângulos** | Fluido e analógico | Escalonado em passos de 45° |
| **Uso de Memória** | Leve (buffer de rollout temporário) | Pesado (Replay Buffer de 50.000+ passos) |
| **Convergência para Tabelas** | Rápida com bônus de potencial | Lenta devido à exploração $\epsilon$-greedy |

---

## 4. O Zoo de Baselines Analíticos

O repositório inclui três oponentes heurísticos que balizam o treinamento:

1. **`HeuristicBot`:** Antecipa 6 ticks de trajetória da bola, contorna para não marcar contra e ataca a baliza.
2. **`WallReboundBot` ("Tabela Master"):** Detecta se a rota direta para o gol está bloqueada por defensores. Se estiver, calcula a reflexão óptica na parede lateral e finaliza por tabela.
3. **`GoalieBot`:** Goleiro especialista em corte de ângulo angular, patrulhando a bissetriz entre bola e traves.

---

## 5. Arquitetura para Operação no Jogo Online

O módulo [`headless_agent.py`](file:///c:/Users/caihe/Documents/antigravity/agitated-hertz/haxball/rl/online_bridge/headless_agent.py) viabiliza o deploy em salas online reais:

1. **Recepção de Estado:** Conecta-se via WebSocket/HTTP ao script de host do HaxBall (`room.onGameTick`).
2. **Inferência em Tempo Real:** Executa a rede neural treinada em menos de 1 milissegundo.
3. **Comando de Saída:** Envia comando nativo aceito pela API da sala:
   ```json
   { "xdir": 1, "ydir": 0, "kick": true }
   ```
4. **Exportação JavaScript:** Permite converter pesos `.pt` para formato JSON puro, possibilitando que a rede neural rode diretamente no navegador do host da sala sem depender de Python em produção.
