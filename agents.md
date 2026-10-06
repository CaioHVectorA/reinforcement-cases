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

---

## 6. Emergência de "Team Play" em Aprendizado Multiagente (MARL)

Em ambientes multiagente cooperativos como o **HaxBall 2v2**, agentes independentes sem incentivo estruturado sofrem do chamado **Dilema do Futebol de Recreio (*Kindergarten Soccer*)**: todos os jogadores convergem cegamente para a bola ao mesmo tempo, colidindo entre si, bloqueando chutes de companheiros e deixando a retaguarda vulnerável.

Para fazer com que os modelos desenvolvam **inteligência coletiva e jogo em equipe (*team play*)**, estruturamos os seguintes pilares de Aprendizado por Reforço:

### 6.1. Atribuição de Crédito Compartilhada (*Shared Team Reward*)
* A recompensa primária de vitória ($\pm 10.0$) é distribuída igualmente entre **todos os membros do time**.
* Isso alinha a função de valor de Bellman: o objetivo de um jogador não é "fazer gol sozinho", mas sim maximizar a probabilidade de vitória coletiva.

### 6.2. Atribuição de Crédito de Assistência e Passe
* **Detecção de Passe:** Se o jogador $A$ chuta a bola e, nos próximos 50 ticks (~0.8s), o jogador $B$ (seu companheiro) recebe e toca na bola sem interceptação adversária:
  * Jogador $A$ (passador) recebe **$+2.5$**.
  * Jogador $B$ (receptor em velocidade) recebe **$+1.5$**.
* **Assistência Oficial:** Se um gol é marcado após um passe confirmado, o passador original recebe um bônus adicional de **Assistência ($+4.0$)**.

### 6.3. Penalidade de Espaçamento Mútuo (*Anti-Clustering Spacing Penalty*)
* Quando dois companheiros de time se aproximam a menos de $R_{\text{cluster}} = 75$ px, aplica-se uma penalidade contínua:
  $$r_{\text{spacing}} = -w_{\text{space}} \cdot \frac{R_{\text{cluster}} - d(p_1, p_2)}{R_{\text{cluster}}}$$
* Esse sinal quebra a simetria de convergência para a bola: o jogador que está mais longe percebe que entrar na rota do companheiro gera prejuízo, forçando-o a **abrir na ala** ou **ficar na cobertura**.

### 6.4. Diferenciação Dinâmica de Papéis: Pressionador vs. Âncora Defensiva
* **Apenas o Companheiro Mais Próximo** recebe recompensa densa por encurtar a distância até a bola ($r_{\text{approach}}$). O segundo jogador não ganha esse bônus, evitando corrida dupla à bola.
* **Cobertura Defensiva (*Defensive Cover*):** Quando a bola está no campo de defesa, o jogador mais recuado ganha recompensa contínua ($+0.02$/tick) enquanto estiver posicionado entre a bola e o próprio gol.

### 6.5. Compartilhamento de Parâmetros (*Parameter Sharing com Perspectiva Ego*)
* Todos os agentes compartilham a mesma rede neural $\pi_\theta(a|s_{\text{ego}})$. Como a observação de cada jogador é orientada sob sua perspectiva individual (sua posição como origem relativa e gol adversário sempre em $+X$), o mesmo modelo aprende a agir como atacante quando próximo à bola e como âncora/garçom quando distante.

---

## 7. Especificação Completa do Sistema de Recompensas (`TeamPlayRewardEngine`)

O [`TeamPlayRewardEngine`](file:///c:/Users/caihe/Documents/antigravity/agitated-hertz/haxball/rl/rewards/reward_engine.py) calcula as seguintes componentes a cada tick:

| Componente | Magnitude | Condição de Acionamento | Impacto Tático |
|---|---|---|---|
| **Gol Marcado** | $+10.0$ | Bola ultrapassa a baliza adversária | Recompensa esparsa compartilhada pelo time |
| **Gol Sofrido** | $-10.0$ | Bola ultrapassa a própria baliza | Punição compartilhada pelo time |
| **Assistência de Gol** | $+4.0$ | Jogador forneceu o passe que originou o gol | Estimula criação de jogadas coletivas |
| **Passe Concluído** | $+2.5$ | Chute recebido com sucesso por companheiro | Estimula toque de bola e desmarque |
| **Recepção em Movimento**| $+1.5$ | Recepção de passe limpo de companheiro | Incentiva companheiro a se posicionar livre |
| **Desarme / Interceptação**| $+1.2$ | Interromper trajetória de chute adversário | Estimula pressão e recomposição |
| **Aproximação da Bola** | $+0.04 \times \Delta d$ | Apenas para o jogador **mais próximo** da bola | Evita aglomeração mútua na disputa |
| **Penalidade de Spacing**| $-0.03 \times \text{overlap}$ | Companheiros a menos de $75$ px de distância | Força abertura de espaço e triangulação |
| **Cobertura Defensiva** | $+0.02$ / tick | Jogador mais recuado entre bola e próprio gol | Cria automaticamente o papel de "goleiro/âncora" |
| **Progressão ao Gol** | $+0.06 \times v_{\text{goal}}$ | Velocidade da bola projetada em direção ao gol | Estimula ataque vertical |
| **Alinhamento do Chute** | $+0.25 \times \cos(\theta)$ | Chute direcionado à baliza oponente | Melhora pontaria das finalizações |
| **Tabela na Parede** | $+0.40$ | Chute com componente $Y$ expressivo e velocidade alta | Estimula drible por ricochete na parede |

---

## 8. Treinamento Recursivo 2x2 (Self-Play) com Aceleração de até 100x

Na Central de Controle (`main.py`), implementamos o **Modo 2x2 Self-Play Treino ao Vivo**, onde os agentes competem recursivamente em campo:

1. **Aceleração Dinâmica ($1\times$ a $100\times$):**
   * Em $1\times$: Simulação a 60 FPS reais para inspeção visual minuciosa dos movimentos.
   * Em $100\times$: Executa **100 passos de física e inferência por frame** (~6.000 passos/segundo), acumulando rollouts e executando updates PPO em poucos segundos.
2. **Botão "Ficar Burro (Reset)":**
   * Reinicializa todos os pesos neurais para distribuições aleatórias com 1 clique, permitindo visualizar a evolução cognitiva a partir do zero absoluto.
3. **As 4 Fases da Evolução Cognitiva:**
   * **Fase 1: Exploração Burra ($< 5.000$ passos):** Movimento caótico, rotação descontrolada, chutes no vazio e gols contra acidentais.
   * **Fase 2: Perseguição de Bola ($5.000$ a $25.000$ passos):** Os agentes aprendem a correr em direção à bola e a empurrá-la na direção geral do ataque.
   * **Fase 3: Alinhamento ao Gol e Espaçamento ($25.000$ a $70.000$ passos):** Surgem chutes direcionados às traves e separação mútua dos companheiros devido à penalidade de *spacing*.
   * **Fase 4: Team-Play e Passes Coordenados ($> 70.000$ passos):** Troca intencional de passes quando pressionado, finalização após assistência, tabelas nas paredes e cobertura na retaguarda.

---

## 9. Auditoria Empírica de Dados e Viabilidade de Behavioral Cloning (BC)

### 9.1. A Falácia da Mineração Pública em Massa para 1v1
A auditoria minuciosa de mais de 4.500 replays obtidos de servidores comunitários e ligas públicas (`replay.thehax.pl` e canais de log de bots no Discord) comprovou empiricamente que:
* **Escassez Extrema de 1v1:** O ecossistema competitivo de HaxBall online gira em torno de ligas 3v3 (Futsal), 4v4/7v7 (Real Soccer) e 5v5 (Big). Menos de 1% dos replays arquivados por bots de hospedagem correspondem a partidas 1v1 reais com jogadores humanos jogando ativamente.
* **Falso Positivo de Placar ("x1"):** Títulos de replay com termos como `2x1`, `3x1` ou `4x1` representam o placar final do confronto e jamais devem ser interpretados isoladamente como indicadores da modalidade 1v1.
* **Poluição de Salas Vazias:** Servidores de host salvam arquivos a cada intervalo de tempo fixo (2 a 5 minutos), gerando milhares de replays sem entradas de teclas de jogadores humanos.

### 9.2. Protocolo Rigoroso de Validação de Replays (Anti-Alucinação)
Qualquer inclusão de dados no pipeline de Behavioral Cloning exige obrigatoriamente:
1. **Desempacotamento do Cabeçalho Binário HBR2:** Leitura direta do nome do estádio decodificado em memória (`stadium_name`), rejeitando categoricamente mapas com descritores de multiagente (`x3`, `3v3`, `x4`, `4v4`, `x5`, `5v5`, `x7`, `real soccer`, `voley`).
2. **Contagem Efetiva de Jogadores:** Verificação da lista de jogadores por frame. O número de atletas ativos em campo deve ser estritamente $1 \text{ Red}$ e $1 \text{ Blue}$.
3. **Volume Mínimo de Inputs Ativos:** Descarte compulsório de gravações com menos de 100 mudanças de estado de controle.

### 9.3. Diretriz de Desenvolvimento para Agentes 1v1
Para evitar desperdício de tempo e alucinações com scraping de fontes descontroladas:
* **Abordagem Principal:** O treinamento de agentes 1v1 apoia-se em **PPO Self-Play e currículo de baselines analíticos (`HeuristicBot`, `WallReboundBot`)**. O aprendizado por reforço descobre a mecânica ótima diretamente do motor de física sem depender de bancos de dados públicos viciados.
* **Abordagem Secundária (BC Controlado):** Caso Behavioral Cloning seja empregado, os dados devem originar-se de sessões intencionais gravadas pelo próprio operador/usuário ou geradas sinteticamente por controladores especialistas, garantindo 100% de integridade e relevância.


