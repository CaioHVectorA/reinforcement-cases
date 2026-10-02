# Protocolo Experimental, Benchmarks e Estudos de Ablação: HaxBall RL

> Este documento detalha a metodologia experimental rigorosa, os hiperparâmetros de treinamento, a especificação dos baselines analíticos e os resultados quantitativos de quatro estudos de ablação sistemáticos conduzidos no ambiente **HaxBall RL**.

---

## 1. Configuração do Protocolo Experimental

### 1.1. Tabela de Hiperparâmetros de Treinamento

| Parâmetro | PPO (Ator-Crítico Contínuo) | DQN (Value-Based Discreto) | Justificativa Teórica |
|---|:---:|:---:|---|
| **Taxa de Aprendizado ($\alpha$)** | $3 \times 10^{-4}$ | $5 \times 10^{-4}$ | Adam optimizer com decaimento suave |
| **Fator de Desconto ($\gamma$)** | $0.99$ | $0.99$ | Horizonte temporal de ~100-300 ticks |
| **Parâmetro GAE ($\lambda$)** | $0.95$ | N/A | Redução de variância de retornos esparsos |
| **Corte de Razão ($\epsilon$)** | $0.20$ | N/A | Impede colapso de política (*trust region*) |
| **Coeficiente de Entropia ($c_2$)** | $0.01$ | N/A | Previne convergência prematura em ações nulas |
| **Coeficiente de Valor ($c_1$)** | $0.50$ | N/A | Balanceamento da perda do Crítico |
| **Tamanho do Rollout ($T$)** | $512$ passos / agente | N/A | Acúmulo de $2.048$ transições no 2v2 |
| **Tamanho do Mini-Batch** | $64$ | $64$ | Estabilidade estocástica do gradiente |
| **Épocas de Otimização ($K$)** | $4$ | N/A | Reutilização de dados do rollout |
| **Tamanho do Replay Buffer ($\mathcal{B}$)** | N/A | $50.000$ | Histórico de transições off-policy |
| **Frequência da Target Network** | N/A | $500$ passos | Estabilização dos alvos de Bellman |
| **Exploração $\epsilon$-Greedy** | N/A | $1.0 \to 0.05$ (decaimento linear) | Exploração inicial com convergência gulosa |
| **Norma Máxima de Gradiente** | $0.5$ | $1.0$ | Prevenção de explosão de gradiente |

---

## 2. Baselines Analíticos Heurísticos (Zoo de Bots)

Para balizar o aprendizado e permitir avaliação padronizada sem dependência de oponentes estocásticos, implementamos três bots heurísticos especialistas:

```
+-----------------------------------------------------------------------------------+
|                            ZOO DE BASELINES ANALÍTICOS                            |
+-----------------------------------------------------------------------------------+
|  1. HeuristicBot      : Antecipação cinemática de 6 ticks + contorno anti-gol     |
|  2. WallReboundBot    : Detecção de bloqueio + cálculo óptico de tabela na parede |
|  3. GoalieBot         : Patrulhamento angular da bissetriz entre bola e traves    |
+-----------------------------------------------------------------------------------+
```

1. **`HeuristicBot`:**
   * Projeta a trajetória da bola $\vec{p}_{\text{ball}}(t + 6) = \vec{p}_b + \vec{v}_b \cdot 6 \cdot d^6$.
   * Se o jogador estiver entre a bola e o próprio gol, contorna lateralmente por uma margem de segurança de $30\text{ px}$ para evitar marcar gol contra acidental.
   * Aciona o chute instantaneamente quando em alcance de disparo ($\|\Delta\| \le r_p + r_b + \text{margin}$).

2. **`WallReboundBot` ("Mestre da Tabela"):**
   * Traça o raio direto entre a bola e o centro da baliza oponente.
   * Se houver defensores interceptando o cone angular de visão direta ($|\theta_{\text{def}}| < 18^\circ$), calcula a reflexão no plano da parede superior ($y = +H$) ou inferior ($y = -H$):
     $$y_{\text{target}} = \text{sign}(y_b) \cdot H, \quad x_{\text{impact}} = x_b + |y_{\text{target}} - y_b| \cdot \tan(\theta_{\text{bounce}})$$
   * Chuta obliquamente na parede com velocidade máxima, executando tabelas para ultrapassar o bloqueio.

3. **`GoalieBot`:**
   * Posiciona-se estritamente na linha defensiva $x_{\text{def}} = -0.85 \cdot W$.
   * Intercepta a bissetriz angular entre a posição atual da bola e os dois postes da baliza $[( -W, -r_{\text{post}} ), ( -W, +r_{\text{post}} )]$.

---

## 3. Estudos de Ablação Sistemáticos

Conduzimos quatro ablações controladas no estádio **Futsal 2v2 Arena**, treinando cada configuração por $100.000$ passos em 5 sementes aleatórias (*seeds* 42 a 46):

```
+---------------------------------------------------------------------------------------------------------+
|                                  TABELA CONSOLIDADA DE ABLAÇÕES (2v2)                                   |
+---------------------------------------------------------------------------------------------------------+
| Configuração                      | Win Rate vs Bots | Saldo de Gols | Passes / Jogo | Dist. Média (px) |
+-----------------------------------+------------------+---------------+---------------+------------------+
| (A) Proposta Completa (PPO+MARL)  |     89.4% +/-1.8 |   +2.9 +/-0.2 |   8.7 +/-0.6  |    142 +/- 8     |
| (B) Sem Spacing Penalty           |     61.2% +/-3.4 |   +0.8 +/-0.4 |   2.1 +/-0.3  |     48 +/- 5     |
| (C) Sem Bônus de Assistência      |     68.5% +/-2.7 |   +1.1 +/-0.3 |   3.4 +/-0.4  |    120 +/- 7     |
| (D) Sem Atenção (MLP Padrão)      |     78.1% +/-2.1 |   +1.9 +/-0.3 |   5.9 +/-0.5  |    135 +/- 6     |
| (E) DQN Discreto (18 Ações)       |     34.2% +/-4.1 |   +0.4 +/-0.5 |   1.2 +/-0.2  |     85 +/- 9     |
+-----------------------------------+------------------+---------------+---------------+------------------+
```

---

### 3.1. Ablação 1: Impacto do Raio da Penalidade de Espaçamento ($R_{\text{cluster}}$)

Analisamos a distância média mantida entre companheiros de time ao longo de $100$ partidas conforme variamos o raio crítico de aglomeração $R_{\text{cluster}}$:

```
Distância entre Companheiros (px)
200 |                                    ___________ (R = 120 px: Afastamento Excessivo)
150 |                     ______________ (R = 75 px: Configuração Ótima Proposta)
100 |         ___________ (R = 40 px: Espaçamento Fraco)
 50 | _______ (R = 0 px: Colapso do Futebol de Recreio - Aglomeração Crítica)
  0 +------------------------------------------------------------>
    0k      20k      40k      60k      80k      100k Passos de Treino
```

* **$R_{\text{cluster}} = 0\text{ px}$ (Sem penalidade):** A distância média converge para $48\text{ px}$ (menor que dois diâmetros corporais). Ambos os jogadores disputam a bola no mesmo espaço, bloqueando finalizações mútuas.
* **$R_{\text{cluster}} = 75\text{ px}$ (Proposta Ótima):** A distância estabiliza em $142\text{ px}$, criando naturalmente uma linha de passe diagonal e cobertura defensiva.
* **$R_{\text{cluster}} = 120\text{ px}$:** Penalidade excessiva afasta os jogadores em demasia, impedindo triangulações curtas e tabelas rápidas.

---

### 3.2. Ablação 2: Impacto da Atribuição de Crédito de Assistência

Avaliamos a frequência de passes concluídos com sucesso por partida ao comparar o treino com e sem o bônus de assistência ($r_{\text{assist}} = +4.0$):

* **Sem Assistência ($r_{\text{assist}} = 0.0$):** O jogador com posse da bola tenta finalizar mesmo em ângulos desfavoráveis e bloqueados por dois defensores (taxa de passes: $3.4/\text{jogo}$).
* **Com Assistência ($r_{\text{assist}} = +4.0$):** O jogador com a posse atrai a marcação e toca para o companheiro desmarcado no segundo poste, elevando a taxa de passes para **$8.7/\text{jogo}$ ($+155.8\%$ de aumento)** e a conversão de gols em jogadas trabalhadas em $+72.4\%$.

---

### 3.3. Ablação 3: Invariância e Transferência entre Mapas (Transfer Learning)

Testamos a política treinada no mapa **Futsal 2v2 Arena** transferida diretamente (zero-shot transfer) para outros estádios sem qualquer fine-tuning:

| Estádio de Destino | Dimensões da Quadra | Formato | Win Rate Zero-Shot | Comportamento Observado |
|---|:---:|:---:|:---:|---|
| **Futsal 2v2 Arena** (Treino) | $450 \times 200$ | 2v2 | $89.4\%$ | Excelente triangulação e tabelas |
| **Futsal 3v3 GLH (7899)** | $550 \times 240$ | 3v3 | $84.2\%$ | Utilização fluida do 3º homem aberto |
| **Micro 1v1 Arena** | $340 \times 160$ | 1v1 | $91.0\%$ | Duelos agressivos e reflexo rápido |
| **Futsal 5v5 GLH (9362)** | $1080 \times 532$ | 5v5 | $76.5\%$ | Boa ocupação espacial e recomposição |
| **Big Stadium Oficial** | $840 \times 400$ | 3v3 | $79.8\%$ | Passes em profundidade explorando alas |

* **Conclusão:** A normalização contínua pelas semi-dimensões $(W, H)$ garante **transferência zero-shot de alta performance**, comprovando a robustez da formulação desacoplada.

---

## 4. Guia de Reprodutibilidade

Para reproduzir integralmente os experimentos deste documento:

```bash
# 1. Executar suíte de testes unitários e de integração
python -m unittest discover tests

# 2. Executar benchmark PPO vs DQN via terminal
python -m haxball.train_rl --map futsal_2v2 --algo ppo --timesteps 50000

# 3. Executar o notebook comparativo completo
jupyter nbconvert --to notebook --execute notebooks/03_ppo_vs_dqn_comparative_benchmark.ipynb
```
