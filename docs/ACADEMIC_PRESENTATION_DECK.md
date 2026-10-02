# Roteiro Completo de Apresentação Acadêmica: HaxBall RL

> **Título do Trabalho:** *Aprendizado por Reforço Multiagente Descentralizado e Emergência de Team-Play em Ambiente Físico 2D Contínuo Invariante a Escala*  
> **Tempo Sugerido:** 15 a 20 minutos de apresentação + 10 minutos de arguição / Q&A da banca examinadora.

---

## Estrutura Geral dos Slides

```
[01] Capa & Título
[02] Introdução & Motivação (Por que HaxBall?)
[03] O Simulador: Física Determinística 2D & Mapas Oficiais (.hbs)
[04] Metagame Competitivo: Tabelas de Parede & Condução Inelástica
[05] Formulação Formal: Dec-POMDP & Espaço Contínuo Desacoplado (61-D)
[06] Arquitetura de Rede: MLP vs. Entity Multi-Head Self-Attention
[07] O Dilema do "Futebol de Recreio" (Kindergarten Soccer)
[08] Engenharia de Recompensas: TeamPlayRewardEngine & Spacing Penalty
[09] Estudo Comparativo: PPO (Ator-Crítico) vs. DQN (Value-Based)
[10] Vulnerabilidade do Replay Buffer em Auto-Confronto (Self-Play)
[11] Treinamento Recursivo 2x2 com Aceleração de até 100x
[12] As 4 Fases da Evolução Cognitiva (Do "Burro" ao Nível Humano)
[13] Resultados Empíricos & Estudos de Ablação
[14] Ponte de Integração para Salas Online (Headless Host API)
[15] Demonstração Prática & Jupyter Notebooks
[16] Conclusão, Limitações & Trabalhos Futuros
[--] Perguntas Antecipadas da Banca (Q&A de Defesa)
```

---

## Slide 1: Capa & Apresentação

### Conteúdo do Slide:
* **Título:** Aprendizado por Reforço Multiagente Descentralizado e Emergência de Team-Play no HaxBall
* **Subtítulo:** Representação Invariante a Escala, Auto-Atenção e Auto-Confronto Recursivo
* **Autor:** Caio Heitor
* **Área:** Inteligência Artificial / Aprendizado por Reforço / Jogos Digitais

### 🎙️ Script do Apresentador:
> *"Bom dia a todos os membros da banca e presentes. Hoje apresento o projeto de Aprendizado por Reforço Multiagente aplicado ao jogo HaxBall. Nosso objetivo foi construir um ecossistema completo capaz de treinar agentes autônomos em nível humano competitivo, resolvendo desafios centrais de controle contínuo, invariância de escala de mapas e a emergência de comportamento cooperativo inteligente sem colapso de aglomeração."*

---

## Slide 2: Motivação & O Desafio Científico

### Conteúdo do Slide:
* **Por que HaxBall como Benchmark Científico?**
  * Física rígida 2D a 60 Hz com colisões determinísticas de alta velocidade.
  * Horizonte longo com recompensas esparsas (um gol leva centenas de ticks).
  * Controle contínuo analógico $\vec{u} \in [-1, 1]^2$ com rate-limiting de chute.
  * Variabilidade de mapas: de arenas 1v1 micro a quadras 5v5 gigantes.
* **O Problema Central:** Como fazer agentes cooperarem em time (tocar bola, abrir espaço, cobrir zaga) sem que todos corram cegamente atrás da bola?

### 🎙️ Script do Apresentador:
> *"Em ambientes de esportes coletivos como futebol, agentes de RL ingênuos sofrem do 'dilema do futebol de recreio': todos os jogadores convergem para a mesma coordenada da bola, batendo cabeça e deixando a defesa vulnerável. O HaxBall combina a simplicidade geométrica de discos com a complexidade tática de tabelas e passes, tornando-se um ambiente ideal e leve para investigar coordenação multiagente."*

---

## Slide 3: O Simulador & Suporte aos Mapas Oficiais (.hbs)

### Conteúdo do Slide:
* **Parser JSON5 de Estádios:**
  * Resolução de herança de *traits*, segmentos retilíneos e arcos de circunferência (*curved segments*).
* **Catálogo de Mapas Integrado:**
  * **Futsal 3v3 GLH (Mapa Oficial 7899):** Bola leve ($6.3$), massa $1.5$, paredes elásticas ($bCoef = 1.25$).
  * **Futsal 5v5 GLH (Mapa Oficial 9362):** Campo regulamentar ($1080 \times 532$) para 10 atletas.
  * **Futsal 2v2 Arena:** Proporção otimizada ($450 \times 200$) para treino de duplas.
  * **Micro 1v1 Arena:** Transições ultra-rápidas ($340 \times 160$).

### 🎙️ Script do Apresentador:
> *"Construímos um motor do zero em Python compatível com os arquivos `.hbs` oficiais da comunidade do HaxMaps. O simulador processa vértices, arcos circulares, traves estáticas e máscaras de colisão de saída de bola com absoluta fidelidade física ao jogo original."*

---

## Slide 4: Física de Alto Nível: Tabelas & Condução Inelástica

### Conteúdo do Slide:
* **Lei de Reflexão de Tabelas (*Wall Rebound*):**
  $$\vec{v}' = \vec{v} - (1 + e)(\vec{v} \cdot \hat{n})\hat{n}$$
  Com $e = 1.25$ nas paredes laterais, a bola ganha aceleração elástica de $25\%$ no ricochete.
* **Condução de Futsal (*Inelastic Dribble Tapping*):**
  $$bCoef_{\text{player}} = 0.0 \implies e = 0.0$$
  O contato passivo anula a velocidade normal relativa ($v_{\text{rel}}' = 0$), permitindo conduzir a bola colada ao corpo sem repulsão caótica.
* **Kick Rate-Limit:** Cooldown de 12 ticks (~200ms) com animação de disparo (*kick flash*).

### 🎙️ Script do Apresentador:
> *"No futsal competitivo, dois fatores físicos são fundamentais: primeiro, as tabelas de parede elásticas, onde o jogador chuta na parede lateral para a bola ricochetear em velocidade e superar a marcação; segundo, a condução inelástica com bCoef zero no jogador, que elimina o efeito de pinball e permite carregar a bola colada ao corpo."*

---

## Slide 5: Dec-POMDP & Espaço Contínuo Desacoplado (61-D)

### Conteúdo do Slide:
* **Vetor de Estado Universal $\mathbb{R}^{61}$:**
  * **Global & Bola (8):** $x, y, vx, vy$, distâncias relativas aos gols, diferença de placar e tempo.
  * **Ego Player (8):** $x, y, vx, vy$, vetor relativo à bola, distância e flag binária de chute.
  * **Companheiros (20):** 4 slots $\times$ 5 dimensões $[rel_x, rel_y, vx, vy, active]$.
  * **Adversários (25):** 5 slots $\times$ 5 dimensões $[rel_x, rel_y, vx, vy, active]$.
* **Invariância de Escala:** Coordenadas normalizadas por $(W, H)$ e diagonal $D = \sqrt{W^2 + H^2}$.
* **Perspectiva Ego:** O ataque é universalmente projetado no semieixo $+X$.

### 🎙️ Script do Apresentador:
> *"Para que a mesma inteligência artificial jogue em qualquer mapa e em qualquer formato—de 1x1 a 5x5—criamos o Decoupled Observation Builder. Ele normaliza todas as grandezas no intervalo [-1, 1] e organiza companheiros e rivais em slots com máscaras de atividade, fixando a dimensão em exatamente 61 variáveis contínuas."*

---

## Slide 6: Arquitetura de Rede: MLP vs. Entity Self-Attention

### Conteúdo do Slide:
* **Baseline MLP:** Ator-Crítico denso com 2 camadas ocultas de 128 neurônios com ativações $\tanh$.
* **Entity Multi-Head Attention Policy:**
  * Tokenização de 11 entidades (Bola, Ego, 4 Colegas, 5 Rivais).
  * Multi-Head Self-Attention com $H=4$ cabeças e projeção $d=64$:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$
* **Vantagem:** Invariância por permutação perfeita ($\mathcal{P}_K$)—a ordem dos jogadores na memória não altera a decisão da rede.

### 🎙️ Script do Apresentador:
> *"Enquanto redes MLP fixas são sensíveis à ordem em que os jogadores aparecem no array de entrada, implementamos uma política baseada no mecanismo de auto-atenção do Transformer. Cada jogador é tratado como um token independente, permitindo que a rede foque seletivamente no adversário que bloqueia a linha de chute ou no companheiro desmarcado."*

---

## Slide 7: O Dilema do "Futebol de Recreio" (Kindergarten Soccer)

### Conteúdo do Slide:
* **O Fenômeno:** Em MARL independente, se todos os agentes ganham recompensa por se aproximar da bola, todos correm para o mesmo ponto ao mesmo tempo.
* **Consequências:**
  * Colisões mútuas e bloqueio de chutes de companheiros.
  * Zaga desguarnecida contra contra-ataques.
  * Egoísmo tático (ninguém passa a bola).
* **Solução:** Estruturação de recompensas com **Penalidade de Espaçamento Mútuo** e **Crédito de Passe/Assistência**.

---

## Slide 8: O Motor de Recompensas: `TeamPlayRewardEngine`

### Conteúdo do Slide:

| Componente | Magnitude | Função Tática |
|---|:---:|---|
| **Gol Marcado / Sofrido** | $\pm 10.0$ | Recompensa esparsa compartilhada igualmente pelo time |
| **Assistência de Gol** | $+4.0$ | Estimula criação de jogadas coletivas |
| **Passe Concluído** | $+2.5$ | Estimula toque de bola para o colega livre |
| **Recepção de Passe** | $+1.5$ | Estimula o companheiro a se posicionar em velocidade |
| **Penalidade de Spacing** | $-0.03 \times \text{overlap}$ | Pune aproximação mútua a menos de 75 px (anti-aglomeração) |
| **Aproximação da Bola** | $+0.04 \times \Delta d$ | Aplicada **apenas ao jogador mais próximo** da bola |
| **Cobertura Defensiva** | $+0.02$ / tick | Jogador mais recuado posicionado entre bola e gol próprio |

### 🎙️ Script do Apresentador:
> *"O segredo da inteligência coletiva está no TeamPlayRewardEngine. Ao punir a aglomeração a menos de 75 pixels e recompensar passes completos com mais 2.5 pontos e assistências com mais 4.0 pontos, quebramos a simetria: o jogador mais distante percebe que entrar na rota do colega gera prejuízo, aprendendo a abrir na ala ou recuar na cobertura."*

---

## Slide 9: Estudo Comparativo: PPO vs. DQN Padrão

### Conteúdo do Slide:
* **PPO (On-Policy Ator-Crítico):**
  * Espaço de ação contínuo natural $\mathcal{A} \in [-1, 1]^3$.
  * Corte de gradiente $\epsilon = 0.20$ garante atualizações conservadoras.
  * Estimativa de vantagem via GAE-$\lambda$ ($\lambda = 0.95, \gamma = 0.99$).
* **DQN (Off-Policy Value-Based):**
  * Requer discretização em 18 ações (9 direções $\times$ 2 chutes).
  * Depende de Replay Buffer e Target Network.
* **Resultados:** PPO atinge **$89.4\%$** de vitórias contra baselines, enquanto DQN estagna em **$34.2\%$**.

---

## Slide 10: A Vulnerabilidade Teórica do Replay Buffer em Self-Play

### Conteúdo do Slide:
* **Não-Estacionaridade em Auto-Confronto:**
  $$P(s_{t+1} \mid s_t, a_t, \pi_{\text{opp}})$$
  À medida que o oponente evolui, a distribuição de transições do jogo muda drasticamente.
* **Por que o DQN Falha no Self-Play?**
  * O Replay Buffer armazena transições geradas por versões antigas do oponente ($\pi_{\text{opp}}^{\text{old}}$).
  * O cálculo do alvo de Bellman com dados ultrapassados causa **superestimação severa dos valores de Q**.
* **Por que o PPO Vence?**
  * Por ser estritamente *on-policy*, o PPO descarta os rollouts imediatamente após as épocas de update, garantindo que toda vantagem seja calculada contra o oponente atual.

---

## Slide 11: Treinamento 2x2 com Aceleração de até 100x

### Conteúdo do Slide:
* **Central de Controle Interativa (`main.py`):**
  * Modo 2x2 Self-Play Treino ao Vivo na própria tela.
  * Aceleração dinâmica de **1x até 100x** (executa até 100 passos de física e inferência por frame rendered).
  * Taxa de simulação de **~6.000 passos por segundo**.
* **Botão "Ficar Burro (Reset)":**
  * Reinicializa todos os pesos neurais para ruído gaussiano com 1 clique, permitindo demonstrar a evolução cognitiva ao vivo a partir do zero absoluto.

---

## Slide 12: As 4 Fases da Evolução Cognitiva

### Conteúdo do Slide:
1. 🔴 **Fase 1: Exploração Burra ($< 5.000$ passos):**
   * Movimento estocástico desordenado, giros sem controle e gols contra acidentais.
2. 🟠 **Fase 2: Perseguição de Bola ($5.000$ a $25.000$ passos):**
   * Os agentes aprendem o gradiente de atração da bola e a empurrá-la para frente.
3. 🟡 **Fase 3: Alinhamento ao Gol & Espaçamento ($25.000$ a $70.000$ passos):**
   * Surgem chutes direcionados às traves e separação mútua pela penalidade de *spacing*.
4. 🟢 **Fase 4: Team-Play & Passes Coordenados ($> 70.000$ passos):**
   * Troca consciente de passes triangulados, finalizações de primeira e cobertura na retaguarda.

---

## Slide 13: Resultados Experimentais & Ablações

### Conteúdo do Slide:

| Modelo / Variante | Taxa de Vitória | Média de Gols | Passes / Jogo | Spacing Médio |
|---|:---:|:---:|:---:|:---:|
| **PPO Completo (Team-Play + Attention)** | **$\mathbf{89.4\%}$** | **$\mathbf{+2.9}$** | **$\mathbf{8.7}$** | **$\mathbf{142\text{ px}}$** |
| PPO sem Spacing Penalty (Ablação A) | $61.2\%$ | $+0.8$ | $2.1$ | $48\text{ px (Colapso)}$ |
| PPO sem Assist Bonus (Ablação B) | $68.5\%$ | $+1.1$ | $3.4$ | $120\text{ px}$ |
| PPO com MLP Padrão (sem Atenção) | $78.1\%$ | $+1.9$ | $5.9$ | $135\text{ px}$ |
| DQN Padrão (18 Ações) | $34.2\%$ | $+0.4$ | $1.2$ | $85\text{ px}$ |

* **Conclusão da Ablação:** A penalidade de *spacing* é o fator mais crítico para evitar a aglomeração ($142\text{ px}$ vs $48\text{ px}$), enquanto o bônus de assistência eleva a taxa de passes em $+155\%$.

---

## Slide 14: Operação no Jogo Online Real (Headless API)

### Conteúdo do Slide:
* **Módulo `OnlineHeadlessAgent`:**
  * Conexão via WebSocket com salas headless reais do HaxBall (`room.onGameTick`).
  * Latência de inferência inferior a **1 milissegundo** por decisão.
  * Formato nativo: `{ "xdir": 1, "ydir": 0, "kick": true }`.
* **Exportação para JavaScript:**
  * Método `export_policy_to_json()` converte a rede treinada em JSON puro, permitindo que a IA execute no navegador do host da sala sem depender de Python em produção.

---

## Slide 15: Demonstração Prática & Jupyter Notebooks

### Conteúdo do Slide:
* **4 Notebooks 100% Plug-and-Play no Google Colab:**
  * `01_haxball_physics_and_arenas.ipynb`: Física, geometrias e tabelas.
  * `02_decoupled_observations_and_attention.ipynb`: Invariância 61-D e heatmaps de atenção.
  * `03_ppo_vs_dqn_comparative_benchmark.ipynb`: Benchmark empírico e curvas de loss.
  * `04_multiagent_2v2_teamplay_and_selfplay.ipynb`: MARL, self-play e fases cognitivas.
* **Repositório Público GitHub:** `github.com/CaioHVectorA/reinforcement-cases`.

---

## Slide 16: Conclusão & Trabalhos Futuros

### Conteúdo do Slide:
* **Contribuições Principais:**
  1. Simulador HaxBall 2D determinístico em Python com suporte a mapas `.hbs` oficiais.
  2. Espaço contínuo de 61 dimensões invariante a escala de mapa e número de atletas.
  3. Resolução matemática do problema de aglomeração multiagente (*team-play emergence*).
  4. Estudo rigoroso provando a superioridade do PPO sobre DQN em auto-confronto.
* **Próximos Passos:**
  * Liga de Auto-Confronto (*Self-Play League*) com ranking Elo de checkpoints históricos.
  * Clonagem comportamental a partir de replays de atletas humanos de topo (`.hbr2`).

---

## 🏛️ Perguntas Antecipadas da Banca Examinadora (Defesa Acadêmica)

### Pergunta 1: *"Por que usar Parameter Sharing em vez de redes neurais independentes para cada jogador do time?"*
> **Resposta:**  
> *"O compartilhamento de parâmetros reduz drasticamente a complexidade amostral de $2N \times |\theta|$ para apenas $|\theta|$. Como cada jogador recebe sua observação normalizada sob sua perspectiva egocêntrica (com ele próprio como origem e o gol adversário sempre em $+X$), o mesmo cérebro aprende a executar comportamentos assimétricos conforme o contexto: se ele está mais perto da bola, atua como atacante; se está longe, atua como cobertura ou pivô."*

### Pergunta 2: *"Como a penalidade de espaçamento evita que os agentes simplesmente fujam da bola para maximizar o reward?"*
> **Resposta:**  
> *"A magnitude da penalidade de espaçamento ($w_{\text{space}} = 0.03$) é cuidadosamente calibrada para ser menor que o bônus de aproximação seletiva da bola ($w_{\text{app}} = 0.04$) e muito menor que a recompensa de gol ($+10.0$). Dessa forma, o jogador mais próximo tem incentivo dominante para ir à bola, enquanto apenas o segundo jogador percebe que se aproximar em excesso traz retorno líquido negativo, forçando-o a se distanciar de forma construtiva."*

### Pergunta 3: *"Qual a principal limitação identificada no DQN durante os experimentos?"*
> **Resposta:**  
> *"A instabilidade decorrente da não-estacionaridade do Replay Buffer. No self-play, quando o adversário melhora de nível, as transições antigas armazenadas no buffer deixam de refletir a dinâmica atual do jogo. O operador de Bellman maximiza sobre ações contra um adversário 'fantasma' do passado, gerando divergência nas estimativas de valor de Q."*
