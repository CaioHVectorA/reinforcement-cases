# Contexto do Projeto: Estudo de Caso HaxBall em Aprendizado por Reforço

> **Repositório:** `reinforcement-cases` / HaxBall RL  
> **Domínio:** Aprendizado por Reforço Contínuo, Controle Multiagente e Simulação de Física 2D  
> **Objetivo Final:** Treinar um agente de Inteligência Artificial via Aprendizado por Reforço (RL) capaz de competir em nível humano de alta performance (*tryhard*) no jogo **HaxBall** e ser implantado em salas online através da API Headless.

---

## 1. O que é o HaxBall?

O **HaxBall** é um jogo online multijogador em tempo real criado em 2010 por Mario Carbajal. Embora visualmente minimalista — assemelhando-se a futebol de botão em visão superior (top-down 2D) —, possui uma profundidade mecânica e tática extraordinária, comparável a ambientes de ponta da literatura como *Rocket League* e *RoboCup 2D Simulation*.

No jogo, cada jogador controla um disco rígido que se desloca livremente no plano $(x, y)$ usando o teclado (WASD ou Setas) e aciona o chute (teclas `Espaço`, `X`, `C` ou `Shift`). Não existem animações decorativas: o motor de física 2D dita integralmente a jogabilidade a 60 ticks por segundo. O objetivo é conduzir a bola até o gol adversário e proteger a própria baliza.

---

## 2. Por que o HaxBall é um Caso de Estudo Primoroso para RL?

1. **Espaço Contínuo Desacoplado de Escala:**
   Partidas podem acontecer em quadras pequenas ou estádios gigantescos, e em formatos variados: **1v1**, **2v2**, **3v3** (futsal clássico) ou **5v5** (futsal de campo expandido). Um agente moderno deve generalizar para qualquer dimensão de mapa e qualquer número de atletas sem exigir redes neurais separadas.
2. **Horizonte Longo e Recompensas Esparsas:**
   Um gol pode levar mais de mil ticks de troca de passes, dribles e marcações. O problema de atribuição de crédito (*credit assignment*) torna essencial a formulação de recompensas densas e currículos de treino.
3. **Ambiente Não-Estacionário Multiagente (Self-Play):**
   Adversários adaptam-se constantemente. Bots estáticos tornam-se triviais para redes neurais explorarem falhas locais. O auto-confronto (*Self-Play League*) é obrigatório para produzir estratégias robustas.
4. **Física Determinística com Tabelas de Parede:**
   O rebote nas paredes é o núcleo do jogo técnico (*tryhard*). Jogadores utilizam cálculos geométricos instantâneos de reflexão para tabelar na parede e superar adversários.

---

## 3. Dinâmica da Física Oficial do HaxBall

A física segue as equações determinísticas de conservação de momento, restituição elástica e atrito viscoso (*damping*) a 60 FPS:

### 3.1. Movimento e Aceleração
A cada frame $\Delta t = \frac{1}{60}\,\text{s}$:
$$\vec{v}_{t+1} = (\vec{v}_t + \vec{u} \cdot a) \cdot d$$
$$\vec{p}_{t+1} = \vec{p}_t + \vec{v}_{t+1}$$

* $\vec{u}$: Vetor unitário direcional do jogador (WASD / Setas).
* $a$: Aceleração (`acceleration`). Se o botão de chute estiver pressionado, assume `kickingAcceleration` (reduzida, permitindo maior precisão na condução).
* $d$: Amortecimento (`damping`).

### 3.2. Mecânica do Chute e Limite de Taxa (*Kick Rate Limit*)
* **Alcance de Chute:** Registrado quando:
  $$\|\vec{p}_{\text{bola}} - \vec{p}_{\text{jogador}}\| \le r_p + r_b + \text{kickMargin}$$
* **Direção do Impulso:** Depende exclusivamente do vetor unitário radial do centro do jogador ao centro da bola:
  $$\hat{k} = \frac{\vec{p}_{\text{bola}} - \vec{p}_{\text{jogador}}}{\|\vec{p}_{\text{bola}} - \vec{p}_{\text{jogador}}\|}$$
* **Impulso:**
  $$\vec{v}_{\text{bola}} \leftarrow \vec{v}_{\text{bola}} + \hat{k} \cdot \text{kickStrength}$$
* **Kick Cooldown:** Para evitar impulsos múltiplos indevidos por segundo caso a tecla fique pressionada, o motor aplica um *kick cooldown* de aproximadamente 12 a 15 ticks (~200ms) e aciona o *kick flash* (anel branco luminoso).

### 3.3. Contato sem Chute (Condução / Drible)
Em mapas competitivos de Futsal, o jogador possui `bCoef = 0.0`. Portanto, quando a bola toca no jogador sem que o chute seja acionado:
$$e = \text{bCoef}_{\text{jogador}} \times \text{bCoef}_{\text{bola}} = 0.0$$
A velocidade normal relativa anula-se. A bola **não rebate descontroladamente**, permitindo condução colada ao corpo (*dribble tapping*).

---

## 4. Estudo de Mapas Oficiais da Comunidade (`.hbs`)

Analisamos e incorporamos os principais mapas competitivos oficiais da comunidade mundial (HaxMaps):

### 4.1. Futsal 3x3 GLH (Mapa 7899 por Bazinga!)
* **Referência:** [haxmaps.com/map/7899](https://haxmaps.com/map/7899)
* **Dimensões:** Campo $648 \times 270$, área útil $550 \times 240$, distância de spawn $350$.
* **Física da Bola:** Raio $6.3$, massa inversa $1.5$ (bola ultraleve e veloz), amortecimento $0.99$, $bCoef = 0.4$.
* **Física do Jogador:** $bCoef = 0$ (sem rebatida de pinball entre jogadores), aceleração $0.11$, $kickingAcceleration = 0.083$, força de chute $4.5$.
* **Paredes com Efeito Elástico:** Segmentos com $bCoef$ de $1.25$ a $2.0$, criando o icônico ricochete de contra-ataque de futsal.

### 4.2. Futsal 5x5 GLH (Mapa 9362 por Bazinga!)
* **Referência:** [haxmaps.com/map/9362](https://haxmaps.com/map/9362)
* **Dimensões:** Campo expandido $1080 \times 532$, área útil $950 \times 460$, distância de spawn $310$.
* **Foco Tático:** Desenvolvido para partidas de 10 jogadores (5v5). O espaço amplo exige inteligência posicional, cobertura e leitura de passe em profundidade.

### 4.3. Dodgeball Arena (Queimada)
* Campo fechado com barreira central que impede travessia de jogadores (`cMask: ["red", "blue"]`), permitindo a livre passagem da bola. Ideal para benchmark de reflexos e arremessos balísticos.

---

## 5. Integração com o Jogo Online (HaxBall Headless Host API)

O objetivo final da pesquisa é colocar a IA para disputar partidas reais em salas online públicas ou de ligas.

A arquitetura foi desenhada para conectar-se nativamente com a **API Headless do HaxBall** (`node-haxball` ou script de sala em navegador):

```
 ┌──────────────────────┐                     ┌──────────────────────┐
 │ Sala HaxBall Online  │                     │  Agente RL (Python)  │
 │ (Headless Host API)  │                     │   (Modelo Treinado)  │
 └──────────┬───────────┘                     └──────────┬───────────┘
            │                                            │
            │─── 1. onGameTick (posições/velocidades) ──▶│
            │                                            │
            │◀── 2. setPlayerInputs ({xdir, ydir, kick}) │
            │                                            │
```

O repositório fornece o módulo [`headless_agent.py`](file:///c:/Users/caihe/Documents/antigravity/agitated-hertz/haxball/rl/online_bridge/headless_agent.py), capaz de:
1. Traduzir o estado da sala para o vetor normalizado desacoplado.
2. Calcular a ação e exportá-la no formato nativo: `{ xdir: -1 | 0 | 1, ydir: -1 | 0 | 1, kick: boolean }`.
3. Exportar pesos da rede neural em formato JSON leve para execução direta em JavaScript sem latência de rede.
