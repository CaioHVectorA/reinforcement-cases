# Reinforcement Learning Cases & Experiments 🧠🎮

Hub e índice de casos práticos e experimentos de **Aprendizado por Reforço (RL)** desenvolvidos por [@CaioHVectorA](https://github.com/CaioHVectorA).

Todos os projetos ativos possuem **repositórios dedicados** para manter ambientes isolados, dependências limpas e pipelines de treino independentes.

---

## 📌 Repositórios Ativos

| Projeto | Repositório | Descrição | Stack | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Haxball RL** ⚽🤖 | [**CaioHVectorA/haxball-rl**](https://github.com/CaioHVectorA/haxball-rl) | Agentes de RL autônomos treinados via PPO self-play com simulação analítica bit-exact em Rust (`HaxballGym`) e deploy direto em salas headless reais. | PyTorch, Rust, Node.js, `uv` | 🟢 Ativo |
| **Snake RL** 🐍 | [**CaioHVectorA/snake-rl**](https://github.com/CaioHVectorA/snake-rl) | Experimentos de RL aplicados ao clássico jogo da cobrinha (representação matricial de estado, Q-learning e DQN). | Python, Pygame | 🟡 Estável |
| **Flappy Bird RL** 🐦 | [**CaioHVectorA/flappy-bird-rl**](https://github.com/CaioHVectorA/flappy-bird-rl) | Experimentos de timing e controle de salto com Gymnasium / Pygame e roadmap para baseline tabular e DQN. | Python, Gymnasium | 🟠 Em reestruturação |

---

## 🚀 Próximos Projetos no Radar (Ideias de Jogos para RL)

Ambientes competitivos em 2D/física arcade selecionados para novos repositórios futuros:

### 1. 🏓 Pong / Air Hockey 2D (`airhockey-rl` / `pong-rl`)
- **Conceito:** Física de rebote com atrito suave, ângulos de corte e tabelas em alta velocidade.
- **Mecânicas de RL:**
  - Self-play competitivo 1v1 (Actor-Critic / PPO).
  - Estratégias emergentes de efeito na bola, variação de velocidade e desarme nas quinas cegas.
  - Deploy web leve via ONNX.js para partidas interativas contra jogadores humanos no navegador.

### 2. 🏐 Slime Volleyball / Slime Soccer (`slime-volley-rl`)
- **Conceito:** O clássico jogo de gosmas físicas saltando e cabeceando a bola sobre a rede sob gravidade parabólica.
- **Mecânicas de RL:**
  - Espaço de ação contínuo ou discreto compacto (mover, pular, cabecear).
  - Controle de timing fino de salto e amortecimento de bola.
  - Curva de aprendizado rápida em PPO para jogadas aéreas e cortadas.

### 3. 💥 Curve Fever / Tron Lightcycles (`curve-fever-rl` / `achtung-rl`)
- **Conceito:** Evolução natural do *Snake* — arena com 4 a 6 cobras contínuas (movimento angular livre) deixando rastros sólidos mortais.
- **Mecânicas de RL:**
  - Disputa multi-agente intensa por controle territorial e sobrevivência.
  - Estratégias emergentes de fechar caminhos, criar espirais de contenção e prever trajetórias dos oponentes.
  - Entrada espacial (visão local ou raycasting com sensores).

---

## 🎯 TODOs dos Repositórios Atuais

### ⚽ Haxball RL
- [ ] Concluir o treino de 4.000 iterações em Futsal 1v1 ($130\text{M}+$ steps).
- [ ] Expandir o self-play para **Futsal 2v2 e 3v3** utilizando a arquitetura de atenção (`EntityAttentionPolicy`).
- [ ] Otimizar inferência do `headless-bot` com WebAssembly (Wasm) ou ONNX Runtime para zero latência em salas públicas de 60 FPS.
- [ ] Implementar ranking Elo dinâmico no bot para avaliar jogadores humanos contra o modelo.

### 🐦 Flappy Bird RL
- [ ] Substituir o emulador Atari por ambiente nativo vetorizado (`flappy-bird-gymnasium` ou Pygame customizado).
- [ ] Implementar Q-Learning Tabular com discretização de $(dx, dy, v_y)$ para baseline instantâneo.
- [ ] Implementar DQN com replay buffer priorizado (PER).
- [ ] Adicionar dashboard de métricas e script de gravação de vídeo/GIF dos melhores voos.

### 🐍 Snake RL
- [ ] Implementar Algoritmo Genético (Neuroevolução) como baseline comparativo ao DQN.
- [ ] Adicionar obstáculos dinâmicos no mapa para avaliar generalização.

---

## 📜 Licença
MIT
