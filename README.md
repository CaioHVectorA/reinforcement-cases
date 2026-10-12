# Reinforcement Learning Cases & Experiments 🧠🎮

Hub e índice de casos práticos e experimentos de **Aprendizado por Reforço (RL)** desenvolvidos por [@CaioHVectorA](https://github.com/CaioHVectorA).

Todos os projetos foram organizados em **repositórios dedicados** para garantir ambientes isolados, rastreamento limpo de dependências e pipelines de treino independentes:

---

## 📌 Repositórios dos Projetos

| Projeto | Repositório | Descrição | Stack | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Haxball RL** ⚽🤖 | [**CaioHVectorA/haxball-rl**](https://github.com/CaioHVectorA/haxball-rl) | Agentes de RL autônomos treinados via PPO self-play com simulação analítica bit-exact em Rust (`HaxballGym`) e deploy direto em salas headless reais. | PyTorch, Rust, Node.js, `uv` | 🟢 Ativo / Treinamento contínuo |
| **Snake RL** 🐍 | [**CaioHVectorA/snake-rl**](https://github.com/CaioHVectorA/snake-rl) | Experimentos de RL aplicados ao clássico jogo da cobrinha (representação matricial de estado, Q-learning e DQN). | Python, Pygame | 🟡 Estável |
| **Flappy Bird RL** 🐦 | [**CaioHVectorA/flappy-bird-rl**](https://github.com/CaioHVectorA/flappy-bird-rl) | Experimentos de timing e controle de salto com Gymnasium / Pygame e roadmap para PPO / DQN tabular. | Python, Gymnasium | 🟠 Em reestruturação |

---

## 🎯 Visão Geral de Objetivos e TODOs Globais

### 1. Haxball RL ⚽
- [ ] Concluir o treino de 4.000 iterações em Futsal 1v1 ($130\text{M}+$ steps).
- [ ] Expandir o self-play para **Futsal 2v2 e 3v3** utilizando a arquitetura de atenção (`EntityAttentionPolicy`).
- [ ] Otimizar inferência do `headless-bot` com WebAssembly (Wasm) ou ONNX Runtime para zero latência em salas públicas de 60 FPS.
- [ ] Implementar ranking Elo dinâmico no bot para avaliar jogadores humanos contra o modelo.

### 2. Snake RL 🐍
- [ ] Implementar algoritmo genético como baseline comparativo ao DQN.
- [ ] Adicionar mapa com obstáculos dinâmicos e testes de generalização do agente.
- [ ] Benchmark comparando entradas matriciais brutas ($40 \times 30$) vs. vetores de distâncias/sensores espaciais.

### 3. Flappy Bird RL 🐦
- [ ] Substituir o emulador Atari por ambiente nativo vetorizado (`flappy-bird-gymnasium` ou Pygame customizado).
- [ ] Implementar Q-Learning Tabular com discretização de $(dx, dy, v_y)$ para baseline instantâneo.
- [ ] Implementar DQN com replay buffer priorizado (PER).
- [ ] Adicionar dashboard de métricas e script de gravação de vídeo/GIF dos melhores voos.

---

## 📜 Licença
MIT
