# Roadmap de Desenvolvimento: HaxBall RL

Este documento organiza as etapas concluídas e o plano de evolução do projeto para levar agentes de Aprendizado por Reforço ao nível humano competitivo em **HaxBall**.

---

## Status Geral do Projeto

- [x] **Fase 1: Motor de Física, Colisões e Parser de Estádios (.hbs)**
- [x] **Fase 2: Suporte aos Mapas Oficiais Competitivos (Futsal 3v3 GLH 7899 e Futsal 5v5 GLH 9362)**
- [x] **Fase 3: Espaço Contínuo Desacoplado de Escala e Número de Jogadores (1v1 a 5v5)**
- [x] **Fase 4: Separação Modular da Arquitetura (Ações, Observações, Modelos e Algoritmos)**
- [x] **Fase 5: Estudo e Implementação PPO vs. RL Padrão (DQN/Value-Based)**
- [x] **Fase 6: Interface Gráfica Completa (GUI Control Center) com Visual HaxBall**
- [x] **Fase 7: Controle Híbrido (WASD + Setas do Teclado) e Modos de Jogo**
- [x] **Fase 8: Ponte de Integração para Salas Online (HaxBall Headless Host API)**
- [ ] **Fase 9: Treinamento em Larga Escala e Liga de Auto-Confronto (*Self-Play League*)**
- [ ] **Fase 10: Avaliação Humana e Clonagem Comportamental com Replays (`.hbr2`)**

---

## Detalhamento das Entregas Realizadas

### Física e Controles
- [x] Suporte unificado de teclado: jogador pode se mover usando **WASD ou Setas do teclado** simultaneamente.
- [x] Múltiplos botões de chute (`Espaço`, `X`, `C`, `Shift`).
- [x] Amortecimento autêntico de futsal: quando a bola toca o jogador sem chute, $bCoef = 0$, permitindo condução colada ao corpo (*dribble tapping*).
- [x] *Kick rate-limiting*: trava de repetição de chute com recarga de 12 ticks e feedback luminoso (*kick flash*).
- [x] Convenção oficial de gols: baliza com `team: "red"` defendida pelo time vermelho (gol para o azul).

### Mapas Oficiais
- [x] **Futsal 3v3 GLH (Mapa 7899 por Bazinga!):** Quadra $648 \times 270$, bola leve $6.3$, massa $1.5$, paredes de tabela com $bCoef = 1.25$.
- [x] **Futsal 5v5 GLH (Mapa 9362 por Bazinga!):** Campo expandido $1080 \times 532$, projetado para 10 jogadores.
- [x] **Classic Stadium:** Campo clássico de grama.
- [x] **Dodgeball Arena:** Modalidade de queimada com barreira central restrita a jogadores.

### Arquitetura de IA e RL Desacoplada
- [x] `DecoupledObservationBuilder`: 61 dimensões contínuas normalizadas pelo tamanho do estádio e slots estruturados de companheiros e adversários com máscara binária ativa.
- [x] Suporte automático a qualquer formação: **1v1, 2v2, 3v3 e 5v5** sob a mesma rede neural.
- [x] `EntityAttentionPolicy`: Rede baseada em Auto-Atenção (Transformer) invariante a permutações de jogadores.
- [x] `ActionHandler`: Decodificador contínuo, discreto, multidiscreto e formatador para API de salas online.
- [x] `PPOTrainer`: Algoritmo on-policy Ator-Crítico com GAE-$\lambda$.
- [x] `DQNTrainer`: Algoritmo off-policy padrão com Replay Buffer e Target Network para comparação teórica e empírica.
- [x] `OnlineHeadlessAgent`: Ponte para conexão com scripts de salas online (`room.setPlayerInputs`) e exportação de pesos para JavaScript.

### Central de Controle (GUI HaxBall)
- [x] Aplicação visual unificada (`main.py` e `haxball/play.py`).
- [x] Placar autêntico HaxBall com caixas de time vermelho, tempo central e time azul.
- [x] Abas dedicadas:
  - **⚽ Partida / Sala:** Jogo ao vivo, alternância rápida 1v1 / 3v3 / 5v5, pause e controle de bots.
  - **🗺 Selecionar Mapa:** Catálogo visual de mapas (.hbs).
  - **🧠 Central de RL:** Iniciar e parar treinamento RL em segundo plano via interface gráfica, alternar entre PPO e DQN, e visualizar métricas em tempo real (passos, recompensa, vitórias e perda).
  - **🎮 Controles:** Manual completo de atalhos e estratégias de alto nível (*tabelas*, *fintas* e *corte de ângulo*).

---

## Próximos Passos (Evolução Contínua)

1. **Liga de Auto-Confronto (*Self-Play League*):**
   - Criar pool histórico com os 10 melhores checkpoints para evitar ciclagem e esquecimento catastrófico.
   - Computar rating Elo entre as diferentes gerações da rede neural.
2. **Vetorização em Alta Velocidade:**
   - Implementar `gymnasium.vector.AsyncVectorEnv` para paralelizar 16 a 64 partidas headless simultâneas por núcleo de CPU.
3. **Implantation Online no Jogo Oficial:**
   - Criar template de bot em Node.js (`haxball-headless`) consumindo a ponte Python/JSON para rodar em salas públicas online do HaxBall.
4. **Behavioral Cloning:**
   - Desenvolver leitor de replays `.hbr2` para inicializar a rede neural com jogadas de atletas humanos de topo.
