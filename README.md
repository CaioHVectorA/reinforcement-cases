# HaxBall RL: Caso de Estudo em Aprendizado por Reforço

> Estudo de caso de Aprendizado por Reforço (RL) focado no jogo **HaxBall**, simulando física precisa em 2D, suporte a estádios da comunidade (`.hbs`), baselines analíticos de alta performance (*tryhards*), interface gráfica completa estilo HaxBall e pipeline de treinamento para atingir nível sobre-humano.

---

## ⚽ O que é o Projeto?

Este repositório contém uma réplica fiel do **HaxBall** desenvolvida especificamente para pesquisa e experimentação em Aprendizado por Reforço Contínuo e Multiagente:

- **Física 2D Fiel:** Colisões elásticas e inelásticas, amortecimento suave de futsal ($bCoef = 0$ no jogador para condução colada), recarga de chute (*kick rate-limiting*), segmentos curvos e traves sólidas.
- **Controle Híbrido Amigável:** O jogador pode se mover usando **WASD ou Setas do teclado** simultaneamente, e chutar com `Espaço`, `X`, `C` ou `Shift`.
- **Mapas Oficiais da Comunidade:**
  - **Futsal 3x3 GLH (Mapa 7899 por Bazinga!):** O mapa mais jogado do cenário competitivo, com bola ultraleve ($6.3$), massa $1.5$ e paredes de tabela elástica ($bCoef = 1.25$).
  - **Futsal 5v5 GLH (Mapa 9362 por Bazinga!):** Campo ampliado ($1080 \times 532$) para duelos de 10 atletas.
  - **Classic Stadium:** O campo tradicional de grama.
  - **Dodgeball Arena:** Modalidade de queimada com barreira divisória central restrita.
- **Espaço de Observação Desacoplado:** Vetor de 61 dimensões totalmente normalizado pelo tamanho da quadra e estruturado com slots de entidades (bola, ego, colegas e oponentes) com máscaras ativas, operando perfeitamente em **1v1, 2v2, 3v3 e 5v5** com a mesma rede neural.
- **Rede Neural com Auto-Atenção (Transformer):** `EntityAttentionPolicy` com camadas de Multi-Head Attention para invariância à permutação de jogadores.
- **Central de Controle Gráfica (GUI Completa):** Interface nativa estilizada com o visual do HaxBall, permitindo alternar mapas, configurar times (1v1 a 5v5), pausar, reiniciar e **iniciar/parar treinamentos de IA diretamente pela UI** sem precisar de linhas de comando.
- **Dois Estudos Algorítmicos:**
  - **PPO (Proximal Policy Optimization):** Abordagem *on-policy* Ator-Crítico com GAE-$\lambda$ e controle contínuo.
  - **DQN Padrão (Deep Q-Network):** Abordagem clássica *value-based* com Replay Buffer e Target Network para estudo comparativo.
- **Ponte para Jogo Online:** Módulo `OnlineHeadlessAgent` compatível com a API Headless do HaxBall online (`room.setPlayerInputs`) e exportador de pesos para JavaScript.

---

## 🚀 Como Executar

### 1. Iniciar a Central de Controle Gráfica (Recomendado)

Basta executar o script principal:

```bash
python main.py
```

A interface abre a sala de jogo completa com:
- **Aba Partida / Sala:** Jogue diretamente no teclado (WASD ou Setas + Espaço/X), pause, reinicie e alterne entre 1v1, 3v3 e 5v5.
- **Aba Selecionar Mapa:** Escolha instantaneamente entre Futsal 3v3 (7899), Futsal 5v5 (9362), Classic ou Dodgeball.
- **Aba Central de RL:** Inicie e acompanhe o treinamento de RL em tempo real (PPO vs DQN, métricas de recompensa e taxa de vitória).
- **Aba Controles:** Manual completo de jogabilidade e táticas de *tryhard* (tabelas e corte de ângulo).

### 2. Executar via Linha de Comando (Opcional)

#### Treinar Agente via Terminal:
```bash
python -m haxball.train_rl --map futsal --bot wall --timesteps 50000
```

#### Executar o pipeline do notebook Colab localmente:
```bash
# Teste rápido: 2 iterações curtas
python train_haxball_rl_local.py --iterations 2 --steps 512 --eval-matches 1

# Treino completo equivalente ao notebook
python train_haxball_rl_local.py --iterations 1000 --steps 4096
```

Durante o treino, o script gera um painel semelhante ao notebook com recompensa média,
win rate, losses PPO, entropia, gauntlet e desempenho contra cada bot. Por padrão ele
salva a imagem sem abrir uma janela em `checkpoints/training_dashboard.png`. Para abrir
o painel ao vivo, use:

```bash
python train_haxball_rl_local.py --plot-mode live
```

Em servidor/headless, desative completamente os gráficos com:

```bash
python train_haxball_rl_local.py --plot-mode off
```

O treinador local detecta CPU/CUDA automaticamente, aceita `--device cpu` ou `--device cuda`,
salva os modelos em `checkpoints/` e grava as métricas em
`checkpoints/training_metrics.json`. `haxball_rl_best.pt` contém os melhores pesos para
inferência; `haxball_rl_latest.pt` contém também o otimizador, scheduler, iteração e
métricas para retomada. Para continuar, use `--warm-start checkpoints/haxball_rl_latest.pt`.
O checkpoint latest é atualizado a cada iteração e também é salvo ao encerrar com `Ctrl+C`.

#### Executar Bateria de Testes Automatizados:
```bash
python -m unittest tests/test_haxball.py
```

---

## 📚 Documentação do Projeto

- [**context.md**](file:///c:/Users/caihe/Documents/antigravity/agitated-hertz/context.md): Estudo aprofundado do HaxBall, física 2D detalhada, análise dos mapas oficiais 7899 e 9362, estratégias do meta competitivo e arquitetura de integração online.
- [**agents.md**](file:///c:/Users/caihe/Documents/antigravity/agitated-hertz/agents.md): Especificação dos agentes, observação de 61 dimensões, estudo comparativo rigoroso PPO vs. DQN, dinâmica de Team-Play e especificação completa de recompensas.
- [**todo.md**](file:///c:/Users/caihe/Documents/antigravity/agitated-hertz/todo.md): Roadmap e marcos de desenvolvimento do projeto.

---

## 📓 Esquema de Jupyter Notebooks (`notebooks/`)

O repositório inclui 4 notebooks didáticos, analíticos e executáveis:

1. **[`01_haxball_physics_and_arenas.ipynb`](notebooks/01_haxball_physics_and_arenas.ipynb):**
   - Comparativo visual de mapas (.hbs): Futsal 2v2, Futsal 3v3 GLH (7899), Futsal 5v5 GLH (9362) e Micro 1v1.
   - Simulação de trajetória e demonstração da lei de reflexão elástica em paredes (*tabelas* com $bCoef = 1.25$).
   - Condução inelástica de futsal ($bCoef = 0.0$) e *kick rate-limit*.
2. **[`02_decoupled_observations_and_attention.ipynb`](notebooks/02_decoupled_observations_and_attention.ipynb):**
   - Espaço contínuo de 61 dimensões normalizado pelas semi-dimensões $(W, H)$ e diagonal $D$.
   - Verificação empírica de invariância para 1v1, 2v2, 3v3 e 5v5 com slots mascarados.
   - Rede neural `EntityAttentionPolicy` e mapas de calor de auto-atenção (*Attention Heatmaps*).
3. **[`03_ppo_vs_dqn_comparative_benchmark.ipynb`](notebooks/03_ppo_vs_dqn_comparative_benchmark.ipynb):**
   - Benchmark comparativo lado a lado: PPO (Ator-Crítico Contínuo) vs DQN (Value-Based Discreto de 18 ações).
   - Curvas de Recompensa Média, Taxa de Vitórias, Função de Perda (*Loss*) e estabilidade de treinamento.
4. **[`04_multiagent_2v2_teamplay_and_selfplay.ipynb`](notebooks/04_multiagent_2v2_teamplay_and_selfplay.ipynb):**
   - Aprendizado Multiagente (MARL) e o dilema do "Futebol de Recreio".
   - Modelagem de recompensas com `TeamPlayRewardEngine` (passes, assistências, penalidade de *spacing* e âncora).
   - Treino 2v2 recursivo (*Self-Play*) com Parameter Sharing e transição pelas 4 Fases Cognitivas.
