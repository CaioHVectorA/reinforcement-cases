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

#### Executar Bateria de Testes Automatizados:
```bash
python -m unittest tests/test_haxball.py
```

---

## 📚 Documentação do Projeto

- [**context.md**](file:///c:/Users/caihe/Documents/antigravity/agitated-hertz/context.md): Estudo aprofundado do HaxBall, física 2D detalhada, análise dos mapas oficiais 7899 e 9362, estratégias do meta competitivo e arquitetura de integração online.
- [**agents.md**](file:///c:/Users/caihe/Documents/antigravity/agitated-hertz/agents.md): Comparação teórica e empírica entre **PPO** e **RL Padrão (DQN)**, especificação do espaço contínuo desacoplado (61 variáveis), rede com Auto-Atenção e zoo de baselines.
- [**todo.md**](file:///c:/Users/caihe/Documents/antigravity/agitated-hertz/todo.md): Roteiro de desenvolvimento com todas as etapas concluídas e próximos passos para ligas de auto-confronto e deploy online.
