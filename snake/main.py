import torch
import numpy as np
from game import SnakeEnv
from agent import Agent
import pygame

# Configurações
EPISODES = 1000  # Número total de episódios de treinamento
BATCH_SIZE = 64  # Tamanho do lote para replay
RENDER = True  # Definir como True para ver o jogo sendo jogado pelo agente

# Inicializa o ambiente e o agente
env = SnakeEnv()
state_size = 40 * 30  # Matriz do jogo (40x30 células)
action_size = 4  # 4 direções possíveis (cima, baixo, esquerda, direita)
agent = Agent(state_size, action_size)

# Loop de treinamento
for episode in range(EPISODES):
    state = env.reset().flatten()  # Obtém o estado inicial e achata para vetor
    total_reward = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        if RENDER:
            env.render()  # Mostra o jogo na tela
        
        # O agente escolhe uma ação
        action_index = agent.act(state)
        action = env.actions[action_index]  # Converte índice para string ("up", "down", etc.)

        # Executa a ação no ambiente
        next_state, reward, done = env.step(action)
        next_state = next_state.flatten()

        # Salva experiência na memória do agente
        agent.remember(state, action_index, reward, next_state, done)

        # Atualiza o estado
        state = next_state
        total_reward += reward

        # Se o jogo acabou
        if done:
            print(f"🏆 Episódio {episode + 1}/{EPISODES} - Recompensa: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")
            break

    # Treina o agente com as experiências passadas
    agent.replay(BATCH_SIZE)

# Fecha o jogo depois do treinamento
pygame.quit()
