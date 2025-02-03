import torch
import json
from src.consts import MULTIPLIER, maxHeight, maxWidth
import numpy as np
from game import SnakeEnv
from agent import Agent
import pygame

# Configurações
def main():
    EPISODES = 50000  # Número total de episódios de treinamento
    BATCH_SIZE = 128  # Tamanho do lote para replay
    RENDER = True  # Definir como True para ver o jogo sendo jogado pelo agente
    rewards_list = []
    # Inicializa o ambiente e o agente
    env = SnakeEnv()
    # State será mudado para [[dx,dy,distance to food, is food left, is food right, is food up, is food down, is wall left, is wall right, is wall up, is wall down]]
    state_size = 11  # Número de parâmetros no estado
    action_size = 3  # 3 direções possíveis (virar a esquerda, virar a direita, fazer nada)
    agent = Agent(state_size, action_size)

    # Loop de treinamento
    for episode in range(EPISODES):
        env.epoch = episode + 1  # Atualiza a época na UI
        state = np.array(env.reset()).flatten()  # Obtém o estado inicial e achata para vetor
        total_reward = 0
        while True:
            if RENDER:
                env.render()  # Mostra o jogo na tela
            
            # O agente escolhe uma ação
            action_index = agent.act(state)
            action = env.actions[action_index]  # Converte índice para string ("up", "down", etc.)

            # Executa a ação no ambiente
            next_state, reward, done = env.step(action)
            next_state = np.array(next_state).flatten()

            # Salva experiência na memória do agente
            agent.remember(state, action_index, reward, next_state, done)

            # Atualiza o estado
            state = next_state
            total_reward += reward

            # Se o jogo acabou
            if done:
                # save reward arr if episode is multiple of 100
                if episode % 100 == 0:
                    rewards_list.append(total_reward)
                    import os
                    os.makedirs('./snake/out', exist_ok=True)
                    json.dump(rewards_list, open("./snake/out/rewards.json", "w"))
                rewards_list.append(total_reward)
                print(f"🏆 Episódio {episode + 1}/{EPISODES} - Recompensa: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")
                break

        # Treina o agente com as experiências passadas
        agent.replay(BATCH_SIZE)
    # Fecha o jogo depois do treinamento
    pygame.quit()
main()