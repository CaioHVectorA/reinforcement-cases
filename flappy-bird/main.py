import numpy as np
import gymnasium as gym


import matplotlib.pyplot as plt

# Criando o ambiente CartPole
env = gym.make('ALE/Tennis-v5', render_mode='human')


num_episodios = 1000
max_timesteps = 200
learning_rate = 0.1
gamma = 0.99  # fator de desconto
def escolher_acao(state):
    return env.action_space.sample()
for ep in range(num_episodios):
    state, info = env.reset()
    done = False
    for t in range(max_timesteps):
        # Renderiza o ambiente (mostra o CartPole se movendo)
        env.render()

        # Escolher ação baseada no estado
        action = escolher_acao(state)

        # Realizar ação e obter o próximo estado e recompensa
        next_state, reward, done, truncated, info = env.step(action)

        # Aqui, você pode adicionar lógica de atualização do agente,
        # como políticas baseadas em Q-learning ou redes neurais

        state = next_state

        if done or truncated:
            print(f"Finalizado episódio {ep+1} com {t+1} timesteps.")
            break

# Fechar o ambiente ao final
env.close()