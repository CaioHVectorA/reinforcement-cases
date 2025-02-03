import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# Definição da rede neural
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Sem ativação para valores Q

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=24000)
        self.gamma = 0.85  # Fator de desconto
        self.epsilon = 1.0  # Probabilidade de exploração
        self.epsilon_min = 0.01 # Valor mínimo de epsilon
        self.epsilon_decay = 0.999 # Taxa de decaimento de epsilon
        self.learning_rate = 0.008 # Taxa de aprendizado
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_size))  # Escolhe ação aleatória
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()  # Escolhe melhor ação

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(torch.tensor(next_state, dtype=torch.float32)))
            predicted = self.model(torch.tensor(state, dtype=torch.float32))[action]
            # loss = self.criterion(predicted, torch.tensor(target))
            loss = self.criterion(predicted, torch.tensor(target, dtype=torch.float32))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
