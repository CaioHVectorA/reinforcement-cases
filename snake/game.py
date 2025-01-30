import numpy as np
from src.consts import MULTIPLIER
import pygame
from src.matrix import generate_game_matrix
import random
from src.gen_food import generate_food
from src.consts import BLACK, BLOCK_SIZE, GREEN, RED, WHITE, WIDTH, HEIGHT, MULTIPLIER

class SnakeEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Snake Game - RL')

        self.clock = pygame.time.Clock()
        self.speed = 600
        self.font = pygame.font.Font(None, 36)

        self.actions = ["left", "right", "up", "down"]
        self.epoch = 0
        self.total_reward = 0
        self.current_steps = 0
        self.max_steps = 100  # Limite inicial de passos
        self.reset()

    def reset(self):
        """Reinicia o jogo com novo cálculo do limite de tempo"""
        self.x, self.y = WIDTH // 2, HEIGHT // 2
        self.dx, self.dy = BLOCK_SIZE, 0
        self.snake = [[self.x, self.y]]
        self.snake_length = 3
        self.food_x, self.food_y = generate_food(self.snake)
        self.done = False
        self.total_reward = 0
        self.current_steps = 0
        
        # Aumenta o limite a cada 100 épocas
        self.max_steps = 100 + ((max(self.epoch, 1) - 1) // 100) * 100
        return self.get_state()

    def step(self, action):
        """Executa uma ação e retorna (novo estado, recompensa, done)."""
        # Inicializa a recompensa com o valor padrão
        reward = -0.02  # Penalização base por movimento

        # Processa a ação
        if action == "left" and self.dx == 0:
            self.dx, self.dy = -BLOCK_SIZE, 0
        elif action == "right" and self.dx == 0:
            self.dx, self.dy = BLOCK_SIZE, 0
        elif action == "up" and self.dy == 0:
            self.dx, self.dy = 0, -BLOCK_SIZE
        elif action == "down" and self.dy == 0:
            self.dx, self.dy = 0, BLOCK_SIZE

        self.x += self.dx
        self.y += self.dy

        # Verifica colisão com as paredes
        if self.x >= WIDTH or self.x < 0 or self.y >= HEIGHT or self.y < 0:
            self.done = True
            reward = -6  # Sobrescreve a recompensa padrão

        # Verifica colisão com o corpo
        for segment in self.snake[:-1]:
            if segment == [self.x, self.y]:
                self.done = True
                reward = -10  # Sobrescreve a recompensa padrão

        # Atualiza a cobra
        self.snake.append([self.x, self.y])
        if len(self.snake) > self.snake_length:
            del self.snake[0]

        # Verifica se comeu a comida
        if self.x == self.food_x and self.y == self.food_y:
            self.food_x, self.food_y = generate_food(self.snake)
            self.snake_length += 1
            reward = 10  # Sobrescreve a recompensa padrão

        # Verifica timeout
        self.current_steps += 1
        if not self.done and self.current_steps >= self.max_steps:
            print("Timeout!")
            self.done = True
            reward = -1  # Penalidade por timeout

        self.total_reward += reward
        return self.get_state(), reward, self.done

    def get_state(self):
        """Retorna o estado atual como uma matriz 40x30."""
        matrix = generate_game_matrix(
            snake_segments=self.snake,
            food_pos=(self.food_x, self.food_y)
        )
        return np.array(matrix)

    def render(self):
        """Desenha o jogo na tela e exibe informações da UI."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill(BLACK)
        pygame.draw.rect(self.screen, RED, [self.food_x, self.food_y, BLOCK_SIZE, BLOCK_SIZE])
        for segment in self.snake:
            pygame.draw.rect(self.screen, GREEN, [segment[0], segment[1], BLOCK_SIZE, BLOCK_SIZE])
        pygame.draw.rect(self.screen, WHITE, [self.snake[-1][0], self.snake[-1][1], BLOCK_SIZE, BLOCK_SIZE])

        # Texto da UI
        epoch_text = self.font.render(f"Época: {self.epoch}", True, WHITE)
        reward_text = self.font.render(f"Recompensa: {self.total_reward:.2f}", True, WHITE)

        self.screen.blit(epoch_text, (10, 10))
        self.screen.blit(reward_text, (10, 40))

        pygame.display.update()
        self.clock.tick(self.speed)