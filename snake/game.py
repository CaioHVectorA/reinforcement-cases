import numpy as np
import pygame
import random
from src.gen_food import generate_food
from src.consts import BLACK, BLOCK_SIZE, GREEN, RED, WHITE, WIDTH, HEIGHT

class SnakeEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Snake Game - RL')

        self.clock = pygame.time.Clock()
        self.speed = 15

        self.actions = ["left", "right", "up", "down"]  # Ações possíveis
        self.reset()  # Inicia o jogo

    def reset(self):
        """Reinicia o jogo e retorna o estado inicial."""
        self.x, self.y = WIDTH // 2, HEIGHT // 2
        self.dx, self.dy = BLOCK_SIZE, 0
        self.snake = [[self.x, self.y]]
        self.snake_length = 3
        self.food_x, self.food_y = generate_food(self.snake)
        self.done = False
        return self.get_state()  # Retorna o estado inicial

    def step(self, action):
        """Executa uma ação e retorna (novo estado, recompensa, done)."""
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

        # Recompensas
        reward = -0.1  # Penalização pequena por cada movimento

        # Se bateu na parede
        if self.x >= WIDTH or self.x < 0 or self.y >= HEIGHT or self.y < 0:
            self.done = True
            reward = -10

        # Se bateu no próprio corpo
        for segment in self.snake[:-1]:
            if segment == [self.x, self.y]:
                self.done = True
                reward = -10

        # Atualiza a cobra
        self.snake.append([self.x, self.y])
        if len(self.snake) > self.snake_length:
            del self.snake[0]

        # Se comeu a comida
        if self.x == self.food_x and self.y == self.food_y:
            self.food_x, self.food_y = generate_food(self.snake)
            self.snake_length += 1
            reward = 10

        return self.get_state(), reward, self.done

    def get_state(self):
        """Retorna o estado atual como uma matriz 40x30."""
        state = np.zeros((40, 30))
        def normalize(num, _min):
            return min(max(num // BLOCK_SIZE, 0), _min)
        for segment in self.snake:
            state[normalize(segment[0], 39), normalize(segment[1], 29)] = 1
        state[normalize(self.x, 39), normalize(self.y, 29)] = 2  # Cabeça
        state[normalize(self.food_x, 39), normalize(self.food_y, 29)] = 3  # Comida
        return state
    def render(self):
        """Desenha o jogo na tela."""
        self.screen.fill(BLACK)
        pygame.draw.rect(self.screen, RED, [self.food_x, self.food_y, BLOCK_SIZE, BLOCK_SIZE])
        for segment in self.snake:
            pygame.draw.rect(self.screen, GREEN, [segment[0], segment[1], BLOCK_SIZE, BLOCK_SIZE])
        pygame.draw.rect(self.screen, WHITE, [self.snake[-1][0], self.snake[-1][1], BLOCK_SIZE, BLOCK_SIZE])
        pygame.display.update()
        self.clock.tick(self.speed)
