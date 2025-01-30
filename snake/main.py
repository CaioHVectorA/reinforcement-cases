import pygame
import random
from src.matrix import generate_game_matrix, print_matrix
from src.gen_food import generate_food
from src.consts import BLACK, BLOCK_SIZE, GREEN, HEIGHT, RED,WHITE, WIDTH
from typing import Tuple, List


# Initialize Pygame
pygame.init()




# Set up the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Snake Game')

# Speed and clock settings
speed = 15
clock = pygame.time.Clock()

# Font settings for the scoreboard
font = pygame.font.SysFont(None, 50)

def draw_snake(snake):
    """Draws the snake on the screen"""
    for block in snake:
        pygame.draw.rect(screen, GREEN, [block[0], block[1], BLOCK_SIZE, BLOCK_SIZE])

# def display_message(msg, color):
#     """Displays a message on the screen"""
#     text = font.render(msg, True, color)
#     screen.blit(text, [WIDTH/6, HEIGHT/3])

def game():
    """Main function that controls the game"""
    game_over = False
    game_ended = False

    # Initial position of the snake
    x = WIDTH // 2
    y = HEIGHT // 2

    # Initial movement
    dx = BLOCK_SIZE
    dy = 0

    snake = []
    snake_length = 3

    # Generate the first food
    food_x, food_y = generate_food(snake)

    while not game_over:
        # Loop for when the game ends
        while game_ended:
            screen.fill(BLACK)
            # display_message("Game Over! Press C to continue or Q to quit", RED)
            # draw_score(snake_length - 1)
            pygame.display.update()

            # Handle events after the game ends
            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         game_over = True
            #         game_ended = False
            #     if event.type == pygame.KEYDOWN:
            #         if event.key == pygame.K_q:
            #             game_over = True
            #             game_ended = False
            #         if event.key == pygame.K_c:
        game()

        # Handle events during the game
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         game_over = True
        #     if event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_LEFT and dx == 0:
        #             dx = -BLOCK_SIZE
        #             dy = 0
        #         elif event.key == pygame.K_RIGHT and dx == 0:
        #             dx = BLOCK_SIZE
        #             dy = 0
        #         elif event.key == pygame.K_UP and dy == 0:
        #             dy = -BLOCK_SIZE
        #             dx = 0
        #         elif event.key == pygame.K_DOWN and dy == 0:
        #             dy = BLOCK_SIZE
        #             dx = 0

        # Simulate AI playing the game
        
        actions = "left", "right", "up", "down"
        action = random.choice(actions)
        if action == "left" and dx == 0:
            dx = -BLOCK_SIZE
            dy = 0
        elif action == "right" and dx == 0:
            dx = BLOCK_SIZE
            dy = 0
        elif action == "up" and dy == 0:
            dy = -BLOCK_SIZE
            dx = 0
        elif action == "down" and dy == 0:
            dy = BLOCK_SIZE
            dx = 0

        # Update snake position
        x += dx
        y += dy

        # Check collision with walls
        if x >= WIDTH or x < 0 or y >= HEIGHT or y < 0:
            game_ended = True

        # Update the snake's body
        head = [x, y]
        snake.append(head)

        # Maintain the correct length of the snake
        if len(snake) > snake_length:
            del snake[0]

        # Check collision with itself
        for block in snake[:-1]:
            if block == head:
                game_ended = True
        # Draw elements on the screen
        screen.fill(BLACK)
        pygame.draw.rect(screen, RED, [food_x, food_y, BLOCK_SIZE, BLOCK_SIZE])
        draw_snake(snake)
        # draw_score(snake_length - 1)
        pygame.display.update()
        # Check if the snake ate the food
        if x == food_x and y == food_y:
            food_x, food_y = generate_food(snake)
            snake_length += 1

        # Control the speed
        clock.tick(speed)

    pygame.quit()
    quit()

# Start the game
game()