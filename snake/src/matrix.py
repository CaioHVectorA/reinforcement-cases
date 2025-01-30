from src.consts import BLACK, BLOCK_SIZE, GREEN, HEIGHT, RED,WHITE, WIDTH
from typing import Tuple, List

def generate_game_matrix(
        snakePos: Tuple[int, int], # 0 to WIDTH, 0 to HEIGHT
        snakeDir: Tuple[int, int], # -10, 0, 10
        snakeLength: int,
        foodPos: Tuple[int, int],
):
    """Generates a matrix to represent the game"""
    # Based on WIDTH 800, HEIGHT 600, BLOCK_SIZE 20
    # The matrix will be 40x30
    # If the snake pos is in (12,10) and moving in (10,0), and the legnth is 3
    # The snake will be in (12,10), (13,10), (14,10)
    # The matrix will be:
    matrix = [[0 for _ in range(WIDTH // BLOCK_SIZE)] for _ in range(HEIGHT // BLOCK_SIZE)]
    x, y = snakePos
    dx, dy = snakeDir
    food_x, food_y = foodPos
    if x < 0: x = 0
    if y < 0: y = 0
    if x >= WIDTH: x = WIDTH - BLOCK_SIZE
    if y >= HEIGHT: y = HEIGHT - BLOCK_SIZE
    for i in range(snakeLength):
        matrix[y // BLOCK_SIZE][x // BLOCK_SIZE] = 1 # Body
        x += dx*-1
        y += dy*-1
    matrix[x // BLOCK_SIZE][y // BLOCK_SIZE] = 2 # Head
    matrix[food_y // BLOCK_SIZE][food_x // BLOCK_SIZE] = 3 # Food
    return matrix
    
def print_matrix(matrix):
    for row in matrix:
        print(row)