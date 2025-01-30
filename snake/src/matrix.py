from src.consts import BLACK, BLOCK_SIZE, GREEN, HEIGHT, RED,WHITE, WIDTH
from typing import Tuple, List

def generate_game_matrix(
    snake_segments: List[List[int]],  # List of [x,y] snake segments
    food_pos: Tuple[int, int],        # (food_x, food_y)
) -> List[List[int]]:
    """Generates a 40x30 game matrix with snake body, head, and food positions."""
    matrix = [[0 for _ in range(WIDTH // BLOCK_SIZE)] for _ in range(HEIGHT // BLOCK_SIZE)]
    
    # Add logging placeholder (you can expand this)
    # print(f"Generating matrix for snake length {len(snake_segments)}")
    
    # Mark snake body (all segments except head)
    for segment in snake_segments[:-1]:
        x, y = segment
        norm_x = x // BLOCK_SIZE
        norm_y = y // BLOCK_SIZE
        if 0 <= norm_x < 40 and 0 <= norm_y < 30:
            matrix[norm_y][norm_x] = 1
    
    # Mark snake head (last segment)
    head_x, head_y = snake_segments[-1]
    norm_head_x = head_x // BLOCK_SIZE
    norm_head_y = head_y // BLOCK_SIZE
    if 0 <= norm_head_x < 40 and 0 <= norm_head_y < 30:
        matrix[norm_head_y][norm_head_x] = 2
    
    # Mark food
    food_x, food_y = food_pos
    matrix[food_y // BLOCK_SIZE][food_x // BLOCK_SIZE] = 3
    return matrix
def print_matrix(matrix):
    for row in matrix:
        print(row)