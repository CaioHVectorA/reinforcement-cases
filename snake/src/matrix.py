from src.consts import BLACK, BLOCK_SIZE, GREEN, HEIGHT, RED,WHITE, WIDTH, maxHeight, maxWidth
import os
import json
from typing import Tuple, List
# index = 0
matrixes = []
def generate_game_matrix(
    snake_segments: List[List[int]],  # List of [x,y] snake segments
    food_pos: Tuple[int, int],        # (food_x, food_y)
) -> List[List[int]]:
    """Generates a 40x30 game matrix with snake body, border head, and food positions."""

    global index
    matrix = [[0 for _ in range(WIDTH // BLOCK_SIZE)] for _ in range(HEIGHT // BLOCK_SIZE)]
    # Add border(4)
    for i in range(maxWidth):
        matrix[0][i] = 4
        matrix[maxHeight][i] = 4
    for i in range(maxHeight):
        matrix[i][0] = 4
        matrix[i][maxWidth] = 4
    for segment in snake_segments[:-1]:
        x, y = segment
        norm_x = x // BLOCK_SIZE
        norm_y = y // BLOCK_SIZE
        if 0 <= norm_x < maxWidth+1 and 0 <= norm_y < maxHeight+1:
            matrix[norm_y][norm_x] = 1
    
    # Mark snake head (last segment)
    head_x, head_y = snake_segments[-1]
    norm_head_x = head_x // BLOCK_SIZE
    norm_head_y = head_y // BLOCK_SIZE
    if 0 <= norm_head_x < maxWidth+1 and 0 <= norm_head_y < maxHeight+1:
        matrix[norm_head_y][norm_head_x] = 2
    # Mark food
    food_x, food_y = food_pos
    mtxStr = print_matrix(matrix)
    if mtxStr != "":
        # index += 1
        os.makedirs(f'./snake/out/', exist_ok=True)
        matrixes.append(matrix)
        if len(matrixes) == 100:
            open(f'./snake/out/matrix.json', 'w').write(json.dumps(matrixes))
    matrix[food_y // BLOCK_SIZE][food_x // BLOCK_SIZE] = 3
    
    return matrix
def print_matrix(matrix):
    res = ""
    for row in matrix:
        res += " ".join(str(cell) for cell in row) + "\n"
        # print(row)
    return res