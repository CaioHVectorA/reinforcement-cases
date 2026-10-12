from src.consts import BLACK, BLOCK_SIZE, GREEN, HEIGHT, RED,WHITE, WIDTH
import random
def generate_food(snake):
    """Generates food at a random position that does not coincide with the snake's body"""
    while True:
        x = random.randint(0, (WIDTH // BLOCK_SIZE) - 1) * BLOCK_SIZE
        y = random.randint(0, (HEIGHT // BLOCK_SIZE) - 1) * BLOCK_SIZE
        # Not in snake body or not in borders
        if [x, y] not in snake and x != 0 and y != 0 and x != WIDTH - BLOCK_SIZE and y != HEIGHT - BLOCK_SIZE:
            print(f"Food generated at {x, y}")
            return x, y