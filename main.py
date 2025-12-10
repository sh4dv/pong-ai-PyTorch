"""
Main executable for playing Pong with keyboard controls.
Left player: W (up) and S (down)
Right player: Arrow keys (up and down)
"""

import pygame
from pong_game import PongGame
from renderer import PongRenderer


def main():
    """Main game loop with human controls."""
    # Initialize game and renderer
    game = PongGame()
    renderer = PongRenderer()
    
    running = True
    
    print("Pong Game Started!")
    print("Left Player: W (up), S (down)")
    print("Right Player: Arrow keys (up, down)")
    print(f"First to {game.score1} points wins!")
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get keyboard state
        keys = pygame.key.get_pressed()
        
        # Determine actions for both players
        action_left = 0  # No movement by default
        action_right = 0
        
        # Left player controls (W/S)
        if keys[pygame.K_w]:
            action_left = 1  # Move up
        elif keys[pygame.K_s]:
            action_left = 2  # Move down
        
        # Right player controls (Arrow keys)
        if keys[pygame.K_UP]:
            action_right = 1  # Move up
        elif keys[pygame.K_DOWN]:
            action_right = 2  # Move down
        
        # Handle restart and quit when game is over
        if game.is_done():
            if keys[pygame.K_r]:
                game.reset()
                print("\nGame restarted!")
            elif keys[pygame.K_q]:
                running = False
        
        # Execute game step
        state, reward_left, reward_right, done = game.step(action_left, action_right)
        
        # Render the game
        renderer.render(game)
        
        # Control frame rate
        renderer.tick()
    
    # Cleanup
    renderer.close()
    print("\nGame ended!")
    score1, score2 = game.get_scores()
    print(f"Final Score - Left Player: {score1}, Right Player: {score2}")


if __name__ == "__main__":
    main()
