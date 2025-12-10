"""
Optional rendering module for Pong game using PyGame.
This module is completely separate from game logic and only handles visualization.
"""

import pygame
from config import *


class PongRenderer:
    """
    Renderer for Pong game using PyGame.
    Handles all graphical output independently from game logic.
    """
    
    def __init__(self, width=WINDOW_WIDTH, height=WINDOW_HEIGHT):
        """
        Initialize the renderer.
        
        Args:
            width (int): Window width
            height (int): Window height
        """
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Pong Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 74)
        self.small_font = pygame.font.Font(None, 36)
    
    def render(self, game):
        """
        Render the current game state.
        
        Args:
            game (PongGame): The game instance to render
        """
        # Clear screen
        self.screen.fill(BLACK)
        
        # Draw center line
        pygame.draw.aaline(
            self.screen,
            GRAY,
            (self.width // 2, 0),
            (self.width // 2, self.height)
        )
        
        # Draw paddles
        self._draw_paddle(PADDLE_OFFSET, game.paddle1_y)
        self._draw_paddle(
            self.width - PADDLE_OFFSET - PADDLE_WIDTH,
            game.paddle2_y
        )
        
        # Draw ball
        pygame.draw.rect(
            self.screen,
            WHITE,
            (game.ball_x, game.ball_y, BALL_SIZE, BALL_SIZE)
        )
        
        # Draw scores
        self._draw_scores(game.score1, game.score2)
        
        # Draw game over message if game is done
        if game.done:
            self._draw_game_over(game.score1, game.score2)
        
        # Update display
        pygame.display.flip()
    
    def _draw_paddle(self, x, y):
        """
        Draw a paddle.
        
        Args:
            x (int): X position
            y (int): Y position
        """
        pygame.draw.rect(
            self.screen,
            WHITE,
            (x, y, PADDLE_WIDTH, PADDLE_HEIGHT)
        )
    
    def _draw_scores(self, score1, score2):
        """
        Draw the current scores.
        
        Args:
            score1 (int): Left player score
            score2 (int): Right player score
        """
        # Left score
        text1 = self.font.render(str(score1), True, WHITE)
        self.screen.blit(text1, (self.width // 4 - text1.get_width() // 2, 20))
        
        # Right score
        text2 = self.font.render(str(score2), True, WHITE)
        self.screen.blit(text2, (3 * self.width // 4 - text2.get_width() // 2, 20))
    
    def _draw_game_over(self, score1, score2):
        """
        Draw game over message.
        
        Args:
            score1 (int): Left player score
            score2 (int): Right player score
        """
        winner = "Left Player Wins!" if score1 > score2 else "Right Player Wins!"
        
        # Semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Winner text
        text = self.font.render(winner, True, WHITE)
        text_rect = text.get_rect(center=(self.width // 2, self.height // 2 - 50))
        self.screen.blit(text, text_rect)
        
        # Instruction text
        instruction = self.small_font.render("Press R to restart or Q to quit", True, WHITE)
        instruction_rect = instruction.get_rect(center=(self.width // 2, self.height // 2 + 50))
        self.screen.blit(instruction, instruction_rect)
    
    def tick(self, fps=FPS):
        """
        Control frame rate.
        
        Args:
            fps (int): Target frames per second
        """
        self.clock.tick(fps)
    
    def close(self):
        """Close the renderer and quit pygame."""
        pygame.quit()
