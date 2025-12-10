"""
Core Pong game logic - completely separated from rendering.
This class can be used for reinforcement learning without any graphics.
"""

import numpy as np
from config import *


class PongGame:
    """
    Pong game engine with logic completely separated from rendering.
    
    State representation:
        [ball_x, ball_y, ball_vel_x, ball_vel_y, paddle1_y, paddle2_y]
    
    Actions:
        0 = no movement
        1 = move up
        2 = move down
    """
    
    def __init__(self):
        """Initialize the game state."""
        self.reset()
    
    def reset(self):
        """
        Reset the game to initial state.
        
        Returns:
            state (np.array): Initial game state as numpy array
        """
        # Ball position (center of screen)
        self.ball_x = WINDOW_WIDTH // 2
        self.ball_y = WINDOW_HEIGHT // 2
        
        # Ball velocity (random direction)
        self.ball_vel_x = BALL_SPEED_X * np.random.choice([-1, 1])
        self.ball_vel_y = BALL_SPEED_Y * np.random.choice([-1, 1])
        
        # Paddle positions (centered vertically)
        self.paddle1_y = WINDOW_HEIGHT // 2 - PADDLE_HEIGHT // 2
        self.paddle2_y = WINDOW_HEIGHT // 2 - PADDLE_HEIGHT // 2
        
        # Scores
        self.score1 = 0
        self.score2 = 0
        
        # Game state
        self.done = False
        
        return self._get_state()
    
    def _get_state(self):
        """
        Get current game state as numpy array.
        
        Returns:
            np.array: [ball_x, ball_y, ball_vel_x, ball_vel_y, paddle1_y, paddle2_y]
        """
        return np.array([
            self.ball_x,
            self.ball_y,
            self.ball_vel_x,
            self.ball_vel_y,
            self.paddle1_y,
            self.paddle2_y
        ], dtype=np.float32)
    
    def step(self, action_left, action_right):
        """
        Execute one step of the game.
        
        Args:
            action_left (int): Action for left paddle (0=none, 1=up, 2=down)
            action_right (int): Action for right paddle (0=none, 1=up, 2=down)
        
        Returns:
            state (np.array): New game state
            reward_left (float): Reward for left player
            reward_right (float): Reward for right player
            done (bool): Whether game is finished
        """
        if self.done:
            return self._get_state(), 0, 0, True
        
        # Initialize rewards
        reward_left = REWARD_NEUTRAL
        reward_right = REWARD_NEUTRAL
        
        # Move paddles based on actions
        self._move_paddle(1, action_left)
        self._move_paddle(2, action_right)
        
        # Move ball
        self.ball_x += self.ball_vel_x
        self.ball_y += self.ball_vel_y
        
        # Ball collision with top and bottom walls
        if self.ball_y <= 0 or self.ball_y >= WINDOW_HEIGHT - BALL_SIZE:
            self.ball_vel_y *= -1
            self.ball_y = np.clip(self.ball_y, 0, WINDOW_HEIGHT - BALL_SIZE)
        
        # Ball collision with left paddle
        if (self.ball_x <= PADDLE_OFFSET + PADDLE_WIDTH and
            self.paddle1_y <= self.ball_y <= self.paddle1_y + PADDLE_HEIGHT):
            self._handle_paddle_hit(1)
            reward_left = REWARD_HIT_BALL
        
        # Ball collision with right paddle
        if (self.ball_x >= WINDOW_WIDTH - PADDLE_OFFSET - PADDLE_WIDTH - BALL_SIZE and
            self.paddle2_y <= self.ball_y <= self.paddle2_y + PADDLE_HEIGHT):
            self._handle_paddle_hit(2)
            reward_right = REWARD_HIT_BALL
        
        # Check if ball went out of bounds (scoring)
        if self.ball_x < 0:
            # Right player scores
            self.score2 += 1
            reward_right = REWARD_SCORE_POINT
            reward_left = REWARD_LOSE_POINT
            self._reset_ball()
            
            if self.score2 >= WINNING_SCORE:
                self.done = True
        
        elif self.ball_x > WINDOW_WIDTH:
            # Left player scores
            self.score1 += 1
            reward_left = REWARD_SCORE_POINT
            reward_right = REWARD_LOSE_POINT
            self._reset_ball()
            
            if self.score1 >= WINNING_SCORE:
                self.done = True
        
        return self._get_state(), reward_left, reward_right, self.done
    
    def _move_paddle(self, paddle_num, action):
        """
        Move paddle based on action.
        
        Args:
            paddle_num (int): Paddle number (1 or 2)
            action (int): Action (0=none, 1=up, 2=down)
        """
        if paddle_num == 1:
            if action == 1:  # Move up
                self.paddle1_y -= PADDLE_SPEED
            elif action == 2:  # Move down
                self.paddle1_y += PADDLE_SPEED
            
            # Keep paddle within bounds
            self.paddle1_y = np.clip(self.paddle1_y, 0, WINDOW_HEIGHT - PADDLE_HEIGHT)
        
        elif paddle_num == 2:
            if action == 1:  # Move up
                self.paddle2_y -= PADDLE_SPEED
            elif action == 2:  # Move down
                self.paddle2_y += PADDLE_SPEED
            
            # Keep paddle within bounds
            self.paddle2_y = np.clip(self.paddle2_y, 0, WINDOW_HEIGHT - PADDLE_HEIGHT)
    
    def _handle_paddle_hit(self, paddle_num):
        """
        Handle ball collision with paddle.
        
        Args:
            paddle_num (int): Paddle number that hit the ball (1 or 2)
        """
        # Reverse ball direction
        self.ball_vel_x *= -1
        
        # Add some variation based on where ball hits paddle
        if paddle_num == 1:
            paddle_center = self.paddle1_y + PADDLE_HEIGHT / 2
            self.ball_x = PADDLE_OFFSET + PADDLE_WIDTH
        else:
            paddle_center = self.paddle2_y + PADDLE_HEIGHT / 2
            self.ball_x = WINDOW_WIDTH - PADDLE_OFFSET - PADDLE_WIDTH - BALL_SIZE
        
        # Modify vertical velocity based on hit position
        hit_pos = (self.ball_y - paddle_center) / (PADDLE_HEIGHT / 2)
        self.ball_vel_y += hit_pos * 2
        
        # Limit maximum speed
        self.ball_vel_y = np.clip(self.ball_vel_y, -BALL_MAX_SPEED, BALL_MAX_SPEED)
        
        # Slightly increase horizontal speed
        if abs(self.ball_vel_x) < BALL_MAX_SPEED:
            self.ball_vel_x *= 1.05
    
    def _reset_ball(self):
        """Reset ball to center with random direction."""
        self.ball_x = WINDOW_WIDTH // 2
        self.ball_y = WINDOW_HEIGHT // 2
        self.ball_vel_x = BALL_SPEED_X * np.random.choice([-1, 1])
        self.ball_vel_y = BALL_SPEED_Y * np.random.choice([-1, 1])
    
    def get_scores(self):
        """
        Get current scores.
        
        Returns:
            tuple: (score_left, score_right)
        """
        return self.score1, self.score2
    
    def is_done(self):
        """
        Check if game is finished.
        
        Returns:
            bool: True if game is over
        """
        return self.done
