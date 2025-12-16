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
        
        # Ball velocity (random direction with bounded angle)
        self.ball_vel_x, self.ball_vel_y = self._random_launch_velocity()
        
        # Paddle positions (centered vertically)
        self.paddle1_y = WINDOW_HEIGHT // 2 - PADDLE_HEIGHT // 2
        self.paddle2_y = WINDOW_HEIGHT // 2 - PADDLE_HEIGHT // 2

        # Paddle speeds (can be customized externally)
        self.paddle1_speed = PADDLE_SPEED
        self.paddle2_speed = PADDLE_SPEED
        
        # Scores
        self.score1 = 0
        self.score2 = 0
        
        # Game state
        self.done = False
        # Track recent hits per rally to prevent rewarding lucky early scores
        self.left_hits_recent = 0
        self.right_hits_recent = 0
        
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
        
        # Store previous ball position for miss detection
        prev_ball_x = self.ball_x
        
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
        paddle_hit_left = False
        if (self.ball_x <= PADDLE_OFFSET + PADDLE_WIDTH and
            self.paddle1_y <= self.ball_y <= self.paddle1_y + PADDLE_HEIGHT):
            self._handle_paddle_hit(1)
            # Add hit reward (do not replace other contributions)
            reward_left += REWARD_HIT_BALL
            paddle_hit_left = True
        # Dense reward: proximity to ball for left paddle (only when ball approaching and not hit)
        elif self.ball_x < WINDOW_WIDTH / 2 and self.ball_vel_x < 0:
            paddle1_center = self.paddle1_y + PADDLE_HEIGHT / 2
            ball_center = self.ball_y + BALL_SIZE / 2
            distance = abs(paddle1_center - ball_center)
            max_distance = WINDOW_HEIGHT
            # Normalize distance and apply small reward
            proximity_reward = REWARD_PROXIMITY * max(0, 1 - distance / max_distance)
            # Add proximity shaping instead of replacing
            reward_left += proximity_reward
        
        # Ball collision with right paddle
        if (self.ball_x >= WINDOW_WIDTH - PADDLE_OFFSET - PADDLE_WIDTH - BALL_SIZE and
            self.paddle2_y <= self.ball_y <= self.paddle2_y + PADDLE_HEIGHT):
            self._handle_paddle_hit(2)
            reward_right += REWARD_HIT_BALL
        
        # Check if ball went out of bounds (scoring)
        if self.ball_x < 0:
            # Right player scores - left paddle missed
            self.score2 += 1
            # Award score to right only if it participated in the rally
            if getattr(self, 'right_hits_recent', 0) > 0:
                reward_right += REWARD_SCORE_POINT
            reward_left += REWARD_LOSE_POINT
            # Additional penalty for missing when ball was in range
            if prev_ball_x > 0 and prev_ball_x < PADDLE_OFFSET + PADDLE_WIDTH + 50:
                reward_left += REWARD_MISS_BALL
            self._reset_ball()
            
            if self.score2 >= WINNING_SCORE:
                self.done = True
        
        elif self.ball_x > WINDOW_WIDTH:
            # Left player scores
            self.score1 += 1
            # Award score to left only if it participated in the rally
            if getattr(self, 'left_hits_recent', 0) > 0:
                reward_left += REWARD_SCORE_POINT
            reward_right += REWARD_LOSE_POINT
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
                self.paddle1_y -= self.paddle1_speed
            elif action == 2:  # Move down
                self.paddle1_y += self.paddle1_speed
            
            # Keep paddle within bounds
            self.paddle1_y = np.clip(self.paddle1_y, 0, WINDOW_HEIGHT - PADDLE_HEIGHT)
        
        elif paddle_num == 2:
            if action == 1:  # Move up
                self.paddle2_y -= self.paddle2_speed
            elif action == 2:  # Move down
                self.paddle2_y += self.paddle2_speed
            
            # Keep paddle within bounds
            self.paddle2_y = np.clip(self.paddle2_y, 0, WINDOW_HEIGHT - PADDLE_HEIGHT)

    def set_paddle_speed(self, paddle_num, speed):
        """Set paddle speed for a given paddle."""
        if paddle_num == 1:
            self.paddle1_speed = speed
        elif paddle_num == 2:
            self.paddle2_speed = speed
    
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
        # Track that this paddle participated in the current rally
        if paddle_num == 1:
            try:
                self.left_hits_recent += 1
            except AttributeError:
                self.left_hits_recent = 1
        else:
            try:
                self.right_hits_recent += 1
            except AttributeError:
                self.right_hits_recent = 1
    
    def _reset_ball(self):
        """Reset ball to center with random direction."""
        self.ball_x = WINDOW_WIDTH // 2
        self.ball_y = WINDOW_HEIGHT // 2
        self.ball_vel_x, self.ball_vel_y = self._random_launch_velocity()
        # Reset per-rally participation counters
        self.left_hits_recent = 0
        self.right_hits_recent = 0

    def _random_launch_velocity(self):
        angle = np.deg2rad(np.random.uniform(LAUNCH_ANGLE_MIN_DEG, LAUNCH_ANGLE_MAX_DEG))
        horiz_sign = np.random.choice([-1, 1])
        vert_sign = np.random.choice([-1, 1])
        vx = horiz_sign * BALL_SPEED_X * np.cos(angle)
        vy = vert_sign * BALL_SPEED_Y * np.sin(angle)
        return vx, vy
    
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
