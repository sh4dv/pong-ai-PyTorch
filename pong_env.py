"""
Gymnasium wrapper for PongGame.
Wraps the existing PongGame class to work with gymnasium's vectorized environments.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pong_game import PongGame
from config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    OPPONENT_SPEED_MIN,
    OPPONENT_SPEED_MAX,
    BALL_SIZE,
    PADDLE_OFFSET,
    PADDLE_WIDTH,
    PADDLE_HEIGHT,
    WINNING_SCORE,
    REWARD_NEUTRAL,
    REWARD_HIT_BALL,
    REWARD_SCORE_POINT,
    REWARD_LOSE_POINT,
    REWARD_PROXIMITY,
    REWARD_MISS_BALL,
    REWARD_FAR_PENALTY,
)


class PongEnv(gym.Env):
    """
    Gymnasium environment wrapper for PongGame.
    
    State: [ball_x, ball_y, ball_vel_x, ball_vel_y, paddle1_center_y, paddle2_center_y]
    Action: Discrete(3) - 0=none, 1=up, 2=down
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 60}
    
    def __init__(self, render_mode=None, frame_skip=1):
        """
        Initialize the Pong environment.
        
        Args:
            render_mode (str): Rendering mode (None or "human")
            frame_skip (int): Number of frames to repeat each action (1 = no skip)
        """
        super().__init__()
        
        self.game = PongGame()
        self.render_mode = render_mode
        self.frame_skip = max(1, frame_skip)  # Ensure at least 1
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0=none, 1=up, 2=down
        
        # Observation space: [ball_x, ball_y, ball_vel_x, ball_vel_y, paddle1_y, paddle2_y]
        # All values are normalized to [0, 1] range
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )
        
        # For rendering
        self.renderer = None
        if self.render_mode == "human":
            try:
                from renderer import PongRenderer
                self.renderer = PongRenderer()
            except ImportError:
                print("Warning: Renderer not available")
    
    def _normalize_state(self, state):
        """
        Normalize state to [0, 1] range.
        
        Args:
            state (np.array): Raw state from game
        
        Returns:
            np.array: Normalized state
        """
        paddle1_center = (state[4] + PADDLE_HEIGHT / 2) / WINDOW_HEIGHT
        paddle2_center = (state[5] + PADDLE_HEIGHT / 2) / WINDOW_HEIGHT

        normalized = np.array([
            state[0] / WINDOW_WIDTH,      # ball_x
            state[1] / WINDOW_HEIGHT,     # ball_y
            (state[2] + 10) / 20,         # ball_vel_x (assumes range [-10, 10])
            (state[3] + 10) / 20,         # ball_vel_y (assumes range [-10, 10])
            paddle1_center,               # paddle1 center_y
            paddle2_center,               # paddle2 center_y
        ], dtype=np.float32)
        
        return np.clip(normalized, 0.0, 1.0)
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment.
        
        Args:
            seed (int): Random seed
            options (dict): Additional options
        
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        state = self.game.reset()
        # Randomize opponent (right paddle) speed per episode
        opp_speed = np.random.uniform(OPPONENT_SPEED_MIN, OPPONENT_SPEED_MAX)
        self.game.set_paddle_speed(2, opp_speed)
        normalized_state = self._normalize_state(state)
        
        info = {}
        
        return normalized_state, info
    
    def step(self, action):
        """
        Execute one step in the environment and expose a reward breakdown.
        
        Args:
            action (int): Action for left paddle (0=none, 1=up, 2=down)
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        total_reward = 0.0
        done = False
        reward_breakdown = {
            'hit_ball': 0.0,
            'score_point': 0.0,
            'lose_point': 0.0,
            'miss_ball': 0.0,
            'proximity': 0.0,
            'far_penalty': 0.0,
            'neutral': 0.0
        }
        
        # Repeat action for frame_skip steps
        for _ in range(self.frame_skip):
            # Simple AI for right paddle
            state = self.game._get_state()
            action_right = self._simple_ai_action(state)
            
            # Execute step
            next_state, reward_left, reward_right, done, frame_breakdown = self._step_with_breakdown(action, action_right)
            total_reward += reward_left
            
            # Accumulate reward breakdown for this episode step
            for key, value in frame_breakdown.items():
                reward_breakdown[key] += value
            
            # Break if episode ended
            if done:
                break
        
        # Normalize state
        normalized_state = self._normalize_state(next_state)
        
        # Gymnasium uses terminated and truncated instead of done
        terminated = done
        truncated = False
        
        # Info
        score_left, score_right = self.game.get_scores()
        info = {
            'score_left': score_left,
            'score_right': score_right,
            'reward_breakdown': reward_breakdown
        }
        
        return normalized_state, total_reward, terminated, truncated, info

    def _step_with_breakdown(self, action_left, action_right):
        """Single frame step that also returns a reward breakdown for the left player."""
        # Mirror PongGame.step but track the source of each reward component
        if self.game.done:
            return self.game._get_state(), 0.0, 0.0, True, {
                'hit_ball': 0.0,
                'score_point': 0.0,
                'lose_point': 0.0,
                'miss_ball': 0.0,
                'proximity': 0.0,
                'far_penalty': 0.0,
                'neutral': 0.0
            }
        
        reward_left = REWARD_NEUTRAL
        reward_right = REWARD_NEUTRAL
        reward_contrib = {
            'hit_ball': 0.0,
            'score_point': 0.0,
            'lose_point': 0.0,
            'miss_ball': 0.0,
            'proximity': 0.0,
            'far_penalty': 0.0,
            'neutral': 0.0,
        }
        prev_ball_x = self.game.ball_x
        
        # Move paddles and ball using the underlying game logic
        self.game._move_paddle(1, action_left)
        self.game._move_paddle(2, action_right)
        self.game.ball_x += self.game.ball_vel_x
        self.game.ball_y += self.game.ball_vel_y
        
        # Ball collision with top and bottom walls
        if self.game.ball_y <= 0 or self.game.ball_y >= WINDOW_HEIGHT - BALL_SIZE:
            self.game.ball_vel_y *= -1
            self.game.ball_y = np.clip(self.game.ball_y, 0, WINDOW_HEIGHT - BALL_SIZE)
        
        # Ball collision with left paddle
        if (self.game.ball_x <= PADDLE_OFFSET + PADDLE_WIDTH and
            self.game.paddle1_y <= self.game.ball_y <= self.game.paddle1_y + PADDLE_HEIGHT):
            self.game._handle_paddle_hit(1)
            reward_left = REWARD_HIT_BALL
            reward_contrib['hit_ball'] += REWARD_HIT_BALL
        # Dense reward only when ball is coming toward the left paddle and still in front of it
        elif self.game.ball_vel_x < 0 and self.game.ball_x <= WINDOW_WIDTH * 0.75:
            paddle1_center = self.game.paddle1_y + PADDLE_HEIGHT / 2
            ball_center = self.game.ball_y + BALL_SIZE / 2
            distance = abs(paddle1_center - ball_center)

            # Close range grants reward, mid range decays to zero, far range gives penalty
            close_thresh = PADDLE_HEIGHT * 0.5
            far_thresh = PADDLE_HEIGHT * 1.2

            if distance <= close_thresh:
                # Stronger shaping near the ball: reward rises quadratically as distance -> 0
                tightness = 1 - distance / close_thresh
                proximity_reward = REWARD_PROXIMITY * (tightness * tightness)
                reward_left = proximity_reward
                reward_contrib['proximity'] += proximity_reward
            elif distance >= far_thresh:
                # Penalize when paddle stays far while the ball is coming in
                span = max(far_thresh, 1.0)
                scale = min(1.0, (distance - far_thresh) / span)
                far_penalty = REWARD_FAR_PENALTY * (1.0 + scale)
                reward_left = far_penalty
                reward_contrib['far_penalty'] += far_penalty
            else:
                span = far_thresh - close_thresh
                scale = (distance - close_thresh) / span
                proximity_reward = REWARD_PROXIMITY * max(0.0, (1 - scale) * (1 - scale))
                reward_left = proximity_reward
                reward_contrib['proximity'] += proximity_reward
        else:
            reward_contrib['neutral'] += REWARD_NEUTRAL
            reward_left = REWARD_NEUTRAL
        
        # Ball collision with right paddle
        if (self.game.ball_x >= WINDOW_WIDTH - PADDLE_OFFSET - PADDLE_WIDTH - BALL_SIZE and
            self.game.paddle2_y <= self.game.ball_y <= self.game.paddle2_y + PADDLE_HEIGHT):
            self.game._handle_paddle_hit(2)
            reward_right = REWARD_HIT_BALL
        
        # Check if ball went out of bounds (scoring)
        if self.game.ball_x < 0:
            # Right player scores - left paddle missed
            self.game.score2 += 1
            reward_contrib['lose_point'] += REWARD_LOSE_POINT
            reward_left = REWARD_LOSE_POINT
            reward_right = REWARD_SCORE_POINT
            # Additional penalty for missing when ball was in range
            if prev_ball_x > 0 and prev_ball_x < PADDLE_OFFSET + PADDLE_WIDTH + 50:
                reward_contrib['miss_ball'] += REWARD_MISS_BALL
                reward_left += REWARD_MISS_BALL
            self.game._reset_ball()
            
            if self.game.score2 >= WINNING_SCORE:
                self.game.done = True
        elif self.game.ball_x > WINDOW_WIDTH:
            # Left player scores
            self.game.score1 += 1
            reward_contrib['score_point'] += REWARD_SCORE_POINT
            reward_left = REWARD_SCORE_POINT
            reward_right = REWARD_LOSE_POINT
            self.game._reset_ball()
            
            if self.game.score1 >= WINNING_SCORE:
                self.game.done = True
        
        # The reward for the left player is the sum of the contributions recorded above
        reward_left_total = sum(reward_contrib.values()) if reward_left == REWARD_NEUTRAL else reward_left
        return self.game._get_state(), reward_left_total, reward_right, self.game.done, reward_contrib
    
    def _simple_ai_action(self, state):
        """
        Simple AI for opponent (right paddle).
        
        Args:
            state (np.array): Current game state
        
        Returns:
            int: Action for right paddle
        """
        from config import PADDLE_HEIGHT
        
        ball_y = state[1]
        paddle2_y = state[5]
        paddle_center = paddle2_y + PADDLE_HEIGHT / 2
        
        # Add some randomness - 50% chance of random action (makes opponent beatable)
        if np.random.random() < 0.5:
            return np.random.randint(0, 3)  # Random action: none/up/down
        
        # Move towards ball
        if ball_y < paddle_center - 10:
            return 1  # Move up
        elif ball_y > paddle_center + 10:
            return 2  # Move down
        else:
            return 0  # Stay
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human" and self.renderer is not None:
            self.renderer.render(self.game)
            self.renderer.tick()
    
    def close(self):
        """Close the environment."""
        if self.renderer is not None:
            self.renderer.close()
