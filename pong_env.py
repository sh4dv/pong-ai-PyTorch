"""
Gymnasium wrapper for PongGame.
Wraps the existing PongGame class to work with gymnasium's vectorized environments.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pong_game import PongGame
from config import WINDOW_WIDTH, WINDOW_HEIGHT


class PongEnv(gym.Env):
    """
    Gymnasium environment wrapper for PongGame.
    
    State: [ball_x, ball_y, ball_vel_x, ball_vel_y, paddle1_y, paddle2_y]
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
        normalized = np.array([
            state[0] / WINDOW_WIDTH,      # ball_x
            state[1] / WINDOW_HEIGHT,     # ball_y
            (state[2] + 10) / 20,          # ball_vel_x (assumes range [-10, 10])
            (state[3] + 10) / 20,          # ball_vel_y (assumes range [-10, 10])
            state[4] / WINDOW_HEIGHT,     # paddle1_y
            state[5] / WINDOW_HEIGHT,     # paddle2_y
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
        normalized_state = self._normalize_state(state)
        
        info = {}
        
        return normalized_state, info
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action (int): Action for left paddle (0=none, 1=up, 2=down)
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        total_reward = 0.0
        done = False
        
        # Repeat action for frame_skip steps
        for _ in range(self.frame_skip):
            # Simple AI for right paddle
            state = self.game._get_state()
            action_right = self._simple_ai_action(state)
            
            # Execute step
            next_state, reward_left, reward_right, done = self.game.step(action, action_right)
            total_reward += reward_left
            
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
            'score_right': score_right
        }
        
        return normalized_state, total_reward, terminated, truncated, info
    
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
