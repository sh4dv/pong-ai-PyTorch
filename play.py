"""
Play Pong with a trained DQN agent.
Visualizes the trained agent playing against a simple AI opponent.
"""

import argparse
import numpy as np
import pygame
from pong_game import PongGame
from dqn_agent import DQNAgent
from renderer import PongRenderer
from config import *


class PongPlayer:
    """
    Player for watching a trained DQN agent play Pong.
    """
    
    def __init__(self, model_path, opponent="simple_ai"):
        """
        Initialize the player.
        
        Args:
            model_path (str): Path to trained model
            opponent (str): Type of opponent ("simple_ai" or "human")
        """
        self.game = PongGame()
        self.agent = DQNAgent()
        self.renderer = PongRenderer()
        self.opponent = opponent
        
        # Load trained model
        if not self.agent.load(model_path):
            raise FileNotFoundError(f"Could not load model from {model_path}")
        
        print(f"\nLoaded trained model from {model_path}")
        print(f"Opponent: {opponent}")
        print(f"Device: {self.agent.device}")
    
    def _normalize_state(self, state):
        """
        Normalize state to [0, 1] range (must match training normalization).
        
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
        
    def play(self, num_games=None):
        """
        Play games with the trained agent.
        
        Args:
            num_games (int): Number of games to play (None = infinite)
        """
        games_played = 0
        total_wins = 0
        total_losses = 0
        
        print("\n" + "=" * 60)
        print("Controls:")
        if self.opponent == "human":
            print("  Right Player (You): Arrow keys (Up/Down)")
        print("  Q: Quit")
        print("  R: Restart game")
        print("=" * 60 + "\n")
        
        running = True
        game_active = True
        
        while running:
            if game_active:
                # Get current state
                state = self.game._get_state()
                
                # Normalize state (CRITICAL: must match training normalization)
                normalized_state = self._normalize_state(state)
                
                # AI agent controls left paddle
                action_left = self.agent.select_action(normalized_state, training=False)
                
                # Right paddle action
                if self.opponent == "human":
                    action_right = self._get_human_action()
                else:
                    action_right = self._simple_ai_action(state)
                
                # Execute step
                next_state, reward_left, reward_right, done = self.game.step(action_left, action_right)
                
                # Check if game ended
                if done:
                    game_active = False
                    games_played += 1
                    
                    score_left, score_right = self.game.get_scores()
                    if score_left > score_right:
                        total_wins += 1
                        result = "AI WINS!"
                    else:
                        total_losses += 1
                        result = "AI LOSES!"
                    
                    print(f"\nGame {games_played} finished: {result}")
                    print(f"Score - AI: {score_left}, Opponent: {score_right}")
                    print(f"Total Record - Wins: {total_wins}, Losses: {total_losses}")
                    
                    # Check if we should continue
                    if num_games is not None and games_played >= num_games:
                        running = False
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Handle keyboard input
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                running = False
            elif keys[pygame.K_r] and not game_active:
                # Restart game
                self.game.reset()
                game_active = True
                print("\nNew game started!")
            
            # Render
            self.renderer.render(self.game)
            self.renderer.tick()
        
        # Cleanup
        self.renderer.close()
        
        print("\n" + "=" * 60)
        print("Play session ended")
        print(f"Games played: {games_played}")
        print(f"AI Record - Wins: {total_wins}, Losses: {total_losses}")
        if games_played > 0:
            win_rate = (total_wins / games_played) * 100
            print(f"Win rate: {win_rate:.1f}%")
        print("=" * 60)
    
    def _get_human_action(self):
        """
        Get action from human player using keyboard.
        
        Returns:
            int: Action (0=none, 1=up, 2=down)
        """
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            return 1
        elif keys[pygame.K_DOWN]:
            return 2
        return 0
    
    def _simple_ai_action(self, state):
        """
        Simple AI opponent that follows the ball.
        
        Args:
            state (np.array): Current game state
        
        Returns:
            int: Action (0=none, 1=up, 2=down)
        """
        ball_y = state[1]
        paddle2_y = state[5]
        paddle_center = paddle2_y + PADDLE_HEIGHT / 2
        
        # Move towards ball with some margin
        if ball_y < paddle_center - 15:
            return 1  # Move up
        elif ball_y > paddle_center + 15:
            return 2  # Move down
        return 0


def main():
    """Main entry point for playing with trained agent."""
    parser = argparse.ArgumentParser(description='Play Pong with trained DQN agent')
    parser.add_argument('--model', type=str, default=MODEL_SAVE_PATH,
                       help='Path to trained model')
    parser.add_argument('--opponent', type=str, default='simple_ai',
                       choices=['simple_ai', 'human'],
                       help='Type of opponent (simple_ai or human)')
    parser.add_argument('--games', type=int, default=None,
                       help='Number of games to play (default: infinite)')
    
    args = parser.parse_args()
    
    # Create player and start playing
    try:
        player = PongPlayer(args.model, args.opponent)
        player.play(args.games)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"Please train a model first using: python train.py")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")


if __name__ == "__main__":
    main()
