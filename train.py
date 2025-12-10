"""
Training script for DQN Pong AI agent.
Supports headless training, periodic rendering, checkpointing, and vectorized environments.
"""

import argparse
import os
import time
import gc
import numpy as np
import torch
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from pong_env import PongEnv
from dqn_agent import DQNAgent
from config import *

# Optional rendering
try:
    from renderer import PongRenderer
    RENDERING_AVAILABLE = True
except ImportError:
    RENDERING_AVAILABLE = False
    print("Warning: Rendering not available")


class PongTrainer:
    """
    Trainer for DQN agent playing Pong with vectorized environments.
    
    Manages training loop, statistics, and optional rendering.
    Supports multiple parallel game instances for faster training.
    """
    
    def __init__(self, num_episodes=NUM_EPISODES, render_every=RENDER_EVERY,
                 save_every=SAVE_EVERY, log_every=LOG_EVERY, headless=True,
                 num_envs=1, async_envs=False, frame_skip=FRAME_SKIP):
        """
        Initialize the trainer.
        
        Args:
            num_episodes (int): Total number of episodes to train
            render_every (int): Render every N episodes (0 = never)
            save_every (int): Save checkpoint every N episodes
            log_every (int): Print stats every N episodes
            headless (bool): Run without rendering even if available
            num_envs (int): Number of parallel environments (1 = no vectorization)
            async_envs (bool): Use async vectorization (faster for many envs)
            frame_skip (int): Number of frames to repeat each action (1 = no skip)
        """
        self.num_episodes = num_episodes
        self.render_every = render_every
        self.save_every = save_every
        self.log_every = log_every
        self.headless = headless or not RENDERING_AVAILABLE
        self.num_envs = num_envs
        self.frame_skip = frame_skip
        
        # Create vectorized or single environment
        if num_envs > 1:
            # For many environments (>100), AsyncVectorEnv is faster
            # Each process runs independently, better CPU core utilization
            if async_envs:
                print(f"Creating {num_envs} parallel environments (async mode - separate processes)")
                env_fns = [lambda fs=frame_skip: PongEnv(frame_skip=fs) for _ in range(num_envs)]
                self.env = AsyncVectorEnv(env_fns)
            else:
                print(f"Creating {num_envs} parallel environments (sync mode - shared memory)")
                env_fns = [lambda fs=frame_skip: PongEnv(frame_skip=fs) for _ in range(num_envs)]
                self.env = SyncVectorEnv(env_fns)
            self.vectorized = True
            print(f"Frame skip enabled: {frame_skip}x faster episodes")
        else:
            self.env = PongEnv(frame_skip=frame_skip)
            self.vectorized = False
            if frame_skip > 1:
                print(f"Frame skip enabled: {frame_skip}x faster episodes")
        
        # Initialize agent
        self.agent = DQNAgent()
        
        # Adaptive batch size for vectorized training
        # More parallel data = larger batch size for better gradient estimates
        self.batch_size = BATCH_SIZE if num_envs <= 8 else min(BATCH_SIZE * 2, 256)
        if self.batch_size > BATCH_SIZE:
            print(f"Using larger batch size ({self.batch_size}) for vectorized training")
        
        # Initialize renderer if needed (only works with single env)
        self.renderer = None
        if not self.headless and RENDERING_AVAILABLE and not self.vectorized:
            self.renderer = PongRenderer()
            print("Rendering enabled")
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.scores_left = []
        self.scores_right = []
    
    def train_episode(self, episode_num):
        """
        Train for one episode (works with both single and vectorized envs).
        
        Args:
            episode_num (int): Current episode number
        
        Returns:
            dict: Episode statistics
        """
        if self.vectorized:
            return self._train_episode_vectorized(episode_num)
        else:
            return self._train_episode_single(episode_num)
    
    def _train_episode_single(self, episode_num):
        """
        Train single environment for one episode.
        
        Args:
            episode_num (int): Current episode number
        
        Returns:
            dict: Episode statistics
        """
        state, _ = self.env.reset()
        total_reward = 0
        episode_length = 0
        episode_losses = []
        
        # Determine if we should render this episode
        should_render = (not self.headless and self.renderer is not None and 
                        self.render_every > 0 and episode_num % self.render_every == 0)
        
        terminated = False
        truncated = False
        action_counts = {0: 0, 1: 0, 2: 0}  # Track action distribution
        
        while not (terminated or truncated):
            # Select action for left paddle (AI agent)
            action = self.agent.select_action(state, training=True)
            action_counts[action] += 1
            
            # Execute step in environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # Store transition in replay buffer
            self.agent.store_transition(state, action, reward, next_state, terminated or truncated)
            
            # Train the agent
            loss = self.agent.train(self.batch_size)
            if loss is not None:
                episode_losses.append(loss)
            
            # Update state and statistics
            state = next_state
            total_reward += reward
            episode_length += 1
            
            # Render if needed
            if should_render:
                self.env.render()
        
        # Decay epsilon after each episode
        self.agent.decay_epsilon()
        
        # Calculate average loss for episode
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        
        # Calculate action distribution
        total_actions = sum(action_counts.values())
        action_dist = {k: (v/total_actions*100 if total_actions > 0 else 0) for k, v in action_counts.items()}
        
        return {
            'reward': total_reward,
            'length': episode_length,
            'loss': avg_loss,
            'score_left': info.get('score_left', 0),
            'score_right': info.get('score_right', 0),
            'action_dist': action_dist
        }
    
    def _train_episode_vectorized(self, episode_num):
        """
        Train vectorized environments for one episode.
        Runs all environments in parallel until all complete.
        
        Args:
            episode_num (int): Current episode number
        
        Returns:
            dict: Aggregated episode statistics
        """
        states, _ = self.env.reset()
        
        total_rewards = np.zeros(self.num_envs)
        episode_lengths = np.zeros(self.num_envs)
        all_losses = []
        final_scores_left = []
        final_scores_right = []
        
        # Track which environments are still running
        active_envs = np.ones(self.num_envs, dtype=bool)
        training_step = 0
        completed_envs = 0
        
        # Print progress for first iteration
        if episode_num == 0:
            print(f"Iteration {episode_num + 1}: Starting {self.num_envs} parallel environments...")
        
        while active_envs.any():
            # Select actions for all active environments (batched for GPU efficiency)
            if active_envs.all():
                # All envs active - use efficient batch processing
                actions = self.agent.select_actions_batch(states, training=True)
            else:
                # Some envs done - use individual selection
                actions = np.array([
                    self.agent.select_action(states[i], training=True) if active_envs[i] else 0
                    for i in range(self.num_envs)
                ])
            
            # Execute step in all environments
            next_states, rewards, terminateds, truncateds, infos = self.env.step(actions)
            
            # Process each environment
            for i in range(self.num_envs):
                if active_envs[i]:
                    # Store transition
                    done = terminateds[i] or truncateds[i]
                    self.agent.store_transition(states[i], actions[i], rewards[i], next_states[i], done)
                    
                    # Update statistics
                    total_rewards[i] += rewards[i]
                    episode_lengths[i] += 1
                    
                    # Check if episode ended
                    if done:
                        active_envs[i] = False
                        completed_envs += 1
                        # Print progress for first iteration
                        if episode_num == 0 and completed_envs in [1, self.num_envs // 4, self.num_envs // 2, self.num_envs]:
                            print(f"  → {completed_envs}/{self.num_envs} environments completed...")
                        # Decay epsilon for each completed episode
                        self.agent.decay_epsilon()
                        # Get final scores from info
                        if 'final_info' in infos and infos['final_info'][i] is not None:
                            final_scores_left.append(infos['final_info'][i].get('score_left', 0))
                            final_scores_right.append(infos['final_info'][i].get('score_right', 0))
                        elif isinstance(infos, dict) and 'score_left' in infos:
                            final_scores_left.append(infos['score_left'][i])
                            final_scores_right.append(infos['score_right'][i])
            
            # Train the agent with adaptive frequency based on number of environments
            # More envs = more data per step, so we can train less frequently
            # This reduces memory usage while maintaining training quality
            training_step += 1
            # For many envs, train less frequently but accumulate more data
            train_freq = max(1, self.num_envs // 32)  # 160 envs: every 5 steps = 800 transitions before training
            if training_step % train_freq == 0:
                loss = self.agent.train(self.batch_size)
                if loss is not None:
                    all_losses.append(loss)
            
            # Update states
            states = next_states
            
            # Periodic memory cleanup for large vectorized training
            if training_step % 100 == 0 and self.num_envs >= 16:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        
        # Note: Epsilon decay is now done per completed environment (see above)
        
        # Calculate average statistics
        avg_loss = np.mean(all_losses) if all_losses else 0
        avg_score_left = np.mean(final_scores_left) if final_scores_left else 0
        avg_score_right = np.mean(final_scores_right) if final_scores_right else 0
        
        return {
            'reward': np.mean(total_rewards),
            'length': np.mean(episode_lengths),
            'loss': avg_loss,
            'score_left': avg_score_left,
            'score_right': avg_score_right
        }
    
    def _simple_ai_action(self, state):
        """
        Simple AI for opponent (right paddle).
        NOTE: Not used anymore - handled in PongEnv
        """
        pass
    
    def train(self, resume_from=None):
        """
        Run the training loop.
        
        Args:
            resume_from (str): Path to checkpoint to resume from (optional)
        """
        start_episode = 0
        
        # Resume from checkpoint if specified
        if resume_from and os.path.exists(resume_from):
            self.agent.load(resume_from)
            print(f"Resumed training from {resume_from}")
        
        print(f"\nStarting training for {self.num_episodes} episodes")
        print(f"Headless mode: {self.headless}")
        print(f"Parallel environments: {self.num_envs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Device: {self.agent.device}")
        
        # GPU diagnostics
        if torch.backends.mps.is_available() or torch.cuda.is_available():
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated() if torch.cuda.is_available() else 'N/A (MPS)'}")
            print("⚠️  For optimal GPU usage with vectorized training:")
            print("   - Use --num-envs that's a multiple of 16 (e.g., 32, 64, 96)")
            print("   - Don't use --async-envs (incompatible with GPU)")
            print("   - Ensure batch operations run on GPU (check activity monitor)")
        
        print("-" * 60)
        
        start_time = time.time()
        
        # Adjust episode count for vectorized training
        # Each "episode" now represents num_envs actual episodes
        episodes_to_run = self.num_episodes if not self.vectorized else max(1, self.num_episodes // self.num_envs)
        actual_episodes = episodes_to_run * (self.num_envs if self.vectorized else 1)
        
        if self.vectorized:
            print(f"Running {episodes_to_run} iterations × {self.num_envs} envs = {actual_episodes} total episodes")
            print(f"Each iteration runs all {self.num_envs} environments in parallel")
            print("-" * 60)
        
        for episode in range(start_episode, episodes_to_run):
            # Train one episode (or batch of episodes for vectorized)
            stats = self.train_episode(episode)
            
            # Store statistics
            self.episode_rewards.append(stats['reward'])
            self.episode_lengths.append(stats['length'])
            self.losses.append(stats['loss'])
            self.scores_left.append(stats['score_left'])
            self.scores_right.append(stats['score_right'])
            
            # Store action distribution for logging
            if 'action_dist' in stats:
                self.last_action_dist = stats['action_dist']
            
            # Update target network periodically
            if episode % TARGET_UPDATE == 0:
                self.agent.update_target_network()
            
            # Save checkpoint periodically
            if self.save_every > 0 and (episode + 1) % self.save_every == 0:
                checkpoint_path = f"{CHECKPOINT_DIR}checkpoint_ep{episode+1}.pth"
                self.agent.save(checkpoint_path)
            
            # Log statistics periodically
            if self.log_every > 0 and (episode + 1) % self.log_every == 0:
                self._log_stats(episode + 1, start_time, episodes_to_run)
            
            # Log detailed action distribution every DEBUG_ACTIONS_EVERY episodes
            if hasattr(self, 'last_action_dist') and (episode + 1) % DEBUG_ACTIONS_EVERY == 0:
                dist = self.last_action_dist
                print(f"  └─> AI Actions: None={dist.get(0, 0):.1f}% | Up={dist.get(1, 0):.1f}% | Down={dist.get(2, 0):.1f}%")
        
        # Save final model
        self.agent.save(MODEL_SAVE_PATH)
        
        # Print final statistics
        print("\n" + "=" * 60)
        print("Training completed!")
        self._log_stats(episodes_to_run, start_time, episodes_to_run)
        print("=" * 60)
        
        # Cleanup memory before closing environments
        print("\nCleaning up memory...")
        self._cleanup_memory()
        
        # Close environments properly
        print("Closing environments...")
        try:
            self.env.close()
            # For AsyncVectorEnv, wait for worker processes to terminate
            if self.vectorized and hasattr(self.env, 'close_extras'):
                self.env.close_extras()
        except Exception as e:
            print(f"Warning during env.close(): {e}")
        
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception as e:
                print(f"Warning during renderer.close(): {e}")
        
        print("Cleanup completed!")
    
    def _log_stats(self, episode, start_time, total_episodes):
        """
        Print training statistics.
        
        Args:
            episode (int): Current episode number
            start_time (float): Training start time
            total_episodes (int): Total episodes to train
        """
        # Calculate statistics for last N episodes
        n = min(self.log_every, len(self.episode_rewards))
        
        avg_reward = np.mean(self.episode_rewards[-n:])
        avg_length = np.mean(self.episode_lengths[-n:])
        avg_loss = np.mean([l for l in self.losses[-n:] if l > 0]) if any(self.losses[-n:]) else 0
        avg_score_left = np.mean(self.scores_left[-n:])
        avg_score_right = np.mean(self.scores_right[-n:])
        
        elapsed_time = time.time() - start_time
        eps_per_sec = episode / elapsed_time if elapsed_time > 0 else 0
        
        # Adjust for vectorized environments
        actual_eps_per_sec = eps_per_sec * self.num_envs if self.vectorized else eps_per_sec
        
        # Memory usage info (optional)
        mem_info = ""
        try:
            import psutil
            process = psutil.Process()
            mem_mb = process.memory_info().rss / 1024 / 1024
            mem_info = f" | RAM: {mem_mb:.0f}MB"
        except ImportError:
            pass
        
        print(f"Episode {episode}/{total_episodes} | "
              f"Reward: {avg_reward:.2f} | "
              f"Length: {avg_length:.0f} | "
              f"Loss: {avg_loss:.4f} | "
              f"Score L/R: {avg_score_left:.1f}/{avg_score_right:.1f} | "
              f"Epsilon: {self.agent.get_epsilon():.3f} | "
              f"Speed: {actual_eps_per_sec:.2f} eps/s{mem_info} | "
              f"Buffer: {len(self.agent.memory)}")
    
    def _cleanup_memory(self):
        """
        Cleanup memory after training to prevent memory leaks.
        Important for vectorized environments (especially AsyncVectorEnv).
        """
        # Clear replay buffer
        if hasattr(self.agent, 'memory'):
            self.agent.memory.clear()
        
        # Clear training statistics
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.losses.clear()
        self.scores_left.clear()
        self.scores_right.clear()
        
        # Clear PyTorch cache
        if hasattr(self.agent, 'device'):
            if self.agent.device.type == 'cuda':
                torch.cuda.empty_cache()
            elif self.agent.device.type == 'mps':
                torch.mps.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Wait a moment for async processes to terminate
        if self.vectorized and hasattr(self.env, 'close'):
            time.sleep(0.5)


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description='Train DQN agent for Pong with vectorized environments')
    parser.add_argument('--episodes', type=int, default=NUM_EPISODES,
                       help='Number of episodes to train')
    parser.add_argument('--num-envs', type=int, default=1,
                       help='Number of parallel environments (1 = no vectorization, 4-8 recommended)')
    parser.add_argument('--async-envs', action='store_true',
                       help='Use async vectorization (faster for many envs)')
    parser.add_argument('--render-every', type=int, default=RENDER_EVERY,
                       help='Render every N episodes (0 = never, only works with 1 env)')
    parser.add_argument('--save-every', type=int, default=SAVE_EVERY,
                       help='Save checkpoint every N episodes')
    parser.add_argument('--log-every', type=int, default=LOG_EVERY,
                       help='Print stats every N episodes')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode (no rendering)')
    parser.add_argument('--frame-skip', type=int, default=FRAME_SKIP,
                       help='Number of frames to repeat each action (1=no skip, 4=4x faster, default=4)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = PongTrainer(
        num_episodes=args.episodes,
        render_every=args.render_every,
        save_every=args.save_every,
        log_every=args.log_every,
        headless=args.headless,
        num_envs=args.num_envs,
        async_envs=args.async_envs,
        frame_skip=args.frame_skip
    )
    
    # Start training
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
