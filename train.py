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
import config

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
                 num_envs=1, async_envs=False, frame_skip=FRAME_SKIP, train_every=1):
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
        self.train_every = max(1, int(train_every))
        
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
        print(f"Training frequency: update every {self.train_every} simulator steps")
        
        # Adaptive batch size for vectorized training
        # More parallel data = larger batch size for better gradient estimates
        base_batch = getattr(config, 'BATCH_SIZE', BATCH_SIZE)
        self.batch_size = base_batch if num_envs <= 8 else min(base_batch * 2, 256)
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
        self.reward_breakdowns = []
        # Track raw positive/negative reward mass per episode to report prize vs penalty split
        self.reward_signals = []

    def _extract_reward_breakdown(self, infos, env_idx, breakdown_keys):
        """Best-effort extraction of reward_breakdown for a specific env index from vector env infos."""
        rb_env = None

        # Case 1: infos is a list/tuple of per-env dicts (AsyncVectorEnv common)
        if isinstance(infos, (list, tuple)):
            if len(infos) > env_idx and isinstance(infos[env_idx], dict):
                rb_env = infos[env_idx].get('reward_breakdown')
                if rb_env is None and 'final_info' in infos[env_idx] and isinstance(infos[env_idx]['final_info'], dict):
                    rb_env = infos[env_idx]['final_info'].get('reward_breakdown')
            return rb_env

        # Case 2: infos is a dict (SyncVectorEnv) possibly containing nested structures
        if isinstance(infos, dict):
            if 'reward_breakdown' in infos:
                rb_all = infos['reward_breakdown']

                # 2a: tuple/list/ndarray of per-env dicts
                if isinstance(rb_all, (list, tuple, np.ndarray)):
                    if len(rb_all) > env_idx:
                        rb_env = rb_all[env_idx]

                # 2b: dict keyed by env index
                elif isinstance(rb_all, dict) and env_idx in rb_all:
                    rb_env = rb_all.get(env_idx)

                # 2c: dict-of-arrays keyed by breakdown keys (some gym versions do this)
                elif isinstance(rb_all, dict) and set(breakdown_keys).issubset(rb_all.keys()):
                    rb_env = {}
                    for k in breakdown_keys:
                        val = rb_all[k]
                        if isinstance(val, (list, tuple, np.ndarray)) and len(val) > env_idx:
                            rb_env[k] = val[env_idx]
                        else:
                            rb_env[k] = val

            # 2d: reward_breakdown may live inside final_info
            if rb_env is None and 'final_info' in infos:
                fi_all = infos['final_info']
                if isinstance(fi_all, (list, tuple, np.ndarray)):
                    if len(fi_all) > env_idx and isinstance(fi_all[env_idx], dict):
                        rb_env = fi_all[env_idx].get('reward_breakdown')
                elif isinstance(fi_all, dict):
                    maybe_fi = fi_all.get(env_idx)
                    if isinstance(maybe_fi, dict):
                        rb_env = maybe_fi.get('reward_breakdown')

        return rb_env
    
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
        pos_reward = 0.0
        neg_reward = 0.0
        episode_length = 0
        episode_losses = []
        episode_breakdown = {
            'hit_ball': 0.0,
            'score_point': 0.0,
            'lose_point': 0.0,
            'miss_ball': 0.0,
            'proximity': 0.0,
            'far_penalty': 0.0,
            'neutral': 0.0
        }
        
        # Determine if we should render this episode
        should_render = (not self.headless and self.renderer is not None and 
                        self.render_every > 0 and episode_num % self.render_every == 0)
        
        terminated = False
        truncated = False
        action_counts = {0: 0, 1: 0, 2: 0}  # Track action distribution
        
        training_step = 0
        while not (terminated or truncated):
            # Select action for left paddle (AI agent)
            action = self.agent.select_action(state, training=True)
            action_counts[action] += 1
            
            # Execute step in environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # Store transition in replay buffer (single-env uses env_idx=0)
            self.agent._store_transition_internal(state, action, reward, next_state, terminated or truncated, env_idx=0)

            # Track reward sources for this episode if available
            if 'reward_breakdown' in info:
                for key, value in info['reward_breakdown'].items():
                    if key in episode_breakdown:
                        episode_breakdown[key] += value
            
            # Train the agent at configured frequency
            training_step += 1
            if training_step % self.train_every == 0:
                loss = self.agent.train(self.batch_size)
                if loss is not None:
                    episode_losses.append(loss)
            
            # Update state and statistics
            state = next_state
            total_reward += reward
            if reward >= 0:
                pos_reward += reward
            else:
                neg_reward -= reward  # store positive magnitude for penalties
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
            'action_dist': action_dist,
            'reward_breakdown': episode_breakdown,
            'pos_reward': pos_reward,
            'neg_reward': neg_reward
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
        pos_rewards = np.zeros(self.num_envs)
        neg_rewards = np.zeros(self.num_envs)
        all_losses = []
        final_scores_left = []
        final_scores_right = []
        # Track reward breakdown per env and aggregated across envs
        breakdown_keys = ['hit_ball', 'score_point', 'lose_point', 'miss_ball', 'proximity', 'far_penalty', 'neutral']
        breakdown_sums = [dict.fromkeys(breakdown_keys, 0.0) for _ in range(self.num_envs)]
        total_breakdown = dict.fromkeys(breakdown_keys, 0.0)
        
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
                    # Store transition for env i (use internal n-step-aware method)
                    done = terminateds[i] or truncateds[i]
                    self.agent._store_transition_internal(states[i], actions[i], rewards[i], next_states[i], done, env_idx=i)

                    # Update reward breakdown if provided (handle different vector info shapes)
                    rb_env = self._extract_reward_breakdown(infos, i, breakdown_keys)
                    if isinstance(rb_env, dict):
                        for k in breakdown_keys:
                            if k in rb_env:
                                breakdown_sums[i][k] += rb_env[k]
                                total_breakdown[k] += rb_env[k]

                    # Update statistics
                    total_rewards[i] += rewards[i]
                    if rewards[i] >= 0:
                        pos_rewards[i] += rewards[i]
                    else:
                        neg_rewards[i] -= rewards[i]
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
            
            # Train the agent at configured frequency
            training_step += 1
            if training_step % self.train_every == 0:
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
            'score_right': avg_score_right,
            'reward_breakdown': total_breakdown,
            'pos_reward': float(np.sum(pos_rewards)),
            'neg_reward': float(np.sum(neg_rewards))
        }
    
    def _simple_ai_action(self, state):
        """
        Simple AI for opponent (right paddle).
        NOTE: Not used anymore - handled in PongEnv
        """
        pass
    
    def train(self, resume_from=None, resume_eps=None):
        """
        Run the training loop.
        
        Args:
            resume_from (str): Path to checkpoint to resume from (optional)
        """
        start_episode = 0
        
        # Resume from checkpoint if specified
        if resume_from and os.path.exists(resume_from):
            self.agent.load(resume_from)
            # If caller requested a resume epsilon, apply it; otherwise reopen
            # exploration to escape local minima using a conservative default
            if resume_eps is not None:
                self.agent.epsilon = float(resume_eps)
                print(f"Resumed training from {resume_from} (epsilon set to {self.agent.epsilon:.3f})")
            else:
                # Reopen exploration when resuming to escape bad local minima
                self.agent.epsilon = max(self.agent.epsilon, 0.35)
                print(f"Resumed training from {resume_from} (epsilon elevated to {self.agent.epsilon:.3f})")
        
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
        
        print("-" * 60)
        
        start_time = time.time()
        # Print an overall final summary with averages across the whole training run
        self._print_final_summary(start_time, episodes_to_run)
        
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

            # Keep reward breakdown history for logging (single or vectorized)
            self.reward_breakdowns.append(stats.get('reward_breakdown'))
            self.reward_signals.append({
                'pos': stats.get('pos_reward', 0.0),
                'neg': stats.get('neg_reward', 0.0)
            })
            
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
        # Show reward/penalty distribution for recent episodes using raw reward mass
        signal_window = self.reward_signals[-n:]
        total_pos = sum(s.get('pos', 0.0) for s in signal_window)
        total_neg = sum(s.get('neg', 0.0) for s in signal_window)
        total_mass = total_pos + total_neg

        if total_mass == 0:
            print("  └─> Reward mix: total=0 (no signal in window)")
        else:
            pos_pct = 100.0 * total_pos / total_mass
            neg_pct = 100.0 * total_neg / total_mass
            print(
                "  └─> Reward mix ± (last {0} eps): positive={1:.1f}% ({2:.1f}), penalty={3:.1f}% ({4:.1f})".format(
                    n,
                    pos_pct,
                    total_pos,
                    neg_pct,
                    total_neg,
                )
            )

        # Optional: detailed category breakdown if available (percent within positive and within penalty)
        breakdown_window = self.reward_breakdowns[-n:]
        pos_by_key = None
        neg_by_key = None
        for rb in breakdown_window:
            if isinstance(rb, dict):
                if pos_by_key is None:
                    pos_by_key = {k: 0.0 for k in rb.keys()}
                    neg_by_key = {k: 0.0 for k in rb.keys()}
                for k, v in rb.items():
                    if v is None:
                        continue
                    if v > 0:
                        pos_by_key[k] += float(v)
                    elif v < 0:
                        neg_by_key[k] += float(-v)
        if pos_by_key and neg_by_key:
            tot_pos = sum(pos_by_key.values())
            tot_neg = sum(neg_by_key.values())

            if tot_pos > 0:
                print(
                    "  └─> Positive sources (within + only, last {0} eps): hit={1:.1f}%, score={2:.1f}%, prox={3:.1f}%".format(
                        n,
                        100.0 * pos_by_key.get('hit_ball', 0.0) / tot_pos,
                        100.0 * pos_by_key.get('score_point', 0.0) / tot_pos,
                        100.0 * pos_by_key.get('proximity', 0.0) / tot_pos,
                    )
                )
            else:
                print(f"  └─> Positive sources (last {n} eps): none")

            if tot_neg > 0:
                print(
                    "  └─> Penalty sources (within - only, last {0} eps): lose={1:.1f}%, miss={2:.1f}%, far={3:.1f}%, neutral={4:.1f}%".format(
                        n,
                        100.0 * neg_by_key.get('lose_point', 0.0) / tot_neg,
                        100.0 * neg_by_key.get('miss_ball', 0.0) / tot_neg,
                        100.0 * neg_by_key.get('far_penalty', 0.0) / tot_neg,
                        100.0 * neg_by_key.get('neutral', 0.0) / tot_neg,
                    )
                )
            else:
                print(f"  └─> Penalty sources (last {n} eps): none")
    
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

    def _print_final_summary(self, start_time, episodes_to_run):
        """
        Print final summary statistics after training completes.

        Shows:
          - Average prize (reward) per actual episode
          - Average speed (episodes / second)
          - Average positive/penalty mass and category sources across whole run
        """
        elapsed = time.time() - start_time

        # Compute actual episode count (each vectorized iteration contains num_envs episodes)
        total_iterations = len(self.episode_rewards)
        actual_episodes = total_iterations * (self.num_envs if self.vectorized else 1)

        # Average prize (total reward mass per actual episode)
        if self.vectorized:
            total_reward_mass = sum(self.episode_rewards) * self.num_envs
        else:
            total_reward_mass = sum(self.episode_rewards)
        avg_prize = total_reward_mass / actual_episodes if actual_episodes > 0 else 0.0

        # Average speed
        avg_speed = actual_episodes / elapsed if elapsed > 0 else 0.0

        # Positive / Penalty mass across entire run
        total_pos = sum(s.get('pos', 0.0) for s in self.reward_signals)
        total_neg = sum(s.get('neg', 0.0) for s in self.reward_signals)
        total_mass = total_pos + total_neg

        pos_per_episode = total_pos / actual_episodes if actual_episodes > 0 else 0.0
        neg_per_episode = total_neg / actual_episodes if actual_episodes > 0 else 0.0

        print("\nFinal summary (across entire training):")
        print(f"  ✅ Average prize (reward) per episode: {avg_prize:.3f}")
        print(f"  ⚡ Average speed: {avg_speed:.2f} eps/s ({actual_episodes} episodes in {elapsed:.1f}s)")

        if total_mass == 0:
            print("  ℹ️  No reward signal recorded during training (pos/neg totals = 0)")
        else:
            pos_pct = 100.0 * total_pos / total_mass
            neg_pct = 100.0 * total_neg / total_mass
            print(f"  🧭 Reward mass: positive={total_pos:.1f} ({pos_pct:.1f}%), penalty={total_neg:.1f} ({neg_pct:.1f}%)")
            print(f"    → Average / episode: +{pos_per_episode:.3f}  /  -{neg_per_episode:.3f}")

        # Aggregate category breakdown across all episodes
        pos_by_key = {}
        neg_by_key = {}
        for rb in self.reward_breakdowns:
            if not isinstance(rb, dict):
                continue
            for k, v in rb.items():
                if v is None:
                    continue
                if v > 0:
                    pos_by_key[k] = pos_by_key.get(k, 0.0) + float(v)
                elif v < 0:
                    neg_by_key[k] = neg_by_key.get(k, 0.0) + float(-v)

        if pos_by_key:
            tot_pos = sum(pos_by_key.values())
            if tot_pos > 0:
                print("  └─> Positive sources (distribution over + mass):")
                for k, v in sorted(pos_by_key.items(), key=lambda x: -x[1])[:5]:
                    print(f"       - {k}: {100.0 * v / tot_pos:.1f}% ({v:.1f})")
        if neg_by_key:
            tot_neg = sum(neg_by_key.values())
            if tot_neg > 0:
                print("  └─> Penalty sources (distribution over - mass):")
                for k, v in sorted(neg_by_key.items(), key=lambda x: -x[1])[:5]:
                    print(f"       - {k}: {100.0 * v / tot_neg:.1f}% ({v:.1f})")


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
    parser.add_argument('--train-every', type=int, default=4,
                       help='How many simulator steps between weight updates (default=4)')
    parser.add_argument('--use-prioritized', action='store_true', help='Use prioritized replay buffer (overrides config)')
    parser.add_argument('--n-step', type=int, default=None, help='N for n-step returns (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size from config')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--eps', type=float, default=None,
                       help='Initial epsilon to set (overrides checkpoint/default). Range 0.0-1.0')
    parser.add_argument('--fresh-start', action='store_true',
                       help='Ignore existing model and start from scratch (disables auto-resume)')
    
    args = parser.parse_args()

    # Determine resume path: default to existing base model unless user opts out
    resume_path = None
    if args.fresh_start:
        if args.resume:
            print('Warning: --fresh-start set, ignoring --resume path')
    else:
        candidate = args.resume or MODEL_SAVE_PATH
        if candidate and os.path.exists(candidate):
            resume_path = candidate
            if args.resume is None and candidate == MODEL_SAVE_PATH:
                print(f'Auto-resuming from default model at {candidate}')
        elif args.resume:
            print(f'Warning: resume path {candidate} not found, starting fresh')
    
    # Validate epsilon flag if provided
    if args.eps is not None:
        if args.eps < 0.0 or args.eps > 1.0:
            parser.error('--eps must be between 0.0 and 1.0')

    # Create trainer
    # Allow runtime overrides for prioritized replay / n-step / batch size
    if args.use_prioritized:
        try:
            import config
            config.USE_PRIORITIZED_REPLAY = True
            print("Enabled prioritized replay (override)")
        except Exception:
            pass
    if args.n_step is not None:
        try:
            import config
            config.N_STEP = max(1, int(args.n_step))
            print(f"Using N-step = {config.N_STEP} (override)")
        except Exception:
            pass
    if args.batch_size is not None:
        try:
            import config
            config.BATCH_SIZE = max(1, int(args.batch_size))
            print(f"Batch size overridden to {config.BATCH_SIZE}")
        except Exception:
            pass

    trainer = PongTrainer(
        num_episodes=args.episodes,
        render_every=args.render_every,
        save_every=args.save_every,
        log_every=args.log_every,
        headless=args.headless,
        num_envs=args.num_envs,
        async_envs=args.async_envs,
        frame_skip=args.frame_skip,
        train_every=args.train_every
    )
    
    # Start training (pass resume epsilon through)
    trainer.train(resume_from=resume_path, resume_eps=args.eps)


if __name__ == "__main__":
    main()
