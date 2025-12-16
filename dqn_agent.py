"""
DQN Agent for Pong game.
Implements Deep Q-Learning with experience replay and target network.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn_model import DQN
from replay_buffer import ReplayBuffer
from config import *


class DQNAgent:
    """
    Deep Q-Network agent with epsilon-greedy exploration.
    
    Features:
    - Experience replay for stable training
    - Target network for stable Q-value targets
    - Epsilon-greedy exploration with decay
    - Support for MPS (Apple Silicon), CUDA, and CPU
    """
    
    def __init__(self, state_size=INPUT_SIZE, action_size=OUTPUT_SIZE, 
                 learning_rate=LEARNING_RATE, gamma=GAMMA, 
                 epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
                 epsilon_decay=EPSILON_DECAY, memory_size=MEMORY_SIZE):
        """
        Initialize the DQN agent.
        
        Args:
            state_size (int): Dimension of state space
            action_size (int): Number of possible actions
            learning_rate (float): Learning rate for optimizer
            gamma (float): Discount factor
            epsilon_start (float): Initial exploration rate
            epsilon_end (float): Minimum exploration rate
            epsilon_decay (float): Epsilon decay rate per episode
            memory_size (int): Replay buffer capacity
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Set device (MPS for M1, CUDA for GPU, CPU otherwise)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Apple Silicon) for training")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA for training")
        else:
            self.device = torch.device("cpu")
            print("Using CPU for training")
        
        # Create policy network and target network
        self.policy_net = DQN(state_size, HIDDEN_SIZE_1, HIDDEN_SIZE_2, action_size).to(self.device)
        self.target_net = DQN(state_size, HIDDEN_SIZE_1, HIDDEN_SIZE_2, action_size).to(self.device)
        
        # Copy weights from policy to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        # Huber loss is more robust to occasional outliers than MSE
        self.criterion = nn.SmoothL1Loss()
        
        # Replay buffer
        self.memory = ReplayBuffer(memory_size)
        
        # Training statistics
        self.training_step = 0
    
    def select_actions_batch(self, states, training=True):
        """
        Select actions for multiple states at once (GPU efficient).
        
        Args:
            states (np.array): Batch of states (num_envs, state_size)
            training (bool): Whether in training mode (uses epsilon-greedy)
        
        Returns:
            np.array: Selected actions for each state
        """
        batch_size = len(states)
        actions = np.zeros(batch_size, dtype=np.int64)
        
        if training:
            # Epsilon-greedy for each state
            random_mask = np.random.random(batch_size) < self.epsilon
            actions[random_mask] = np.random.randint(0, self.action_size, size=np.sum(random_mask))
            
            # Greedy actions for non-random states (batched GPU computation)
            if not random_mask.all():
                greedy_indices = np.where(~random_mask)[0]
                states_tensor = torch.FloatTensor(states[greedy_indices]).to(self.device)
                with torch.no_grad():
                    q_values = self.policy_net(states_tensor)
                    greedy_actions = q_values.argmax(dim=1).cpu().numpy()
                actions[greedy_indices] = greedy_actions
                del states_tensor, q_values
        else:
            # All greedy (batched)
            states_tensor = torch.FloatTensor(states).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(states_tensor)
                actions = q_values.argmax(dim=1).cpu().numpy()
            del states_tensor, q_values
        
        return actions
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state (np.array): Current state
            training (bool): Whether in training mode (uses epsilon-greedy)
        
        Returns:
            int: Selected action (0=none, 1=up, 2=down)
        """
        # During evaluation, always use greedy policy
        if not training:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax().item()
            del state_tensor, q_values
            return action
        
        # Epsilon-greedy exploration during training
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)  # Random action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax().item()  # Greedy action
            del state_tensor, q_values
            return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in replay buffer.
        
        Args:
            state (np.array): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.array): Next state
            done (bool): Whether episode ended
        """
        self.memory.add(state, action, reward, next_state, done)
    
    def train(self, batch_size=BATCH_SIZE):
        """
        Train the network using a batch from replay buffer.
        
        Args:
            batch_size (int): Size of training batch
        
        Returns:
            float: Loss value (None if not enough samples)
        """
        # Check if we have enough samples
        if not self.memory.is_ready(batch_size):
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Guard against NaN/Inf coming from the environment or replay buffer
        if not np.isfinite(states).all() or not np.isfinite(next_states).all() or not np.isfinite(rewards).all():
            print("⚠️  Warning: Non-finite values in batch, skipping")
            return 0.0

        # Clip extreme rewards to keep targets bounded (manual play rewards are smaller)
        rewards = np.clip(rewards, -20.0, 50.0)
        
        # Convert to tensors and move to device (non_blocking for speed)
        states = torch.FloatTensor(states).to(self.device, non_blocking=True)
        actions = torch.LongTensor(actions).to(self.device, non_blocking=True)
        rewards = torch.FloatTensor(rewards).to(self.device, non_blocking=True)
        next_states = torch.FloatTensor(next_states).to(self.device, non_blocking=True)
        dones = torch.FloatTensor(dones).to(self.device, non_blocking=True)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            # Clamp targets to keep them in a sane numeric range
            target_q_values = torch.clamp(target_q_values, -50.0, 50.0)
            # Keep targets conservative to avoid exploding losses in online/manual play
        
        # Clamp current Qs as well to reduce gradient spikes from rare outliers
        current_q_values = torch.clamp(current_q_values, -50.0, 50.0)

        # Replace any non-finite values before loss to prevent blowups
        if not torch.isfinite(current_q_values).all() or not torch.isfinite(target_q_values).all():
            # Detailed diagnostics for debugging NaN/Inf issues
            print("⚠️  Warning: Non-finite Q values, skipping batch")
            try:
                import time
                ts = int(time.time())
                # Summaries (use nan-aware ops)
                try:
                    curr_min = torch.nanmin(current_q_values).item()
                    curr_max = torch.nanmax(current_q_values).item()
                    curr_mean = torch.nanmean(current_q_values).item()
                except Exception:
                    curr_min = curr_max = curr_mean = float('nan')

                try:
                    targ_min = torch.nanmin(target_q_values).item()
                    targ_max = torch.nanmax(target_q_values).item()
                    targ_mean = torch.nanmean(target_q_values).item()
                except Exception:
                    targ_min = targ_max = targ_mean = float('nan')

                print(f"  current_q -> min={curr_min}, max={curr_max}, mean={curr_mean}")
                print(f"  target_q  -> min={targ_min}, max={targ_max}, mean={targ_mean}")

                # Show indices of first few non-finite entries
                nonfin_curr_idx = torch.where(~torch.isfinite(current_q_values))[0][:10].cpu().numpy()
                nonfin_targ_idx = torch.where(~torch.isfinite(target_q_values))[0][:10].cpu().numpy()
                print(f"  non-finite current_q indices (up to 10): {nonfin_curr_idx}")
                print(f"  non-finite target_q indices  (up to 10): {nonfin_targ_idx}")

                # Save offending batch and a small snapshot of model/optimizer for offline inspection
                debug_path = os.path.join("models", f"debug_nonfinite_batch_{ts}.npz")
                try:
                    np.savez_compressed(
                        debug_path,
                        states=states.detach().cpu().numpy(),
                        actions=actions.detach().cpu().numpy(),
                        rewards=rewards.detach().cpu().numpy(),
                        next_states=next_states.detach().cpu().numpy(),
                        dones=dones.detach().cpu().numpy(),
                        current_q=current_q_values.detach().cpu().numpy(),
                        target_q=target_q_values.detach().cpu().numpy(),
                    )
                    print(f"  Debug batch saved to {debug_path}")
                except Exception as e:
                    print(f"  Failed to save debug batch: {e}")

                # Check model parameters for NaN/Inf
                def inspect_params(net, name):
                    bad = []
                    for n, p in net.named_parameters():
                        if not torch.isfinite(p).all():
                            bad.append(n)
                    if bad:
                        print(f"  {name} has non-finite params: {bad}")
                    else:
                        # Print parameter norms for quick sanity checking
                        total_norm = 0.0
                        for p in net.parameters():
                            total_norm += float(torch.norm(p).item())
                        print(f"  {name} parameter norm (sum of L2 norms): {total_norm:.3f}")

                inspect_params(self.policy_net, "policy_net")
                inspect_params(self.target_net, "target_net")

                # Inspect optimizer state (e.g. Adam running averages)
                try:
                    bad_opt = False
                    for sk, sv in self.optimizer.state.items():
                        for k, v in sv.items():
                            if isinstance(v, torch.Tensor):
                                if not torch.isfinite(v).all():
                                    bad_opt = True
                    print(f"  Optimizer state contains non-finite tensors: {bad_opt}")
                except Exception:
                    print("  Failed to inspect optimizer state")
            except Exception as e:
                print(f"  Diagnostic logging failed: {e}")

            return 0.0

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Check for NaN before backward pass
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️  Warning: Loss is {loss.item()}, skipping this batch")
            return 0.0
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients (more aggressive)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        
        self.training_step += 1
        
        # Detach and cleanup to prevent memory leaks
        loss_value = loss.item()
        
        # Final safety check for nan/inf
        if not np.isfinite(loss_value):
            print(f"⚠️  Warning: Loss is {loss_value} after training, returning 0")
            loss_value = 0.0

        # Quick post-step sanity check: ensure parameters are finite
        try:
            any_bad = False
            for p in self.policy_net.parameters():
                if not torch.isfinite(p).all():
                    any_bad = True
                    break
            if any_bad:
                ts = int(__import__('time').time())
                snapshot = os.path.join("models", f"nan_params_snapshot_{ts}.pth")
                torch.save(self.policy_net.state_dict(), snapshot)
                print(f"⚠️  Non-finite parameters detected; model snapshot saved to {snapshot}. Resetting optimizer state.")
                # reset optimizer state to avoid perpetuating NaNs in running averages
                try:
                    self.optimizer.state = {}
                except Exception:
                    pass
                # Return early - this batch is not reliable
                return 0.0
        except Exception:
            # Non-fatal: keep running but inform
            print("⚠️  Warning: Failed post-step parameter sanity check")
        
        del loss, current_q_values, target_q_values, next_q_values
        
        return loss_value
    
    def update_target_network(self):
        """Copy weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """
        Save model checkpoint.
        
        Args:
            filepath (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load model checkpoint.
        
        Args:
            filepath (str): Path to load the model from
        """
        if not os.path.exists(filepath):
            print(f"Error: Model file {filepath} not found")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        
        print(f"Model loaded from {filepath}")
        return True
    
    def get_epsilon(self):
        """Get current epsilon value."""
        return self.epsilon
