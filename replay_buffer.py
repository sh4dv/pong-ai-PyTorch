"""
Experience Replay Buffer for DQN training.
Stores transitions and samples random batches for training.
"""

import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Circular buffer for storing experience transitions.
    
    Stores tuples of (state, action, reward, next_state, done) and
    provides random sampling for breaking correlation in training data.
    """
    
    def __init__(self, capacity):
        """
        Initialize the replay buffer.
        
        Args:
            capacity (int): Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state (np.array): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.array): Next state
            done (bool): Whether episode ended
        """
        # Guard against storing non-finite values which later poison training
        try:
            s = np.array(state, dtype=np.float32)
            ns = np.array(next_state, dtype=np.float32)
            r = float(reward)
            if not (np.isfinite(s).all() and np.isfinite(ns).all() and np.isfinite(r)):
                print("⚠️  Warning: Attempt to store non-finite transition, skipping")
                return
            # Sanity check: skip storing extremely large but finite values which
            # are likely caused by env/model corruption (prevent poisoning replay)
            if (np.abs(s) > 1e6).any() or (np.abs(ns) > 1e6).any():
                print("⚠️  Warning: Attempt to store out-of-range transition (abs>1e6), skipping")
                return
        except Exception:
            # In case of unexpected types, skip the entry rather than crash
            print("⚠️  Warning: Invalid transition format, skipping")
            return

        # Clip reward to reasonable range to avoid exploding n-step returns
        try:
            reward = float(np.clip(reward, -50.0, 50.0))
        except Exception:
            reward = float(reward)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size (int): Number of transitions to sample
        
        Returns:
            tuple: (states, actions, rewards, next_states, dones) as numpy arrays
        """
        # Sample random transitions
        batch = random.sample(self.buffer, batch_size)
        
        # Unzip the batch into separate arrays
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """
        Get current size of buffer.
        
        Returns:
            int: Number of stored transitions
        """
        return len(self.buffer)
    
    def is_ready(self, batch_size):
        """
        Check if buffer has enough samples for training.
        
        Args:
            batch_size (int): Required batch size
        
        Returns:
            bool: True if buffer has at least batch_size samples
        """
        return len(self.buffer) >= batch_size
    
    def clear(self):
        """
        Clear all stored transitions from buffer.
        Useful for cleanup and preventing memory leaks.
        """
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """
    Simple proportional prioritized replay buffer (prototype).
    Uses a list/deque for transitions and a parallel list for priorities.
    This is not a highly optimized sum-tree implementation but is sufficient
    for moderate buffer sizes and experimentation.
    """

    def __init__(self, capacity, alpha=0.6, eps=1e-6):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.eps = eps

    def add(self, state, action, reward, next_state, done):
        # priority: new items get max priority so they are sampled at least once
        max_prio = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_prio)

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            raise ValueError("Empty buffer")

        prios = np.array(self.priorities, dtype=np.float64)
        probs = prios ** self.alpha
        s = probs.sum()
        if s <= 0 or not np.isfinite(s):
            # Fallback to uniform probabilities if numeric issues occur
            probs = np.ones_like(probs, dtype=np.float64)
            s = probs.sum()
        probs /= s

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]

        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # importance-sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize
        weights = weights.astype(np.float32)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, p in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = abs(p) + self.eps

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size):
        return len(self.buffer) >= batch_size

    def clear(self):
        self.buffer.clear()
        self.priorities.clear()
