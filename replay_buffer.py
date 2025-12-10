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
