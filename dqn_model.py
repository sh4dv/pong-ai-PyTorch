"""
Deep Q-Network (DQN) model architecture for Pong AI.
Multi-layer perceptron with ReLU activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE


class DQN(nn.Module):
    """
    Deep Q-Network with 2 hidden layers.
    
    Architecture:
        Input Layer:  6 neurons (state: ball_x, ball_y, ball_vel_x, ball_vel_y, paddle1_y, paddle2_y)
        Hidden Layer 1: 128 neurons with ReLU activation
        Hidden Layer 2: 128 neurons with ReLU activation
        Output Layer: 3 neurons (Q-values for actions: none, up, down)
    """
    
    def __init__(self, input_size=INPUT_SIZE, hidden1=HIDDEN_SIZE_1, 
                 hidden2=HIDDEN_SIZE_2, output_size=OUTPUT_SIZE):
        """
        Initialize the DQN model.
        
        Args:
            input_size (int): Number of input features (state size)
            hidden1 (int): Size of first hidden layer
            hidden2 (int): Size of second hidden layer
            output_size (int): Number of actions (output size)
        """
        super(DQN, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        
        # Initialize weights using He initialization (good for ReLU)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using He initialization."""
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.xavier_normal_(self.fc3.weight)  # Xavier for output layer
        
        # Initialize biases - zero for output layer to avoid action bias!
        nn.init.constant_(self.fc1.bias, 0.0)  # Zero instead of 0.01
        nn.init.constant_(self.fc2.bias, 0.0)  # Zero instead of 0.01
        nn.init.constant_(self.fc3.bias, 0.0)  # CRITICAL: Zero bias = no action preference
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state tensor of shape (batch_size, input_size)
        
        Returns:
            torch.Tensor: Q-values for each action, shape (batch_size, output_size)
        """
        x = F.relu(self.fc1(x))  # First hidden layer with ReLU
        x = F.relu(self.fc2(x))  # Second hidden layer with ReLU
        x = self.fc3(x)          # Output layer (no activation, raw Q-values)
        return x
    
    def get_action(self, state):
        """
        Get the best action for a given state (greedy policy).
        
        Args:
            state (torch.Tensor): State tensor
        
        Returns:
            int: Action with highest Q-value
        """
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax().item()
