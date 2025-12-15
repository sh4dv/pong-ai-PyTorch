"""
Game configuration and constants for Pong game.
All dimensions, speeds, and parameters are defined here.
"""

# Window dimensions
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Paddle settings
PADDLE_WIDTH = 15
PADDLE_HEIGHT = 90
PADDLE_SPEED = 10
PADDLE_OFFSET = 30  # Distance from edge of screen
OPPONENT_SPEED_MIN = 5   # Min speed for right paddle (AI)
OPPONENT_SPEED_MAX = 10   # Max speed for right paddle (AI)

# Ball settings
BALL_SIZE = 15
BALL_SPEED_X = 8
BALL_SPEED_Y = 8
BALL_MAX_SPEED = 15
LAUNCH_ANGLE_MIN_DEG = 15  # Min launch angle off horizontal (avoid straight lines)
LAUNCH_ANGLE_MAX_DEG = 60  # Max launch angle off horizontal

# Game settings
WINNING_SCORE = 3    # Points needed to win (longer episodes = more training steps)
FPS = 60
FRAME_SKIP = 1      # Number of frames to repeat each action (1 = no skip, 4 = 4x faster)

# Colors (RGB)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)

# ============================================================================
# REINFORCEMENT LEARNING SETTINGS
# ============================================================================

# Reward values
REWARD_HIT_BALL = 50.0       # Reward for hitting the ball with paddle (strong positive signal)
REWARD_SCORE_POINT = 5.0   # Reward for scoring a point (main objective!)
REWARD_LOSE_POINT = -5.0   # Penalty for opponent scoring (strong penalty)
REWARD_NEUTRAL = 0.0     # DISABLED FOR NOW | Tiny step cost to avoid rewarding stalling
REWARD_PROXIMITY = 2.0      # Reward shaping to pull paddle toward incoming ball
REWARD_MISS_BALL = -20.0     # Penalty when ball passes by paddle (strong penalty)
REWARD_FAR_PENALTY = -0.05   # Softer penalty when far from ball

# DQN Hyperparameters
LEARNING_RATE = 0.0001       # Learning rate (increased to escape local minimum)
GAMMA = 0.99                # Discount factor for future rewards
EPSILON_START = 1.0         # Starting epsilon for epsilon-greedy
EPSILON_END = 0.1           # Minimum epsilon value (higher for continuous exploration)
EPSILON_DECAY = 0.997       # Epsilon decay rate per episode (slower decay)
BATCH_SIZE = 256            # Batch size for training (larger for better GPU utilization)
MEMORY_SIZE = 35000         # Replay buffer capacity (reduced to save RAM)
TARGET_UPDATE = 3           # Update target network every N episodes (frequent updates)

# Neural Network Architecture
HIDDEN_SIZE_1 = 256         # First hidden layer size (increased for better GPU utilization)
HIDDEN_SIZE_2 = 256         # Second hidden layer size (increased for better GPU utilization)
INPUT_SIZE = 6              # State size: [ball_x, ball_y, ball_vel_x, ball_vel_y, paddle1_y, paddle2_y]
OUTPUT_SIZE = 3             # Action size: [0=none, 1=up, 2=down]

# Training Configuration
NUM_EPISODES = 3000         # Total number of training episodes (increased for better learning)
RENDER_EVERY = 100          # Render every N episodes (0 = never render during training)
SAVE_EVERY = 100            # Save model checkpoint every N episodes
LOG_EVERY = 10              # Print training stats every N episodes (5 = ~15-30 sec, 10 = ~1-2 min)
DEBUG_ACTIONS_EVERY = 50    # Log action distribution every N episodes for debugging
MIN_EPISODES_FOR_LEARNING = 1000  # Minimum episodes recommended for decent performance

# Training Modes
TRAIN_HEADLESS = "headless"         # Train without rendering
TRAIN_WITH_RENDER = "render"        # Train with periodic rendering
PLAY_MODE = "play"                  # Play with trained model

# Model save paths
MODEL_SAVE_PATH = "models/dqn_pong.pth"
CHECKPOINT_DIR = "models/checkpoints/"