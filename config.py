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
WINNING_SCORE = 5    # Points needed to win (longer episodes = more training steps)
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
REWARD_HIT_BALL = 12.0       # Reward for hitting the ball with paddle (strong positive signal)
REWARD_SCORE_POINT = 8.0   # DISABLED FOR NOW | Reward for scoring a point (main objective!)
REWARD_LOSE_POINT = 0.0   # DISABLED FOR NOW | Penalty for opponent scoring (strong penalty)
# Slightly stronger step penalty to discourage idling
REWARD_NEUTRAL = -0.00     # Tiny step cost to discourage doing nothing (penalize stalling)
# Increase proximity shaping so paddle is more strongly pulled toward incoming ball
REWARD_PROXIMITY = 1.5      # Reward shaping to pull paddle toward incoming ball
REWARD_MISS_BALL = -5.0     # Penalty when ball passes by paddle (strong penalty)
# Make far penalty stronger (more negative) to discourage sitting far from incoming ball
REWARD_FAR_PENALTY = -0.2   # Stronger penalty when far from ball

# DQN Hyperparameters
LEARNING_RATE = 0.0001       # Learning rate (increased to escape local minimum)
GAMMA = 0.99                # Discount factor for future rewards
EPSILON_START = 0.9         # Starting epsilon for epsilon-greedy (slightly lower to favor early policy)
EPSILON_END = 0.02           # Minimum epsilon value (reduce final exploration)
EPSILON_DECAY = 0.997       # Epsilon decay rate per episode (faster decay to exploit learned policy)
BATCH_SIZE = 128            # Batch size for training (larger for better GPU utilization)
MEMORY_SIZE = 100_000         # Replay buffer capacity (reduce to save RAM)
TARGET_UPDATE = 5           # Update target network every N episodes (frequent updates)

# Neural Network Architecture
N_STEP = 3                  # N-step returns (helps propagate reward to earlier transitions)
USE_PRIORITIZED_REPLAY = False  # Use prioritized replay buffer (may improve sample efficiency)

# Reduced sizes for better throughput on lightweight hardware (e.g., M1 MacBook Air)
# These are conservative defaults to balance speed and representational capacity
HIDDEN_SIZE_1 = 128         # First hidden layer size
HIDDEN_SIZE_2 = 128         # Second hidden layer size
BATCH_SIZE = 256            # Batch size for training (better utilization when training less frequently)
INPUT_SIZE = 8              # State size: [ball_x, ball_y, ball_vel_x, ball_vel_y, paddle1_center_y, paddle2_center_y, ball_speed_abs, ball_dist_paddle1]
OUTPUT_SIZE = 3             # Action size: [0=none, 1=up, 2=down]

# Training Configuration
NUM_EPISODES = 3000         # Total number of training episodes (increased for better learning)
RENDER_EVERY = 0          # Render every N episodes (0 = never render during training)
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