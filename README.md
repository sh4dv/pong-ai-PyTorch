# Pong AI - Deep Q-Network Training

A complete implementation of a Deep Q-Network (DQN) agent that learns to play Pong using PyTorch. Optimized for Apple Silicon (M1/M2) with MPS support and **vectorized environments for 4-8x faster training**.

## Features

- **DQN Architecture**: Multi-layer perceptron with 2 hidden layers and ReLU activation
- **Experience Replay**: Efficient memory buffer for stable training
- **Target Network**: Separate target network updated periodically for stable Q-value estimation
- **Epsilon-Greedy Exploration**: Adaptive exploration with decay
- **Vectorized Training**: Train on 4-16 game instances simultaneously for massive speedup
- **Flexible Training Modes**: 
  - Headless training (fast)
  - Training with periodic rendering
  - Playing with trained model
- **Apple Silicon Optimization**: Automatic MPS device detection for M1/M2 Macs
- **Checkpointing**: Save and resume training at any point
- **N-step returns & Prioritized Replay**: Optional enhancements (configurable in `config.py`)
- **Robust Training Diagnostics**: NaN/Inf guards, gradient checks and automatic safety fallbacks

## Project Structure

```
.
├── config.py           # All configuration and hyperparameters
├── pong_game.py        # Game logic (headless-capable)
├── pong_env.py         # Gymnasium environment wrapper
├── renderer.py         # Optional PyGame rendering
├── dqn_model.py        # Neural network architecture
├── replay_buffer.py    # Experience replay buffer
├── dqn_agent.py        # DQN agent with training logic
├── train.py            # Training script with vectorization
├── play.py             # Play with trained model
├── evaluate.py         # Evaluation script (greedy eval / breakdowns)
├── manual_train.py     # Human-in-the-loop training utilities
├── check_model_status.py # Quick model diagnostics and smoke test
├── cleanup_envs.py     # Utilities to manage/cleanup env processes
├── main.py             # Human vs human game
├── benchmark/          # Benchmark scripts (envs / gpu / parallel)
└── requirements.txt    # Dependencies
```

## Installation

1. **Create and activate virtual environment (cross-platform):**
   - macOS / Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - Windows (cmd):
     ```cmd
     python -m venv .venv
     .\.venv\Scripts\activate
     ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **PyTorch / GPU notes:**
   - For CUDA or platform-specific wheels (GPU/MPS), follow the official PyTorch install page:
     https://pytorch.org/get-started/locally/
   - On Apple Silicon (M1/M2) PyTorch will expose MPS when available (requires PyTorch 2.0+).

## Usage

### 1. Train the Agent

**Standard training (1 game at a time):**
```bash
python train.py --headless --episodes 1000
```

**🚀 Vectorized training (4 games simultaneously - 4x faster!):**
```bash
python train.py --episodes 1000 --num-envs 4
```

**⚡ Super fast async training (8 games simultaneously - 8x faster!):**
```bash
python train.py --episodes 1000 --num-envs 8 --async-envs
```

**Training with periodic rendering (every 100 episodes, only with --num-envs 1):**
```bash
python train.py --render-every 100 --episodes 1000
```

**Custom training configuration:**
```bash
python train.py --episodes 2000 --num-envs 4 --save-every 100 --log-every 10 --headless
```

**Resume from checkpoint:**
```bash
python train.py --resume models/checkpoints/checkpoint_ep500.pth --episodes 1000 --num-envs 4
```

**Note about vectorized training:** When `--num-envs > 1`, `--episodes` is treated as the total number of episodes across all environments. The trainer divides episodes into iterations so that `--episodes 1000 --num-envs 4` runs 250 iterations × 4 envs = 1000 total episodes.

**Useful CLI flags:**
- `--eps` — set initial epsilon when resuming (`0.0-1.0`)
- `--fresh-start` — ignore existing model and start from scratch (will override current dqn_pong.pth)
- `--train-every` — number of simulator steps between updates
- `--frame-skip` — repeat each action for N frames (increase to speed up training, recommended 2-4)
- `--use-prioritized` / `--n-step` — override replay/prioritization settings at runtime (recommended to add --use-prioritized)

### Quick utilities
- Evaluate a trained model (greedy evaluation, reward breakdown):
```bash
python evaluate.py --model models/dqn_pong.pth --episodes 100
```
- Manual (human-in-the-loop) online training:
```bash
python manual_train.py --games 10
```
- Quick model diagnostics / smoke test:
```bash
python check_model_status.py --model models/dqn_pong.pth
```
- Run benchmarks (see `benchmark/`):
```bash
python benchmark/benchmark_envs.py
```

### 2. Play with Trained Model

*"Simple AI" is just a paddle simply following the ball - no AI here.
**Watch AI vs Simple AI:**
```bash
python play.py
```

**Play against the AI yourself:**
```bash
python play.py --opponent human
```

**Use specific model:**
```bash
python play.py --model models/checkpoints/checkpoint_ep500.pth
```

### 3. Human vs Human Mode

```bash
python main.py
```

Controls:
- Left player: W (up), S (down)
- Right player: Arrow keys

## Configuration

All hyperparameters live in `config.py` (source of truth). Below are the current defaults used by the codebase:

### Reward System
- `REWARD_HIT_BALL = 12.0` - Reward for hitting the ball
- `REWARD_SCORE_POINT = 20.0` - Reward for scoring a point
- `REWARD_LOSE_POINT = 0.0` - (disabled by default) Penalty for opponent scoring
- `REWARD_NEUTRAL = -0.00` - (disabled by default) Small step penalty to discourage stalling
- `REWARD_PROXIMITY = 1.5` - Shaping reward for being close to incoming ball
- `REWARD_MISS_BALL = -8.0` - Penalty when ball is missed
- `REWARD_FAR_PENALTY = -0.2` - Penalty when far from the ball

### DQN Hyperparameters
- `LEARNING_RATE = 0.0003`
- `GAMMA = 0.99`
- `EPSILON_START = 0.9`
- `EPSILON_END = 0.05`
- `EPSILON_DECAY = 0.997`
- `BATCH_SIZE = 256` *(trainer can adapt batch size when using many envs)*
- `MEMORY_SIZE = 100000`
- `TARGET_UPDATE = 5`
- `N_STEP = 3`
- `USE_PRIORITIZED_REPLAY = False`

### Network Architecture
- `INPUT_SIZE = 8` - State vector size
- `HIDDEN_SIZE_1 = 128` - First hidden layer
- `HIDDEN_SIZE_2 = 128` - Second hidden layer
- `OUTPUT_SIZE = 3` - Number of actions

### Training configuration
- `NUM_EPISODES = 3000` - Default total number of episodes
- `FRAME_SKIP = 1` - Number of frames repeated per action (increase to speed up, e.g., 4)
- `SAVE_EVERY = 100`
- `LOG_EVERY = 10`

Note: These are the defaults at the time of writing; if you want to change them permanently edit `config.py`, or override at runtime using the CLI flags (e.g. `--n-step`, `--batch-size`, `--use-prioritized`).

## How It Works

### State Representation
The agent observes the game state as an 8-dimensional vector:
```python
[ball_x, ball_y, ball_vel_x, ball_vel_y, paddle1_center_y, paddle2_center_y, ball_speed_abs, ball_dist_to_paddle1]
```

Note: the last element is a signed vertical offset (ball relative to left paddle, roughly in [-1,1]) and feature normalization is applied in `PongEnv` / `play.py` (see source for exact preprocessing).

### Action Space
The agent can take 3 actions:
- `0` - No movement
- `1` - Move paddle up
- `2` - Move paddle down

### Training Process

1. **Initialization**: Create policy network and target network with same weights
2. **Episode Loop**:
   - Reset game environment
   - For each step:
     - Select action using epsilon-greedy policy
     - Execute action in environment
     - Receive reward and next state
     - Store transition in replay buffer
     - Sample random batch from buffer
     - Compute Q-values and target Q-values
     - Update policy network using MSE loss
   - Decay epsilon for less exploration over time
   - Periodically update target network
3. **Checkpointing**: Save model periodically and at end of training

### DQN Algorithm

The agent uses the Deep Q-Learning algorithm:
- **Q-Learning Update**: `Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]`
- **Experience Replay**: Random sampling breaks temporal correlation
- **Target Network**: Stabilizes training by fixing Q-targets for multiple updates
- **Epsilon-Greedy**: Balances exploration vs exploitation

## Performance Tips

### For Faster Training:
- Use headless mode: `--headless`
- Increase batch size in `config.py`
- Reduce logging frequency: `--log-every 50`

### For Better Results:
- Train for more episodes (2000+)
- Tune reward values in `config.py`
- Adjust network architecture (hidden layer sizes)
- Experiment with learning rate and gamma

### Apple Silicon Optimization:
The code automatically detects and uses MPS (Metal Performance Shaders) on M1/M2 Macs for significant speedup.

## Training Output Example

```
Starting training for 1000 episodes
Headless mode: True
Device: mps
------------------------------------------------------------
Episode 10/1000 | Reward: -5.30 | Length: 324 | Loss: 0.1234 | Score L/R: 2.1/10.0 | Epsilon: 0.951 | Speed: 12.34 eps/s
Episode 20/1000 | Reward: -3.20 | Length: 412 | Loss: 0.0987 | Score L/R: 3.5/10.0 | Epsilon: 0.904 | Speed: 13.21 eps/s
...
Episode 1000/1000 | Reward: 8.70 | Length: 856 | Loss: 0.0234 | Score L/R: 10.0/4.2 | Epsilon: 0.010 | Speed: 15.67 eps/s
```

## Saved Models

Models are saved in:
- `models/dqn_pong.pth` - Final trained model
- `models/checkpoints/checkpoint_ep{N}.pth` - Periodic checkpoints

## Troubleshooting

**ImportError for torch:**
```bash
pip install torch
```

**No MPS device on M1 Mac:**
Make sure you have PyTorch 2.0+ installed. The code will fall back to CPU if MPS is unavailable.

**Game too fast/slow:**
Adjust `FPS` in `config.py`

**Training not improving:**
- Try different reward values
- Increase training episodes
- Adjust learning rate or network size
- Check epsilon decay rate

## Future Enhancements

- [ ] Train both paddles simultaneously
- [ ] Add more sophisticated opponents
- [ ] Implement Double DQN or Dueling DQN
- [ ] Add TensorBoard logging
- [ ] Implement prioritized experience replay
- [ ] Add curriculum learning

## License

This project is released under the MIT License.

Author: sh4dv

Created for reinforcement learning education and experimentation.

## Contributing

Contributions, bug reports and PRs are welcome. Please open an issue first if you plan a larger change. Keep changes small and focused and include reproducible steps for any bugfix.
