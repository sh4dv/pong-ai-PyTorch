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
├── main.py             # Human vs human game
└── requirements.txt    # Dependencies
```

## Installation

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   For Apple Silicon (M1/M2), PyTorch will automatically use MPS acceleration.

## Usage

### 1. Train the Agent

**Standard training (1 game at a time):**
```bash
python train.py --headless --episodes 1000
```

**🚀 Vectorized training (4 games simultaneously - 4x faster!):**
```bash
python train.py --headless --episodes 1000 --num-envs 4
```

**⚡ Super fast training (8 games simultaneously - 8x faster!):**
```bash
python train.py --headless --episodes 1000 --num-envs 8
```

**💨 Maximum speed (16 games async - 12x faster!):**
```bash
python train.py --headless --episodes 2000 --num-envs 16 --async-envs
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

### 2. Play with Trained Model

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

All hyperparameters can be adjusted in `config.py`:

### Reward System
- `REWARD_HIT_BALL = 1.0` - Reward for hitting the ball
- `REWARD_SCORE_POINT = 10.0` - Reward for scoring
- `REWARD_LOSE_POINT = -10.0` - Penalty for losing a point
- `REWARD_NEUTRAL = 0.0` - Neutral step reward

### DQN Hyperparameters
- `LEARNING_RATE = 0.0001` - Adam optimizer learning rate
- `GAMMA = 0.99` - Discount factor for future rewards
- `EPSILON_START = 1.0` - Initial exploration rate
- `EPSILON_END = 0.01` - Minimum exploration rate
- `EPSILON_DECAY = 0.995` - Decay per episode
- `BATCH_SIZE = 64` - Training batch size
- `MEMORY_SIZE = 10000` - Replay buffer capacity
- `TARGET_UPDATE = 10` - Target network update frequency

### Network Architecture
- `INPUT_SIZE = 6` - State vector size
- `HIDDEN_SIZE_1 = 128` - First hidden layer
- `HIDDEN_SIZE_2 = 128` - Second hidden layer
- `OUTPUT_SIZE = 3` - Number of actions

## How It Works

### State Representation
The agent observes the game state as a 6-dimensional vector:
```python
[ball_x, ball_y, ball_vel_x, ball_vel_y, paddle1_y, paddle2_y]
```

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

MIT License

## Author

Created for reinforcement learning education and experimentation.
