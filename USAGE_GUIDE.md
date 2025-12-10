# Quick Usage Guide - Pong AI DQN

## Prerequisites

Activate the virtual environment before running any commands:
```bash
source .venv/bin/activate
```

---

## 1. Training the AI Agent

### Fast Headless Training (Recommended for speed)
Train without rendering for maximum speed:
```bash
python train.py --headless --episodes 500
```

### Standard Training
Train for 1000 episodes (better results):
```bash
python train.py --headless --episodes 1000
```

### Long Training (Best Results)
Train for 2000+ episodes for optimal performance:
```bash
python train.py --headless --episodes 2000
```

### Custom Training
Adjust all parameters:
```bash
python train.py --episodes 1500 --save-every 100 --log-every 10 --headless
```

**Training Parameters:**
- `--episodes N` - Number of training episodes (default: 1000)
- `--headless` - Train without rendering (faster)
- `--save-every N` - Save checkpoint every N episodes (default: 100)
- `--log-every N` - Print statistics every N episodes (default: 10)
- `--resume PATH` - Resume from checkpoint

**Example Output:**
```
Episode 100/1000 | Reward: -3.20 | Length: 412 | Loss: 0.0987 | 
Score L/R: 3.5/10.0 | Epsilon: 0.904 | Speed: 13.21 eps/s
```

---

## 2. Watching AI Training Progress (Rendering)

### Render Every 100 Episodes
See the AI play every 100 episodes during training:
```bash
python train.py --render-every 100 --episodes 1000
```

### Render Every 50 Episodes
More frequent visualization:
```bash
python train.py --render-every 50 --episodes 1000
```

### Debug Mode with Frequent Rendering
Render every 10 episodes (slower, for debugging):
```bash
python train.py --render-every 10 --episodes 100
```

**Note:** Rendering slows down training. Use `--headless` for fastest training, then use play mode to watch the trained agent.

---

## 3. Playing with Trained AI

### Watch AI vs Simple AI
Default mode - watch your trained AI play:
```bash
python play.py
```

### Play Against the AI Yourself
You control the right paddle (arrow keys):
```bash
python play.py --opponent human
```

**Controls (when playing as human):**
- ↑ Arrow Up - Move paddle up
- ↓ Arrow Down - Move paddle down
- R - Restart game
- Q - Quit

### Use Specific Model Checkpoint
Play with a specific saved model:
```bash
python play.py --model models/checkpoints/checkpoint_ep500.pth
```

### Limited Number of Games
Watch AI play exactly 5 games:
```bash
python play.py --games 5
```

---

## Quick Start with Preset Profiles

Use the quickstart script for easy access:

```bash
# Check installation
python quickstart.py --check

# Fast training (500 episodes)
python quickstart.py --train fast

# Standard training (1000 episodes)
python quickstart.py --train standard

# Long training (2000 episodes)
python quickstart.py --train long

# Debug with rendering (100 episodes)
python quickstart.py --train debug

# Play with trained model
python quickstart.py --play

# Human vs human mode
python quickstart.py --human
```

---

## Complete Training & Playing Workflow

### Step 1: Train the AI
```bash
# Fast training for testing (5-10 minutes)
python train.py --headless --episodes 500

# OR longer for better results (15-30 minutes)
python train.py --headless --episodes 1000
```

### Step 2: Watch the AI Play
```bash
python play.py
```

### Step 3: Play Against the AI
```bash
python play.py --opponent human
```

---

## Saved Models Location

- **Final model:** `models/dqn_pong.pth`
- **Checkpoints:** `models/checkpoints/checkpoint_ep{N}.pth`

Resume training from checkpoint:
```bash
python train.py --resume models/checkpoints/checkpoint_ep500.pth --episodes 1000
```

---

## Troubleshooting

**Training is slow:**
- Use `--headless` flag
- Increase `--log-every` to reduce console output
- Check that MPS (Apple Silicon) is being used - you should see "Using MPS" in output

**AI not improving:**
- Train for more episodes (try 1500-2000)
- Check epsilon value - should decay from 1.0 to ~0.01
- Adjust rewards in `config.py`

**Can't find model:**
```bash
# Check if model exists
ls -la models/
ls -la models/checkpoints/

# Train first if no model exists
python train.py --headless --episodes 500
```

**Want to test setup:**
```bash
python test_setup.py
```

---

## Performance Tips

**Fastest Training:**
```bash
python train.py --headless --episodes 1000 --log-every 50
```

**Best Visual Feedback:**
```bash
python train.py --render-every 100 --episodes 1000 --log-every 10
```

**Optimal Balance:**
```bash
python train.py --headless --episodes 1500 --save-every 100 --log-every 20
```

---

## Advanced Usage

### Modify Hyperparameters
Edit `config.py` to adjust:
- Reward values (`REWARD_HIT_BALL`, `REWARD_SCORE_POINT`, etc.)
- Learning rate (`LEARNING_RATE`)
- Network architecture (`HIDDEN_SIZE_1`, `HIDDEN_SIZE_2`)
- Epsilon decay (`EPSILON_DECAY`)

### Monitor Training Progress
Watch the metrics during training:
- **Reward:** Higher is better (should increase over time)
- **Score L/R:** AI score vs opponent (left should increase)
- **Epsilon:** Exploration rate (decreases over time)
- **Loss:** Training loss (should decrease and stabilize)

### Save Training Results
Redirect output to file:
```bash
python train.py --headless --episodes 1000 | tee training_log.txt
```
