# Quick Usage Guide - Pong AI DQN

## Prerequisites

Activate the virtual environment before running any commands:

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

---

## 1. Training the AI

**Quick test (short):**
```bash
python train.py --headless --episodes 100
```

**Standard training (recommended for short experiments):**
```bash
python train.py --headless --episodes 1000
```

**Long training (recommended for better performance — defaults are larger):**
```bash
python train.py --headless --episodes 3000
```

**Note about vectorized training:** `--episodes` is interpreted as the total number of episodes across all environments. For example `--episodes 1000 --num-envs 4` runs 250 iterations × 4 envs = 1000 total episodes.

**Training Parameters:**
- `--episodes N` - Number of training episodes (default: 3000)
- `--headless` - Train without rendering (faster)
- `--save-every N` - Save checkpoint every N episodes (default: 100)
- `--log-every N` - Print statistics every N episodes (default: 10)
- `--resume PATH` - Resume from checkpoint

**Example Output:**
```
Episode 100/3000 | Reward: -3.20 | Length: 412 | Loss: 0.0987 | 
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

To verify your installation, run a quick headless training session:
```bash
python train.py --headless --episodes 10
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

### Evaluate and Debug
- Evaluate a trained model:
```bash
python evaluate.py --model models/dqn_pong.pth --episodes 100
```
- Quick model check:
```bash
python check_model_status.py --model models/dqn_pong.pth
```
- Manual (human-in-the-loop) training:
```bash
python manual_train.py --games 10
```
