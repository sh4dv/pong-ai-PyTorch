"""
Debug script to analyze AI agent behavior in real-time.
Shows what AI is thinking and why it makes certain decisions.
"""

import argparse
import numpy as np
import torch
from pong_env import PongEnv
from dqn_agent import DQNAgent
from config import *

def debug_agent(model_path, num_steps=1000, verbose=True):
    """
    Run agent and show detailed debug information.
    
    Args:
        model_path: Path to trained model
        num_steps: Number of steps to run
        verbose: Show detailed Q-values and state info
    """
    # Load environment and agent
    env = PongEnv()
    agent = DQNAgent()
    
    try:
        agent.load(model_path)
        print(f"✓ Loaded model from {model_path}")
    except:
        print(f"⚠ Could not load model, using random agent")
    
    print(f"Epsilon: {agent.epsilon:.3f}")
    print("-" * 80)
    
    state, _ = env.reset()
    total_reward = 0
    step = 0
    
    # Action tracking
    action_counts = {0: 0, 1: 0, 2: 0}
    action_names = {0: "NONE", 1: "UP", 2: "DOWN"}
    
    # Position tracking
    last_paddle_y = state[4]
    stuck_counter = 0
    
    terminated = False
    truncated = False
    
    print("Starting debug session...")
    print("State format: [ball_x, ball_y, ball_vx, ball_vy, paddle1_y, paddle2_y]")
    print("-" * 80)
    
    while not (terminated or truncated) and step < num_steps:
        # Get Q-values for current state
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            q_values = agent.policy_net(state_tensor).cpu().numpy()[0]
        
        # Select action
        action = agent.select_action(state, training=False)
        action_counts[action] += 1
        
        # Check if paddle is stuck
        if abs(state[4] - last_paddle_y) < 1.0:
            stuck_counter += 1
        else:
            stuck_counter = 0
        
        # Print detailed info every 50 steps or when stuck
        if verbose and (step % 50 == 0 or stuck_counter > 20):
            print(f"\nStep {step}:")
            print(f"  State: ball=({state[0]:.0f}, {state[1]:.0f}) vel=({state[2]:.1f}, {state[3]:.1f}) paddle1_y={state[4]:.0f} paddle2_y={state[5]:.0f}")
            print(f"  Q-values: NONE={q_values[0]:.3f}, UP={q_values[1]:.3f}, DOWN={q_values[2]:.3f}")
            print(f"  Action: {action_names[action]} (max Q: {action_names[np.argmax(q_values)]})")
            print(f"  Paddle Y: {state[4]:.0f} (last: {last_paddle_y:.0f})")
            
            if stuck_counter > 20:
                print(f"  ⚠ WARNING: Paddle stuck for {stuck_counter} steps!")
                
                # Check if all Q-values are similar
                q_std = np.std(q_values)
                if q_std < 0.01:
                    print(f"  ⚠ Q-values are too similar (std={q_std:.4f}) - network not learning!")
        
        # Execute action
        last_paddle_y = state[4]
        next_state, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        state = next_state
        step += 1
        
        # Render occasionally
        if step % 10 == 0:
            env.render()
    
    env.close()
    
    # Print summary
    print("\n" + "=" * 80)
    print("DEBUG SUMMARY")
    print("=" * 80)
    total_actions = sum(action_counts.values())
    print(f"Total steps: {step}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final score: {info.get('score_left', 0)}-{info.get('score_right', 0)}")
    print(f"\nAction distribution:")
    for action, count in action_counts.items():
        pct = (count / total_actions * 100) if total_actions > 0 else 0
        print(f"  {action_names[action]:6s}: {count:4d} ({pct:5.1f}%)")
    
    # Diagnosis
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    none_pct = (action_counts[0] / total_actions * 100) if total_actions > 0 else 0
    up_pct = (action_counts[1] / total_actions * 100) if total_actions > 0 else 0
    down_pct = (action_counts[2] / total_actions * 100) if total_actions > 0 else 0
    
    if none_pct > 80:
        print("⚠ AI is mostly doing NOTHING - not learning to move!")
        print("  → Increase REWARD_PROXIMITY to encourage movement")
        print("  → Decrease epsilon (currently too much exploration)")
    elif up_pct > 80 or down_pct > 80:
        print("⚠ AI is stuck moving in ONE DIRECTION!")
        print("  → Network may have converged to local minimum")
        print("  → Try increasing learning rate or epsilon")
    elif abs(up_pct - down_pct) < 10 and up_pct + down_pct > 80:
        print("⚠ AI is moving randomly UP/DOWN (equal distribution)")
        print("  → Not learning ball tracking")
        print("  → Increase REWARD_HIT_BALL")
    else:
        print("✓ Action distribution looks reasonable")
    
    if total_reward < 0:
        print(f"⚠ Negative reward ({total_reward:.1f}) - AI is losing badly!")
        print("  → Network needs more training or better rewards")
    
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Debug DQN agent behavior')
    parser.add_argument('--model', type=str, default=MODEL_SAVE_PATH,
                       help='Path to model file')
    parser.add_argument('--steps', type=int, default=500,
                       help='Number of steps to debug')
    parser.add_argument('--quiet', action='store_true',
                       help='Less verbose output')
    
    args = parser.parse_args()
    
    debug_agent(args.model, num_steps=args.steps, verbose=not args.quiet)
