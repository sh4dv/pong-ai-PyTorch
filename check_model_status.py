"""
Check the current status of a trained model.
Shows epsilon, memory size, and prediction quality.
"""

import torch
import numpy as np
from dqn_agent import DQNAgent
from pong_env import PongEnv
from config import *
import os


def check_model_status(model_path=MODEL_SAVE_PATH):
    """
    Check and display model status.
    
    Args:
        model_path (str): Path to saved model
    """
    print("=" * 70)
    print("MODEL STATUS CHECK")
    print("=" * 70)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found at: {model_path}")
        print("   Train a model first with: python train.py")
        return
    
    # Load model
    print(f"\n📂 Loading model from: {model_path}")
    agent = DQNAgent()
    checkpoint = torch.load(model_path, map_location=agent.device)
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    agent.epsilon = checkpoint.get('epsilon', agent.epsilon)
    
    # Get training info
    episode = checkpoint.get('episode', 'Unknown')
    total_steps = checkpoint.get('total_steps', 'Unknown')
    
    print(f"✅ Model loaded successfully!")
    print()
    
    # Display training statistics
    print("📊 TRAINING STATISTICS")
    print("-" * 70)
    print(f"Episode:              {episode}")
    print(f"Total Steps:          {total_steps}")
    print(f"Epsilon (ε):          {agent.epsilon:.4f}")
    print(f"Device:               {agent.device}")
    print()
    
    # Interpret epsilon
    print("🎯 EXPLORATION vs EXPLOITATION")
    print("-" * 70)
    exploration_rate = agent.epsilon * 100
    exploitation_rate = (1 - agent.epsilon) * 100
    
    print(f"Random Actions:       {exploration_rate:.1f}%")
    print(f"Learned Actions:      {exploitation_rate:.1f}%")
    
    if agent.epsilon > 0.5:
        print("📍 Status: HEAVY EXPLORATION - Still learning basics")
    elif agent.epsilon > 0.2:
        print("📍 Status: BALANCED - Mix of exploration and learned behavior")
    elif agent.epsilon > 0.05:
        print("📍 Status: MOSTLY EXPLOITING - Using learned strategy")
    else:
        print("📍 Status: FULL EXPLOITATION - Relying on learned policy")
    print()
    
    # Memory/Experience info
    print("💾 EXPERIENCE REPLAY BUFFER")
    print("-" * 70)
    print(f"Buffer Capacity:      {MEMORY_SIZE:,}")
    if total_steps != 'Unknown':
        if isinstance(total_steps, int):
            filled = min(total_steps, MEMORY_SIZE)
            fill_percent = (filled / MEMORY_SIZE) * 100
            print(f"Experiences Stored:   {filled:,} ({fill_percent:.1f}% full)")
            print(f"Total Interactions:   {total_steps:,}")
        else:
            print(f"Experiences Stored:   Unknown")
    print()
    
    # Test model performance
    print("🎮 QUICK PERFORMANCE TEST (5 games)")
    print("-" * 70)
    
    env = PongEnv()
    wins = 0
    total_reward = 0
    total_length = 0
    
    for i in range(5):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Use greedy action (no exploration)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_values = agent.policy_net(state_tensor)
                action = q_values.argmax().item()
            
            state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            episode_reward += reward
            steps += 1
        
        total_reward += episode_reward
        total_length += steps
        
        # Check if won (positive reward usually means winning)
        if episode_reward > 0:
            wins += 1
        
        print(f"  Game {i+1}: Reward={episode_reward:+6.1f}, Steps={steps}")
    
    avg_reward = total_reward / 5
    avg_length = total_length / 5
    win_rate = (wins / 5) * 100
    
    print()
    print(f"Average Reward:       {avg_reward:+.1f}")
    print(f"Average Game Length:  {avg_length:.0f} steps")
    print(f"Win Rate:             {win_rate:.0f}%")
    print()
    
    # Performance interpretation
    print("📈 PERFORMANCE LEVEL")
    print("-" * 70)
    if avg_reward < -30:
        print("❌ BEGINNER - Losing most games, needs more training")
    elif avg_reward < 0:
        print("⚠️  LEARNING - Starting to compete but still losing")
    elif avg_reward < 30:
        print("✅ INTERMEDIATE - Winning some games")
    elif avg_reward < 60:
        print("🔥 GOOD - Winning most games")
    else:
        print("🏆 EXPERT - Dominating the game!")
    
    print()
    print("=" * 70)
    print(f"💡 TIP: Current epsilon {agent.epsilon:.4f} means {exploitation_rate:.1f}% of")
    print(f"    actions are based on learned strategy, {exploration_rate:.1f}% are random.")
    print("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Check model training status')
    parser.add_argument('--model', type=str, default=MODEL_SAVE_PATH,
                       help='Path to model checkpoint')
    
    args = parser.parse_args()
    check_model_status(args.model)


if __name__ == "__main__":
    main()
