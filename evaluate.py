"""
Evaluate a trained DQN model on the Pong environment.

Defaults to `models/dqn_pong.pth` and runs greedy evaluation (no exploration).
Prints mean reward, win-rate, mean episode length, and aggregated reward breakdown.
"""

import argparse
import statistics
import time
from collections import defaultdict

from pong_env import PongEnv
from dqn_agent import DQNAgent
from config import MODEL_SAVE_PATH


def evaluate(model_path=MODEL_SAVE_PATH, episodes=100, render=False, frame_skip=1, render_delay=0.0, verbose=False):
    env = PongEnv(render_mode="human" if render else "headless", frame_skip=frame_skip)
    agent = DQNAgent()

    if not agent.load(model_path):
        print(f"Failed to load model from {model_path}")
        env.close()
        return None

    rewards = []
    lengths = []
    wins = 0
    breakdown_totals = defaultdict(float)

    print(f"Starting evaluation: model={model_path} | episodes={episodes} | render={render}")

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0

        while not done:
            # Greedy action (evaluation)
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            ep_length += 1

            # accumulate reward breakdown if provided
            rb = info.get("reward_breakdown")
            if isinstance(rb, dict):
                for k, v in rb.items():
                    breakdown_totals[k] += float(v)

            state = next_state

            if render:
                env.render()
                if render_delay > 0:
                    time.sleep(render_delay)

        # episode ended
        rewards.append(ep_reward)
        lengths.append(ep_length)

        score_left = info.get("score_left", 0)
        score_right = info.get("score_right", 0)
        if score_left > score_right:
            wins += 1

        if verbose:
            print(f"Episode {ep}/{episodes}: reward={ep_reward:.2f} | length={ep_length} | score L/R={score_left}/{score_right}")

    env.close()

    mean_reward = statistics.mean(rewards) if rewards else 0.0
    stdev_reward = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    mean_length = statistics.mean(lengths) if lengths else 0.0
    win_rate = 100.0 * wins / episodes if episodes > 0 else 0.0

    print("\n=== Evaluation Summary ===")
    print(f"Episodes: {episodes}")
    print(f"Mean reward: {mean_reward:.2f} ± {stdev_reward:.2f}")
    print(f"Mean length: {mean_length:.1f} steps")
    print(f"Win rate (score_left > score_right): {win_rate:.1f}% ({wins}/{episodes})")

    if breakdown_totals:
        tot = sum(abs(v) for v in breakdown_totals.values()) or 1.0
        print("Reward breakdown (absolute shares):")
        for k, v in breakdown_totals.items():
            print(f"  {k}: {v:.2f} ({100.0 * abs(v)/tot:.1f}%)")

    return {
        'episodes': episodes,
        'mean_reward': mean_reward,
        'stdev_reward': stdev_reward,
        'mean_length': mean_length,
        'win_rate': win_rate,
        'wins': wins,
        'breakdown': dict(breakdown_totals),
        'rewards': rewards,
        'lengths': lengths,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN model on Pong")
    parser.add_argument("--model", type=str, default=MODEL_SAVE_PATH, help="Path to model file (default: models/dqn_pong.pth)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run")
    parser.add_argument("--render", action="store_true", help="Render during evaluation (slower)" )
    parser.add_argument("--frame-skip", type=int, default=1, help="Frame-skip to use for environment during evaluation")
    parser.add_argument("--render-delay", type=float, default=0.0, help="Seconds to wait between frames when rendering")
    parser.add_argument("--verbose", action="store_true", help="Print per-episode details")
    args = parser.parse_args()

    evaluate(model_path=args.model, episodes=args.episodes, render=args.render, frame_skip=args.frame_skip, render_delay=args.render_delay, verbose=args.verbose)
