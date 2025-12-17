"""
Manual training mode for Pong.
Lets a human play against the built-in simple AI while logging transitions
into the DQN replay buffer and training the model online.

Controls (left paddle):
  W - move up
  S - move down
  Q - quit session

During play, each frame is recorded as (state, human action, reward, next_state, done)
and fed into the DQN replay buffer. The agent is trained online using these
human demonstrations.
"""

import argparse
import os
import pygame

from pong_env import PongEnv
from dqn_agent import DQNAgent
from config import MODEL_SAVE_PATH, TARGET_UPDATE, BATCH_SIZE


class ManualTrainer:
    """Run human-vs-AI games and train the DQN from recorded transitions."""

    def __init__(
        self,
        model_path=MODEL_SAVE_PATH,
        save_path=MODEL_SAVE_PATH,
        checkpoint_every=1,
        max_games=None,
        batch_size=max(64, BATCH_SIZE // 4),
        train_steps_per_frame=1,
        frame_skip=1,
        skip_load=False,
        min_replay=500,
    ):
        self.env = PongEnv(render_mode="human", frame_skip=frame_skip)
        self.agent = DQNAgent()
        self.batch_size = batch_size
        self.train_steps_per_frame = max(1, train_steps_per_frame)
        self.checkpoint_every = checkpoint_every
        self.max_games = max_games
        self.save_path = save_path or model_path or MODEL_SAVE_PATH
        self.min_replay = max(1, min_replay)
        self.transitions = 0
        self.episode_reward = 0.0
        self.episode_losses = []
        self.all_losses = []
        self.games_played = 0
        self.episode_reward_breakdown = {
            "hit_ball": 0.0,
            "score_point": 0.0,
            "lose_point": 0.0,
            "miss_ball": 0.0,
            "proximity": 0.0,
            "far_penalty": 0.0,
            "neutral": 0.0,
        }

        # Load existing model unless explicitly skipped or missing
        if not skip_load and model_path and os.path.exists(model_path):
            self.agent.load(model_path)
            print(f"Loaded existing model from {model_path}")
        elif skip_load:
            print("Starting with fresh weights (load skipped by user)")
        else:
            print(f"No model found at {model_path}, starting from scratch")

    def _get_human_action(self):
        """Translate keyboard input to paddle action."""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            return 1
        if keys[pygame.K_s]:
            return 2
        return 0

    def run(self):
        """Main loop for manual training."""
        state, _ = self.env.reset()
        running = True
        action_counts = {0: 0, 1: 0, 2: 0}

        print("\nManual training started")
        print("You control the LEFT paddle (W/S). Simple AI controls the right paddle.")
        print("Press Q to quit. Window close also exits.")

        while running:
            # Handle quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if not running:
                break

            # Allow quick quit via keyboard
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                running = False
                break

            # Human action for left paddle
            action_left = self._get_human_action()

            # Track how often each action is used
            action_counts[action_left] = action_counts.get(action_left, 0) + 1

            # Step environment (simple AI handled inside env.step)
            next_state, reward, terminated, truncated, info = self.env.step(action_left)
            done = terminated or truncated

            # Aggregate reward sources for reporting
            rb = info.get("reward_breakdown")
            if isinstance(rb, dict):
                for k, v in rb.items():
                    if k in self.episode_reward_breakdown:
                        self.episode_reward_breakdown[k] += float(v)

            # Store transition and train online (only after warmup to stabilize)
                # single-env path: use env_idx=0 for n-step handling
                self.agent._store_transition_internal(state, action_left, reward, next_state, done, env_idx=0)
            self.transitions += 1
            # Wait until buffer has enough diverse samples before training
            if self.agent.memory.is_ready(max(self.batch_size, self.min_replay)):
                for _ in range(self.train_steps_per_frame):
                    loss = self.agent.train(self.batch_size)
                    if loss is not None:
                        self.episode_losses.append(loss)
                        self.all_losses.append(loss)

            # Accumulate reward and advance state
            self.episode_reward += reward
            state = next_state

            # Render the game
            self.env.render()

            # End-of-game handling
            if done:
                self.games_played += 1
                score_left = info.get("score_left", 0)
                score_right = info.get("score_right", 0)
                avg_loss = sum(self.episode_losses) / len(self.episode_losses) if self.episode_losses else 0.0
                total_actions = sum(action_counts.values()) or 1  # avoid zero-div
                pct_none = 100.0 * action_counts.get(0, 0) / total_actions
                pct_up = 100.0 * action_counts.get(1, 0) / total_actions
                pct_down = 100.0 * action_counts.get(2, 0) / total_actions
                pos_mass = sum(v for k, v in self.episode_reward_breakdown.items() if v > 0)
                neg_mass = sum(-v for k, v in self.episode_reward_breakdown.items() if v < 0)
                tot_mass = pos_mass + neg_mass or 1.0
                pct_pos = 100.0 * pos_mass / tot_mass
                pct_neg = 100.0 * neg_mass / tot_mass
                # Breakdown within positive and negative parts
                def pct_part(keys, total):
                    return {k: (100.0 * self.episode_reward_breakdown[k] / total) if total > 0 else 0.0 for k in keys}

                pos_keys = ["hit_ball", "score_point", "proximity"]
                neg_keys = ["lose_point", "miss_ball", "far_penalty", "neutral"]
                pos_split = pct_part(pos_keys, pos_mass)
                neg_split = pct_part(neg_keys, neg_mass)
                print(
                    f"Game {self.games_played}: reward={self.episode_reward:.2f} | "
                    f"avg loss={avg_loss:.4f} | score L/R={score_left}/{score_right} | "
                    f"transitions={self.transitions} | actions none/up/down={pct_none:.1f}%/{pct_up:.1f}%/{pct_down:.1f}%"
                )
                print(
                    f"  Reward mix: +{pct_pos:.1f}%/-{pct_neg:.1f}% | "
                    f"pos hit/score/prox={pos_split['hit_ball']:.1f}%/{pos_split['score_point']:.1f}%/{pos_split['proximity']:.1f}% | "
                    f"neg lose/miss/far/neutral={neg_split['lose_point']:.1f}%/{neg_split['miss_ball']:.1f}%/{neg_split['far_penalty']:.1f}%/{neg_split['neutral']:.1f}%"
                )

                # Target network update cadence mirrors standard training
                if TARGET_UPDATE > 0 and self.games_played % TARGET_UPDATE == 0:
                    self.agent.update_target_network()

                # Decay epsilon per completed game to reduce random actions over time
                self.agent.decay_epsilon()

                # Optional checkpointing
                if self.checkpoint_every and self.games_played % self.checkpoint_every == 0:
                    save_path = self._checkpoint_path(self.games_played)
                    self.agent.save(save_path)

                # Stop if max games reached
                if self.max_games and self.games_played >= self.max_games:
                    break

                # Reset for next game
                state, _ = self.env.reset()
                self.episode_reward = 0.0
                self.episode_losses.clear()
                action_counts = {0: 0, 1: 0, 2: 0}
                for k in self.episode_reward_breakdown:
                    self.episode_reward_breakdown[k] = 0.0

        self.env.close()
        print("Manual training ended")
        print(
            f"Played {self.games_played} game(s), stored {self.transitions} transitions. "
            f"Latest epsilon={self.agent.get_epsilon():.3f}"
        )

    def _checkpoint_path(self, episode_idx):
        """Return a checkpoint path; append episode index when saving multiple files."""
        if self.checkpoint_every and self.checkpoint_every > 1:
            root, ext = os.path.splitext(self.save_path)
            return f"{root}_ep{episode_idx}{ext or '.pth'}"
        return self.save_path


def main():
    parser = argparse.ArgumentParser(description="Manual training: human vs simple AI")
    parser.add_argument("--model", type=str, default=MODEL_SAVE_PATH, help="Model path to load (if exists)")
    parser.add_argument("--save-path", type=str, default=MODEL_SAVE_PATH, help="Where to save checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save every N completed games (0 to disable)")
    parser.add_argument("--games", type=int, default=None, help="Number of games to play (None = until quit)")
    parser.add_argument("--batch-size", type=int, default=max(64, BATCH_SIZE // 4), help="Batch size for online updates")
    parser.add_argument("--train-steps", type=int, default=1, help="How many gradient steps per frame")
    parser.add_argument("--frame-skip", type=int, default=1, help="Repeat each action for N frames")
    parser.add_argument("--skip-load", action="store_true", help="Start with fresh weights instead of loading a model")
    parser.add_argument("--min-replay", type=int, default=500, help="Transitions to collect before training starts")

    args = parser.parse_args()

    trainer = ManualTrainer(
        model_path=args.model,
        save_path=args.save_path,
        checkpoint_every=args.checkpoint_every,
        max_games=args.games,
        batch_size=args.batch_size,
        train_steps_per_frame=args.train_steps,
        frame_skip=args.frame_skip,
        skip_load=args.skip_load,
        min_replay=args.min_replay,
    )
    trainer.run()


if __name__ == "__main__":
    main()
