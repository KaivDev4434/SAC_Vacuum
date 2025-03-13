import numpy as np
import torch
from continuous_vacuum_env import ContinuousVacuumCleanerEnv
from ddpg_agent import DDPGAgent
import time


# Function to preprocess observation
def preprocess_observation(observation):
    position = observation["position"]
    coverage = observation["coverage"]
    return np.concatenate([position, coverage])


def run_agent(model_path, render=True, episodes=1):
    # Create environment
    env = ContinuousVacuumCleanerEnv(
        size=10.0, resolution=50, coverage_radius=0.5, max_steps=1000
    )

    # Get state and action dimensions
    obs = env.reset()
    state_dim = len(preprocess_observation(obs))
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Create agent
    agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, max_action=max_action)

    # Load trained model
    agent.load(model_path)

    for episode in range(episodes):
        # Reset environment
        obs = env.reset()
        state = preprocess_observation(obs)
        done = False
        episode_reward = 0
        step = 0

        print(f"Starting episode {episode+1}...")

        while not done:
            # Select action (no noise for demonstration)
            action = agent.select_action(state, noise=0.0)

            # Execute action
            next_obs, reward, done, info = env.step(action)
            next_state = preprocess_observation(next_obs)

            state = next_state
            episode_reward += reward
            step += 1

            if render:
                env.render()
                time.sleep(0.01)  # Slow down visualization

            # Print progress
            if step % 50 == 0:
                print(f"Step: {step}, Coverage: {info['coverage_percentage']:.2%}")

            if done:
                print(
                    f"Episode {episode+1} finished - "
                    f"Reward: {episode_reward:.2f}, "
                    f"Coverage: {info['coverage_percentage']:.2%}, "
                    f"Steps: {step}"
                )
                break

    # Close environment
    env.close()


if __name__ == "__main__":
    # Specify the path to your trained model
    model_path = "models/vacuum_agent_episode_500"  # Update with actual path

    # Run the agent
    run_agent(model_path, render=True, episodes=3)
