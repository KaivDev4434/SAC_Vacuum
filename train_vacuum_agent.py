import numpy as np
import matplotlib.pyplot as plt
import torch
from continuous_vacuum_env import ContinuousVacuumCleanerEnv
from ddpg_agent import DDPGAgent


# Function to preprocess observation from dictionary to flat array
def preprocess_observation(observation):
    position = observation["position"]
    coverage = observation["coverage"]
    return np.concatenate([position, coverage])


def train(
    env,
    agent,
    num_episodes=1000,
    max_steps=1000,
    batch_size=64,
    gamma=0.99,
    tau=0.005,
    eval_freq=10,
    print_freq=10,
    save_freq=100,
):

    # Metrics for plotting
    episode_rewards = []
    episode_coverages = []

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        state = preprocess_observation(obs)
        episode_reward = 0

        for step in range(max_steps):
            # Select action
            noise_scale = max(0.1, 1.0 - episode / 500)  # Reduce noise over time
            action = agent.select_action(state, noise=noise_scale)

            # Execute action
            next_obs, reward, done, info = env.step(action)
            next_state = preprocess_observation(next_obs)

            # Store in replay buffer
            agent.replay_buffer.push(
                state, action, next_state, np.array([reward]), np.array([float(done)])
            )

            # Update agent
            agent.update(gamma=gamma, tau=tau)

            state = next_state
            episode_reward += reward

            # Render occasionally
            if episode % eval_freq == 0:
                env.render()

            if done:
                break

        # Track metrics
        episode_rewards.append(episode_reward)
        episode_coverages.append(info["coverage_percentage"])

        # Print progress
        if episode % print_freq == 0:
            print(
                f"Episode {episode}/{num_episodes} - "
                f"Reward: {episode_reward:.2f}, "
                f"Coverage: {info['coverage_percentage']:.2%}, "
                f"Steps: {step+1}"
            )

        # Save model periodically
        if episode % save_freq == 0:
            agent.save(f"models/vacuum_agent_episode_{episode}")

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.subplot(1, 2, 2)
    plt.plot(episode_coverages)
    plt.title("Coverage Percentage")
    plt.xlabel("Episode")
    plt.ylabel("Coverage %")

    plt.tight_layout()
    plt.savefig("training_results.png")
    plt.show()

    return episode_rewards, episode_coverages


def evaluate(env, agent, num_episodes=5, render=True):
    """Evaluate the trained agent"""
    for episode in range(num_episodes):
        obs = env.reset()
        state = preprocess_observation(obs)
        episode_reward = 0
        done = False
        step = 0

        while not done:
            # Select action (no noise for evaluation)
            action = agent.select_action(state, noise=0.0)

            # Execute action
            next_obs, reward, done, info = env.step(action)
            next_state = preprocess_observation(next_obs)

            state = next_state
            episode_reward += reward
            step += 1

            if render:
                env.render()

            if done:
                print(
                    f"Evaluation Episode {episode+1} - "
                    f"Reward: {episode_reward:.2f}, "
                    f"Coverage: {info['coverage_percentage']:.2%}, "
                    f"Steps: {step}"
                )
                break


if __name__ == "__main__":
    # Create environment
    env = ContinuousVacuumCleanerEnv(
        size=10.0, resolution=50, coverage_radius=0.5, max_steps=1000
    )

    # Get state and action dimensions
    obs = env.reset()
    state_dim = len(preprocess_observation(obs))
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Max action: {max_action}")

    # Create agent
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        buffer_size=100000,
        batch_size=64,
    )

    # Create models directory if it doesn't exist
    import os

    if not os.path.exists("models"):
        os.makedirs("models")

    # Train agent
    train(
        env=env,
        agent=agent,
        num_episodes=500,
        max_steps=1000,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        eval_freq=50,  # Render every 50 episodes
        print_freq=10,
        save_freq=100,
    )

    # Evaluate agent
    print("\nFinal Evaluation:")
    evaluate(env, agent, num_episodes=3, render=True)

    # Close environment
    env.close()
