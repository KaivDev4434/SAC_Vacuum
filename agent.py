import os
import time
import numpy as np
from stable_baselines3 import SAC, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from env import ContinuousVacuumEnv


class TrainingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"model_{self.n_calls}")
            self.model.save(model_path)
        return True


class VacuumAgent:
    def __init__(self, env, algo="sac", policy="MlpPolicy"):
        self.env = env
        self.algo = algo.lower()
        n_actions = env.action_space.shape[-1]

        if self.algo == "sac":
            self.model = SAC(
                policy,
                env,
                verbose=1,
                learning_rate=3e-4,
                buffer_size=1_000_000,
                learning_starts=10000,
                batch_size=256,
                ent_coef="auto",
                gamma=0.99,
            )
        elif self.algo == "td3":
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
            self.model = TD3(
                policy,
                env,
                verbose=1,
                learning_rate=3e-4,
                buffer_size=1_000_000,
                learning_starts=10000,
                batch_size=256,
                action_noise=action_noise,
                gamma=0.99,
            )
        elif self.algo == "ddpg":
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
            self.model = DDPG(
                policy,
                env,
                verbose=1,
                learning_rate=3e-4,
                buffer_size=1_000_000,
                learning_starts=10000,
                batch_size=256,
                action_noise=action_noise,
                gamma=0.99,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")

    def train(self, timesteps=1_000_000, log_dir="./logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.env = Monitor(self.env, log_dir)
        callback = TrainingCallback(check_freq=10000, save_path=log_dir)

        start_time = time.time()
        self.model.learn(
            total_timesteps=timesteps,
            callback=callback,
            log_interval=4,
            tb_log_name=f"{self.algo}",
        )
        print(f"Training completed in {time.time()-start_time:.2f}s")

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = self.model.load(path)

    def evaluate(self, episodes=10, render=False):
        total_rewards = []
        coverage_percentages = []

        for _ in range(episodes):
            obs = self.env.reset()
            done = False
            ep_reward = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                ep_reward += reward
                if render:
                    self.env.render()

            total_rewards.append(ep_reward)
            coverage_percentages.append(info["coverage"])

        print(f"Evaluation over {episodes} episodes:")
        print(
            f"Mean reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}"
        )
        print(
            f"Mean coverage: {np.mean(coverage_percentages):.2%} ± {np.std(coverage_percentages):.2%}"
        )


if __name__ == "__main__":
    # Example usage
    env = ContinuousVacuumEnv(size=10, resolution=50)

    # Initialize agent
    agent = VacuumAgent(env, algo="sac")

    # Train the agent
    agent.train(timesteps=100_000, log_dir="./sac_logs")

    # Evaluate
    agent.evaluate(episodes=5, render=True)
