import math

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from matplotlib.patches import Circle, Rectangle


class ContinuousVacuumCleanerEnv(gym.Env):
    """
    A continuous 2D environment for a vacuum cleaner agent with continuous actions.

    State space:
    - Agent position (x, y)
    - Agent orientation (theta)
    - Coverage map

    Action space:
    - Linear velocity (forward/backward movement)
    - Angular velocity (turning)
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, size=10.0, resolution=50, coverage_radius=0.5, max_steps=1000):
        super(ContinuousVacuumCleanerEnv, self).__init__()

        # Environment parameters
        self.size = size  # Size of the square environment
        self.resolution = resolution  # Resolution of the coverage grid
        self.coverage_radius = coverage_radius  # Radius within which the agent cleans
        self.max_steps = max_steps  # Maximum steps before episode terminates
        self.cell_size = size / resolution  # Size of each cell in the coverage grid

        # Agent parameters
        self.max_linear_velocity = 1.0  # Maximum linear velocity
        self.max_angular_velocity = np.pi  # Maximum angular velocity (radians/s)
        self.dt = 0.1  # Time step for simulation

        # Action space: [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(
            low=np.array([-self.max_linear_velocity, -self.max_angular_velocity]),
            high=np.array([self.max_linear_velocity, self.max_angular_velocity]),
            dtype=np.float32,
        )

        # Observation space: [x, y, theta, flattened_coverage_map]
        # We flatten the coverage map for simplicity
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(
                    low=np.array([0, 0, -np.pi]),
                    high=np.array([size, size, np.pi]),
                    dtype=np.float32,
                ),
                "coverage": spaces.Box(
                    low=0, high=1, shape=(resolution * resolution,), dtype=np.float32
                ),
            }
        )

        # Initialize state
        self.reset()

        # For visualization
        self.fig = None
        self.ax = None

    def reset(self):
        """Reset the environment to initial state."""
        # Agent starts at a random position
        self.agent_position = np.array([0.0, 0.0], dtype=np.float32)

        # Agent starts with a random orientation
        self.agent_orientation = np.random.uniform(-np.pi, np.pi)

        # Initialize coverage grid (0: not covered, 1: covered)
        self.coverage_grid = np.zeros(
            (self.resolution, self.resolution), dtype=np.float32
        )

        # Update coverage based on initial position
        self._update_coverage()

        # Step counter
        self.steps = 0

        # Coverage percentage
        self.coverage_percentage = 0.0

        # Return initial observation
        return self._get_observation()

    def step(self, action):
        """Take a step in the environment given an action."""
        # Clip action to ensure it's within bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Extract actions
        linear_velocity, angular_velocity = action

        # Update orientation
        self.agent_orientation += angular_velocity * self.dt
        self.agent_orientation = (
            (self.agent_orientation + np.pi) % (2 * np.pi)
        ) - np.pi  # Normalize to [-π, π]

        # Calculate movement vector
        delta_x = linear_velocity * math.cos(self.agent_orientation) * self.dt
        delta_y = linear_velocity * math.sin(self.agent_orientation) * self.dt

        # Update position
        new_position = self.agent_position + np.array([delta_x, delta_y])

        # Ensure agent stays within bounds
        new_position = np.clip(new_position, [0, 0], [self.size, self.size])

        # Set new position
        self.agent_position = new_position

        # Update coverage
        self._update_coverage()

        # Calculate reward
        reward = self._calculate_reward()

        # Increment step counter
        self.steps += 1

        # Check if done
        done = (self.coverage_percentage >= 0.95) or (self.steps >= self.max_steps)

        # Additional info
        info = {"coverage_percentage": self.coverage_percentage, "steps": self.steps}

        # Return observation, reward, done, info
        return self._get_observation(), reward, done, info

    def _update_coverage(self):
        """Update the coverage grid based on the agent's position."""
        # Get agent position in grid coordinates
        agent_x, agent_y = self.agent_position

        # Calculate grid cells within coverage radius
        coverage_radius_cells = self.coverage_radius / self.cell_size

        # Get bounds for the affected area in grid coordinates
        min_x = max(0, int((agent_x - self.coverage_radius) / self.cell_size))
        max_x = min(
            self.resolution - 1, int((agent_x + self.coverage_radius) / self.cell_size)
        )
        min_y = max(0, int((agent_y - self.coverage_radius) / self.cell_size))
        max_y = min(
            self.resolution - 1, int((agent_y + self.coverage_radius) / self.cell_size)
        )

        # Count newly covered cells
        newly_covered = 0
        total_cells = (max_x - min_x + 1) * (max_y - min_y + 1)

        # Check each cell in the affected area
        for i in range(min_x, max_x + 1):
            for j in range(min_y, max_y + 1):
                # Calculate the center of this cell
                cell_center_x = (i + 0.5) * self.cell_size
                cell_center_y = (j + 0.5) * self.cell_size

                # Check if this cell is within the coverage radius
                if (
                    (cell_center_x - agent_x) ** 2 + (cell_center_y - agent_y) ** 2
                ) <= self.coverage_radius**2:
                    # Mark as covered if it wasn't already
                    if self.coverage_grid[j, i] == 0:
                        newly_covered += 1
                    self.coverage_grid[j, i] = 1

        # Update coverage percentage
        self.coverage_percentage = np.sum(self.coverage_grid) / (
            self.resolution * self.resolution
        )

        # Return the number of newly covered cells and total cells in the area
        return newly_covered, total_cells

    def _calculate_reward(self):
        """Calculate the reward for the current state."""
        # Reward is based on the coverage percentage
        reward = 0

        # Reward for coverage (newly covered cells from _update_coverage)
        newly_covered, _ = self._update_coverage()
        reward += newly_covered * 1.0

        # Small time penalty to encourage efficiency
        reward -= 0.01

        # Bonus for completing the task
        if self.coverage_percentage >= 0.95:
            reward += 100

        return reward

    def _get_observation(self):
        """Return the current observation of the environment."""
        return {
            "position": np.array(
                [
                    self.agent_position[0],
                    self.agent_position[1],
                    self.agent_orientation,
                ],
                dtype=np.float32,
            ),
            "coverage": self.coverage_grid.flatten(),
        }

    def render(self, mode="human"):
        """Render the environment."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            plt.ion()

        self.ax.clear()

        # Plot the coverage grid
        coverage_img = self.ax.imshow(
            self.coverage_grid,
            extent=[0, self.size, 0, self.size],
            origin="lower",
            cmap="Blues",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        # Plot the agent as a circle with a line indicating orientation
        agent_circle = Circle(
            self.agent_position,
            radius=self.coverage_radius,
            facecolor="red",
            alpha=0.5,
            edgecolor="black",
        )
        self.ax.add_patch(agent_circle)

        # Draw the orientation line
        line_length = self.coverage_radius * 1.5
        end_x = self.agent_position[0] + line_length * math.cos(self.agent_orientation)
        end_y = self.agent_position[1] + line_length * math.sin(self.agent_orientation)
        self.ax.plot(
            [self.agent_position[0], end_x], [self.agent_position[1], end_y], "k-"
        )

        # Draw the environment boundaries
        boundary = Rectangle(
            (0, 0),
            self.size,
            self.size,
            edgecolor="black",
            facecolor="none",
            linewidth=2,
        )
        self.ax.add_patch(boundary)

        # Set labels and title
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)
        self.ax.set_title(
            f"Step: {self.steps} | Coverage: {self.coverage_percentage:.2%}"
        )

        # Add colorbar
        if not hasattr(self, "colorbar"):
            self.colorbar = plt.colorbar(coverage_img, ax=self.ax)
            self.colorbar.set_label("Coverage")

        plt.draw()
        plt.pause(0.01)

        if mode == "rgb_array":
            self.fig.canvas.draw()
            img = np.array(self.fig.canvas.renderer.buffer_rgba())
            return img

    def close(self):
        """Close the environment."""
        if self.fig is not None:
            plt.close(self.fig)
            plt.ioff()
            self.fig = None
            self.ax = None


# Example usage
if __name__ == "__main__":
    # Create the environment
    env = ContinuousVacuumCleanerEnv(size=10.0, resolution=50, coverage_radius=0.5)

    # Reset the environment
    obs = env.reset()

    # Run for a few steps with a simple heuristic policy
    done = False
    total_reward = 0

    while not done:
        # Simple heuristic: move forward with small random turns
        action = np.array(
            [0.8, np.random.uniform(-0.5, 0.5)]  # Forward velocity  # Random turning
        )

        # Step the environment
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Render the environment
        env.render()

        print(
            f"Step: {info['steps']}, Reward: {reward:.2f}, Coverage: {info['coverage_percentage']:.2%}"
        )

        if info["steps"] >= 2000:  # Limit the example run
            break

    print(f"Total reward: {total_reward:.2f}")

    # Close the environment
    env.close()
