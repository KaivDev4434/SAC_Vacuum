import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches


class VacuumCleanerEnv(gym.Env):
    """
    A 2D grid environment for a vacuum cleaner agent.

    State space:
    - Agent position (x, y)
    - Visited cells

    Action space:
    - 0: Move up
    - 1: Move right
    - 2: Move down
    - 3: Move left
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, grid_size=10):
        super(VacuumCleanerEnv, self).__init__()

        # Environment parameters
        self.grid_size = grid_size

        # Action space: Up, Right, Down, Left
        self.action_space = spaces.Discrete(4)

        # Observation space: Agent position and visited cells
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(
                    low=0, high=grid_size - 1, shape=(2,), dtype=np.int32
                ),
                "visited": spaces.Box(
                    low=0, high=1, shape=(grid_size, grid_size), dtype=np.int8
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
        self.agent_position = np.array(
            [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)]
        )

        # Initialize visited cells matrix (0: not visited, 1: visited)
        self.visited = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Mark initial position as visited
        self.visited[tuple(self.agent_position)] = 1

        # Step counter
        self.steps = 0

        # Return initial observation
        return self._get_observation()

    def step(self, action):
        """Take a step in the environment given an action."""
        # Increment step counter
        self.steps += 1

        # Move agent based on action
        new_position = self.agent_position.copy()

        if action == 0:  # Up
            new_position[1] = max(0, new_position[1] - 1)
        elif action == 1:  # Right
            new_position[0] = min(self.grid_size - 1, new_position[0] + 1)
        elif action == 2:  # Down
            new_position[1] = min(self.grid_size - 1, new_position[1] + 1)
        elif action == 3:  # Left
            new_position[0] = max(0, new_position[0] - 1)

        # Update agent position
        self.agent_position = new_position

        # Calculate reward
        reward = self._calculate_reward()

        # Update visited cells
        self.visited[tuple(self.agent_position)] = 1

        # Check if all cells are visited
        done = np.all(self.visited == 1)

        # Return observation, reward, done, info
        return self._get_observation(), reward, done, {}

    def _calculate_reward(self):
        """Calculate the reward for the current state."""
        # Check if the current cell was already visited
        pos = tuple(self.agent_position)

        if self.visited[pos] == 1:
            # Revisiting a cell gives a small negative reward
            return -0.1
        else:
            # Visiting a new cell gives a positive reward
            return 1.0

    def _get_observation(self):
        """Return the current observation of the environment."""
        return {"position": self.agent_position, "visited": self.visited}

    def render(self, mode="human"):
        """Render the environment."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            plt.ion()

        self.ax.clear()

        # Create a colormap for visited cells
        cmap = ListedColormap(["white", "lightblue"])

        # Plot the visited cells
        self.ax.imshow(self.visited, cmap=cmap)

        # Plot the agent
        self.ax.add_patch(
            patches.Circle(
                (self.agent_position[0], self.agent_position[1]),
                radius=0.3,
                facecolor="red",
            )
        )

        # Add grid lines
        for i in range(self.grid_size + 1):
            self.ax.axhline(i - 0.5, color="black", linestyle="-", linewidth=1)
            self.ax.axvline(i - 0.5, color="black", linestyle="-", linewidth=1)

        # Set labels and title
        self.ax.set_title(
            f"Step: {self.steps} | Visited: {np.sum(self.visited)}/{self.grid_size**2}"
        )
        self.ax.set_xticks(range(self.grid_size))
        self.ax.set_yticks(range(self.grid_size))
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)

        plt.draw()
        plt.pause(0.1)

        if mode == "rgb_array":
            # Convert the figure to an RGB array
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
    env = VacuumCleanerEnv(grid_size=10)

    # Reset the environment
    obs = env.reset()

    # Run for a few steps
    for _ in range(20):
        # Take a random action
        action = env.action_space.sample()

        # Step the environment
        obs, reward, done, _ = env.step(action)

        # Render the environment
        env.render()

        print(f"Action: {action}, Reward: {reward}")

        if done:
            print("All cells visited!")
            break

    # Close the environment
    env.close()
