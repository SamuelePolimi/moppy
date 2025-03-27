---
title: DeepProMP Utility Functions
created: 2024-11-02
last_updated: 2024-11-02
---

### set_seed()

The `set_seed` function sets the random seed for Python, NumPy, and PyTorch, enabling reproducibility across experiments by controlling the randomness in operations.

**Method Signature**

`set_seed(seed: int) -> None`

**Parameters**

- **seed** (`int`): The seed value used to initialize the random generators.

**Example**

```python
# Set a specific seed to ensure reproducible results
set_seed(42)
```

---

### plot_trajectories()

The `plot_trajectories` function generates a plot for a list of labeled trajectories and saves it to a file. This function is designed to work with `SinusState` trajectories.

**Method Signature**

`plot_trajectories(labeled_trajectories: List[Tuple[Trajectory[SinusState], str]], file_name: str, plot_title: str) -> None`

**Parameters**

- **labeled_trajectories** (`List[Tuple[Trajectory, str]]`): A list of tuples, each containing a trajectory and its label.
- **file_name** (`str`): The name of the file where the plot will be saved.
- **plot_title** (`str`): Title of the plot.

**Example**

```python
# Plot labeled trajectories and save the image
plot_trajectories([(traj1, 'Trajectory 1'), (traj2, 'Trajectory 2')], 'trajectories_plot.png', 'Example Trajectories')
```

---

### generate_sin_trajectory()

The `generate_sin_trajectory` function creates a sinusoidal trajectory based on the specified amplitude, frequency, and number of points, returning it as a `Trajectory` object.

**Method Signature**

`generate_sin_trajectory(amplitude: float | int, frequency: float | int, num_steps: int = 100) -> Trajectory`

**Parameters**

- **amplitude** (`float` | `int`): The amplitude of the sinusoidal trajectory.
- **frequency** (`float` | `int`): The frequency of the sinusoidal trajectory.
- **num_steps** (`int`, optional): The number of points in the trajectory. Defaults to `100`.

**Returns**

- **Trajectory**: A sinusoidal trajectory with `num_steps` points.

**Example**

```python
# Generate a sinusoidal trajectory with amplitude 5 and frequency 1
trajectory = generate_sin_trajectory(amplitude=5, frequency=1)
```

---

### generate_sin_trajectory_set()

The `generate_sin_trajectory_set` function generates a set of sinusoidal trajectories with randomly selected amplitudes and frequencies from specified ranges.

**Method Signature**

`generate_sin_trajectory_set(n: int, amplitude_range: Tuple[int, int] = (1, 10), frequency_range: Tuple[int, int] = (1, 1)) -> List[Trajectory]`

**Parameters**

- **n** (`int`): The number of trajectories to generate.
- **amplitude_range** (`Tuple[int, int]`, optional): Range for amplitude values. Defaults to `(1, 10)`.
- **frequency_range** (`Tuple[int, int]`, optional): Range for frequency values. Defaults to `(1, 1)`.

**Returns**

- **List[Trajectory]**: A list of generated sinusoidal trajectories.

**Example**

```python
# Generate 5 sinusoidal trajectories with random amplitude and frequency within specified ranges
trajectories = generate_sin_trajectory_set(5, amplitude_range=(1, 5), frequency_range=(1, 2))
```

---

### generate_sin_trajectory_set_labeled()

The `generate_sin_trajectory_set_labeled` function generates a labeled set of sinusoidal trajectories with random amplitudes and frequencies within specified ranges.

**Method Signature**

`generate_sin_trajectory_set_labeled(n: int, amplitude_range: Tuple[int, int] = (1, 10), frequency_range: Tuple[int, int] = (1, 1)) -> List[dict]`

**Parameters**

- **n** (`int`): The number of trajectories to generate.
- **amplitude_range** (`Tuple[int, int]`, optional): Range for amplitude values. Defaults to `(1, 10)`.
- **frequency_range** (`Tuple[int, int]`, optional): Range for frequency values. Defaults to `(1, 1)`.

**Returns**

- **List[dict]**: A list of dictionaries, each containing a trajectory, amplitude, and frequency. Example: `{'traj': traj, 'amplitude': 5, 'frequency': 1}`.

**Example**

```python
# Generate 5 labeled sinusoidal trajectories with random amplitude and frequency within the specified ranges
labeled_trajectories = generate_sin_trajectory_set_labeled(5, amplitude_range=(1, 5), frequency_range=(1, 2))
```
