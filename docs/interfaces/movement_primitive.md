---
title: Movement Primitive
created: 2024-11-02
last_updated: 2024-11-02
---

`moppy.interfaces.movement_primitive.MovementPrimitive(self, name: str, encoder: LatentEncoder, decoder: LatentDecoder)`

The `MovementPrimitive` class serves as a base class for movement primitives, encapsulating the functionality of an encoder and decoder architecture. It provides a way to obtain a distribution of trajectory states based on a given context trajectory and a normalized time step.

### Parameters

- **name** (`str`): The name of the movement primitive.
- **encoder** (`LatentEncoder`): The encoder used to encode the context trajectory into a latent variable.
- **decoder** (`LatentDecoder`): The decoder used to decode the latent variable back into trajectory states.

---

### <span class="highlight-text">get_state_distribution_at()</span>

Obtains the state distribution for a given context trajectory at a specified normalized time step.

**Method Signature:**

`get_state_distribution_at(self, context_trajectory: Trajectory, time: float)`

**Parameters:**

- **context_trajectory** (`Trajectory`): A context trajectory that serves as the input for encoding.
- **time** (`float`): A normalized time step between 0.0 and 1.0.

**Returns:**

- **Normal Distribution**: A normal distribution for each dimension of a `TrajectoryState`.

**Raises:**

- **ValueError**: If the context trajectory is `None`.

---

### <span class="highlight-text">train()</span>

An abstract method intended for training the movement primitive with the given trajectories.

**Method Signature:**

`train(self, trajectories: List[Trajectory])`

**Parameters:**

- **trajectories** (`List[Trajectory]`): A list of trajectories for training.

**Note:** This method is abstract and should be implemented by subclasses.

```python
@abstractmethod
def train(self, trajectories: List[Trajectory]):
    pass
```
