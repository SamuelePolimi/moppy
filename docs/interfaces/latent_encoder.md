---
title: LatentEncoder
created: 2024-11-02
last_updated: 2024-11-02
---

`moppy.interfaces.LatentEncoder`

The `LatentEncoder` class serves as an abstract base class for all latent encoder implementations. It defines the interface for encoding a trajectory into a latent variable and sampling from it.

---

## Functions

### <span class="highlight-text">encode_to_latent_variable()</span>

Encodes a trajectory into a latent variable **`z`**, which is a vector of a fixed dimension.

**Method Signature:**

`encode_to_latent_variable(self, trajectory: Trajectory)`

**Parameters:**

- **trajectory** (`Trajectory`): The trajectory to encode.

**Returns:**

- **Tuple**: A tuple containing the sampled latent variable **`z`** represented as **`mu`** (mean) and **`sigma`** (standard deviation).

**Note:** This method is abstract and must be implemented by subclasses.

```python
@abstractmethod
def encode_to_latent_variable(self, trajectory: Trajectory):
    pass
```

---

### <span class="highlight-text">sample_latent_variable()</span>

Samples a latent variable **`z`** from a normal distribution specified by **`mu`** and **`sigma`**.

**Method Signature:**

`sample_latent_variable(self, mu, sigma, percentage_of_standard_deviation=None)`

**Parameters:**

- **mu**: The mean of the normal distribution.
- **sigma**: The standard deviation of the normal distribution.
- **percentage_of_standard_deviation** (optional): The percentage of the standard deviation to sample from.

**Returns:**

- **Tensor**: The sampled latent variable **`z`**.

**Note:** This method is abstract and must be implemented by subclasses.

```python
@abstractmethod
def sample_latent_variable(self, mu, sigma, percentage_of_standard_deviation=None):
    pass
```

---
