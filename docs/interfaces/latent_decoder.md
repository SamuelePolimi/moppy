---
title: LatentDecoder
created: 2024-11-02
last_updated: 2024-11-02
---

`moppy.interfaces.LatentDecoder`

The `LatentDecoder` class serves as an abstract base class for all latent decoder implementations. It defines the interface for decoding a latent variable into a normal distribution representing trajectory states.

---

### <span class="highlight-text">decode_from_latent_variable()</span>

Decodes a sampled latent variable into a normal distribution for each dimension of a `TrajectoryState`.

**Method Signature:**

`decode_from_latent_variable(self, latent_variable, time)`

**Parameters:**

- **latent_variable**: A sampled latent variable \( z \) vector.
- **time**: The time step at which the decoding is performed.

**Returns:**

- **Normal Distribution**: A normal distribution for each dimension of a `TrajectoryState`.

**Note:** This method is abstract and must be implemented by subclasses.

```python
@abstractmethod
def decode_from_latent_variable(self, latent_variable, time):
    pass
```
