---
title: Classes
created: 2024-10-01
last_updated: 2024-10-12
---

## DeepProMP

`moppy.deep_promp.deep_promp.DeepProMP(name: str, encoder: EncoderDeepProMP, decoder: DecoderDeepProMP, save_path: str = './deep_promp/output/', log_to_tensorboard: bool = False, learning_rate: float = 0.005, epochs: int = 100, beta: float = 0.01)`

Bases: [`moppy.interfaces.movement_primitive.MovementPrimitive`](/interfaces/movement_primitive)

### Parameters

- **name** (`str`): Name of the model.
- **encoder** (`EncoderDeepProMP`): Encoder component for DeepProMP.
- **decoder** (`DecoderDeepProMP`): Decoder component for DeepProMP.
- **save_path** (`str`, optional): Path to save the output files. Defaults to `./deep_promp/output/`.
- **log_to_tensorboard** (`bool`, optional): Whether to log training to TensorBoard. Defaults to `False`.
- **learning_rate** (`float`, optional): Learning rate for the optimizer. Defaults to `0.005`.
- **epochs** (`int`, optional): Number of epochs for training. Defaults to `100`.
- **beta** (`float`, optional): Regularization parameter for the model. Defaults to `0.01`.

---

### Functions

#### <span class="highlight-text">kl_annealing_scheduler()</span>

The `kl_annealing_scheduler` is a static method used to adjust the Kullback-Leibler (KL) divergence term during training, commonly used in variational models. This annealing process helps in gradually increasing the weight of the KL divergence term, preventing the model from relying too heavily on the prior too early during training. The scheduler operates over multiple cycles and gradually saturates based on the provided `saturation_point`.

<h4>Method Signature</h4>

`kl_annealing_scheduler(current_epoch: int, n_cycles: int = 4, max_epoch: int = 1000, saturation_point: float = 0.5)`

<h4>Parameters</h4>

- **current_epoch** (`int`): The current epoch during training.
- **n_cycles** (`int`, optional): Number of cycles to repeat the annealing process. Defaults to `4`.
- **max_epoch** (`int`, optional): Maximum number of training epochs. Defaults to `1000`.
- **saturation_point** (`float`, optional): The point at which the annealing process saturates (i.e., reaches its maximum). Defaults to `0.5`.

<h4>Returns</h4>

- **tau** (`float`): A value between `0` and `1`, representing the current weight of the KL divergence term.

<h4>Example</h4>

```python
# Using the KL annealing scheduler in training
kl_weight = DeepProMP.kl_annealing_scheduler(current_epoch=50)
```

---

#### <span class="highlight-text">gauss_kl()</span>

The `gauss_kl` method calculates the Kullback-Leibler (KL) divergence between a given Gaussian distribution (with parameters `mu_q` and `std_q`) and a standard Gaussian distribution. This divergence is a measure of how one probability distribution diverges from a second, reference distribution (in this case, the standard Gaussian). It's a key component of variational inference models.

<h4>Method Signature</h4>

`gauss_kl(mu_q: torch.Tensor, std_q: torch.Tensor) -> torch.Tensor`

<h4>Parameters</h4>

- **mu_q** (`torch.Tensor`): The mean of the approximate posterior distribution.
- **std_q** (`torch.Tensor`): The standard deviation of the approximate posterior distribution.

<h4>Returns</h4>

- **kl_divergence** (`torch.Tensor`): The mean KL divergence between the approximate posterior and the standard Gaussian distribution.

<h4>Example</h4>

```python
# Calculate KL divergence for a Gaussian distribution with given mean and std deviation
kl_div = DeepProMP.gauss_kl(mu_q=mu_tensor, std_q=std_tensor)
```

---

#### <span class="highlight-text">calculate_elbo()</span>

The `calculate_elbo` method computes the Evidence Lower Bound (ELBO), which is the objective function used to train variational models like DeepProMP. The ELBO consists of two main components: the reconstruction loss (typically Mean Squared Error) and the KL divergence, weighted by a factor `beta`. The goal of ELBO is to balance between fitting the data and keeping the posterior distribution close to the prior.

<h4>Method Signature</h4>

`calculate_elbo(y_pred: torch.Tensor, y_star: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

<h4>Parameters</h4>

- **y_pred** (`torch.Tensor`): The predicted output from the model.
- **y_star** (`torch.Tensor`): The ground truth target values.
- **mu** (`torch.Tensor`): The mean of the approximate posterior distribution.
- **sigma** (`torch.Tensor`): The standard deviation of the approximate posterior distribution.
- **beta** (`float`, optional): The weight applied to the KL divergence term in the ELBO. Defaults to `1.0`.

<h4>Returns</h4>

- **elbo** (`torch.Tensor`): The total ELBO loss, combining the reconstruction loss and KL divergence.
- **mse** (`torch.Tensor`): The Mean Squared Error reconstruction loss.
- **kl** (`torch.Tensor`): The KL divergence between the approximate posterior and the prior.

<h4>Example</h4>

```python
# Calculate ELBO during training
elbo_loss, mse_loss, kl_div = DeepProMP.calculate_elbo(y_pred=pred, y_star=target, mu=mu_tensor, sigma=std_tensor)
```

---

#### <span class="highlight-text">train()</span>

The `train` method is responsible for training the `DeepProMP` model using the provided trajectories. The training process is guided by the Evidence Lower Bound (ELBO) as the loss function. It leverages the Adam optimizer for updating model parameters and supports KL annealing for better regularization during training. The method divides the data into training and validation sets, logs metrics, and saves the model and losses after training.

<h4>Method Signature</h4>

`train(trajectories: List[Trajectory], kl_annealing: bool = True, beta: float = None, learning_rate: float = None, epochs: int = None) -> None`

<h4>Parameters</h4>

- **trajectories** (`List[Trajectory]`): A list of trajectory data used for training the model.
- **kl_annealing** (`bool`, optional): If `True`, applies KL annealing during training. Defaults to `True`.
- **beta** (`float`, optional): The regularization weight for the KL divergence term. If not specified, uses the default value in the class.
- **learning_rate** (`float`, optional): The learning rate for the optimizer. If not specified, uses the default value in the class.
- **epochs** (`int`, optional): The number of training epochs. If not specified, uses the default value in the class.

<h4>Returns</h4>

- **None**

<h4>Description</h4>

The `train` method performs the following steps:

1. Optionally adjusts the `beta`, `learning_rate`, and `epochs` values based on the provided arguments.
2. Divides the input `trajectories` into training and validation sets.
3. Initializes the Adam optimizer to update both the encoder and decoder parameters.
4. For each epoch:
    - Loops over the training set, computes the ELBO loss (a combination of Mean Squared Error and KL divergence), and updates the model.
    - If KL annealing is enabled, it applies a scheduling function to gradually increase the weight of the KL divergence term over training.
    - Logs the metrics for each epoch.
    - Validates the model on the validation set.
5. After training is completed:
    - Saves the training losses, validation losses, and models.
    - Generates loss plots.

<h4>Example</h4>

```python
# Train the DeepProMP model with a list of trajectories
model = DeepProMP(name="example_model", encoder=encoder, decoder=decoder)
model.train(trajectories=trajectories_list, kl_annealing=True, beta=0.01, learning_rate=0.001, epochs=100)
```

---

#### <span class="highlight-text">validate()</span>

The `validate` method evaluates the performance of the trained `DeepProMP` model on a set of trajectories. It computes the Mean Squared Error (MSE) between the predicted (decoded) output and the actual trajectory data for each trajectory in the validation set. The average loss over all validation trajectories is returned as the validation loss.

<h4>Method Signature</h4>

`validate(trajectories: List[Trajectory]) -> float`

<h4>Parameters</h4>

- **trajectories** (`List[Trajectory]`): A list of trajectory data used for validation.

<h4>Returns</h4>

- **validation_loss** (`float`): The average Mean Squared Error (MSE) loss over all validation trajectories.

<h4>Description</h4>

The `validate` method performs the following steps:

1. For each trajectory in the validation set:
    - Passes the trajectory through the encoder to obtain the posterior distribution parameters (`mu` and `sigma`).
    - Samples the latent variable `z` from the posterior.
    - Decodes the latent variable at each time point to reconstruct the trajectory.
    - Computes the Mean Squared Error (MSE) between the reconstructed trajectory and the ground truth trajectory.
2. Averages the MSE loss over all validation trajectories.
3. Returns the average validation loss.

<h4>Example</h4>

```python
# Validate the DeepProMP model on a list of trajectories
validation_loss = model.validate(trajectories=validation_trajectories_list)
print(f"Validation Loss: {validation_loss}")
```

---

#### <span class="highlight-text">save_models()</span>

The `save_models` method saves the encoder and decoder models to the specified path. If no path is provided, it uses the default save path.

<h4>Method Signature</h4>

`save_models(save_path: str = None) -> None`

<h4>Parameters:</h4>

- **save_path** (`str`, optional): The path where the models should be saved. If `None`, the default `save_path` is used.

<h4>Example:</h4>

```python
# Save models to the default path
model.save_models()

# Save models to a custom path
model.save_models(save_path='./custom_path/')
```

#### <span class="highlight-text">save_losses()</span>

The `save_losses` method saves the losses (training, validation, KL divergence, MSE) to the specified path. If no path is provided, it uses the default save path.

<h4>Method Signature</h4>

`save_losses(save_path: str = None) -> None`

<h4>Parameters</h4>

- **save_path** (`str`, optional): The path where the losses should be saved. If `None`, the default `save_path` is used.

<h4>Description</h4>

This method saves the following loss values to disk:

- Validation loss (`validation_loss.pth`)
- KL divergence loss (`kl_loss.pth`)
- Mean Squared Error loss (`mse_loss.pth`)
- Training loss (`train_loss.pth`)

For each type of loss, the method saves the values using the `torch.save()` function.

<h4>Example</h4>

```python
# Save losses to the default path
model.save_losses()

# Save losses to a custom path
model.save_losses(save_path='./custom_path/')
```

---

#### <span class="highlight-text">plot_values()</span>

The `plot_values` method plots the provided values and saves the plot to the specified path. If no path is provided, it uses the default save path.

<h4>Method Signature</h4>

`plot_values(values: List[List], file_name: str, plot_title: str = "Plot", path: str = None) -> None`

<h4>Parameters</h4>

- **values** (`List[List]`): The values to be plotted. Each inner list represents a line in the plot.
- **file_name** (`str`): The name of the file where the plot will be saved.
- **plot_title** (`str`, optional): The title of the plot. Defaults to `"Plot"`.
- **path** (`str`, optional): The path where the plot should be saved. If `None`, the default `save_path` is used.

<h4>Example</h4>

```python
# Plot the values and save to the default path
model.plot_values(values=[[1, 2, 3], [4, 5, 6]], file_name='example_plot.png', plot_title="Example Plot")

# Plot the values and save to a custom path
model.plot_values(values=[[1, 2, 3], [4, 5, 6]], file_name='example_plot.png', plot_title="Example Plot", path='./custom_path/')
```

---

## DecoderDeepProMP

`moppy.deep_promp.decoder_deep_pro_mp.DeepProMP(self, latent_variable_dimension: int, hidden_neurons: List[int], trajectory_state_class: Type[TrajectoryState] = JointConfiguration, activation_function: Type[nn.Module] = nn.ReLU, activation_function_params: dict = {})`

Bases: [`moppy.interfaces.movement_primitive.LatentDecoder`](/interfaces/latent_decoder)

The `DecoderDeepProMP` class implements a latent decoder architecture, extending `LatentDecoder` and `nn.Module`. It is designed to decode a latent variable into a trajectory state.

### Parameters

- **latent_variable_dimension** (`int`): The dimension of the latent variable.
- **hidden_neurons** (`List[int]`): A list of integers representing the number of neurons in each hidden layer.
- **trajectory_state_class** (`Type[TrajectoryState]`): The class of the trajectory state (default: `JointConfiguration`).
- **activation_function** (`Type[nn.Module]`): The activation function to be used in the network (default: `nn.ReLU`).
- **activation_function_params** (`dict`): Parameters for the activation function.

<h4>Raises:</h4>

- **TypeError**: If `trajectory_state_class` is not a subclass of `TrajectoryState`.
- **ValueError**: If `latent_variable_dimension` is less than or equal to 0, if the number of neurons is less than 2, or if any neuron count is not greater than 0.

---

### Functions

#### <span class="highlight-text">load_from_save_file()</span>

Loads a model from a file and returns a `DecoderDeepProMP` instance. It uses a the savefile created by [`save_decoder`](#save_decoder).

<h4>Method Signature</h4>

`load_from_save_file(cls, path: str = '', file: str = "decoder_deep_pro_mp.pth") -> 'DecoderDeepProMP'`

<h4>Parameters:</h4>

- **path** (`str`): The path to the directory containing the model file.
- **file** (`str`): The name of the file to load (default: `"decoder_deep_pro_mp.pth"`).

<h4>Returns:</h4>

- **DecoderDeepProMP**: An instance of `DecoderDeepProMP`.

---

#### <span class="highlight-text">create_layers()</span>

Creates the layers of the decoder network based on the specified number of neurons.

**Returns:**

- **List[nn.Module]**: A list of layers for the neural network.

---

#### <span class="highlight-text">__init_weights()</span>

Initializes the weights and biases of the network using Xavier initialization.

<h4>Method Signature</h4>

`__init_weights(self, m) -> None`

<h4>Parameters:</h4>

- **m**: The layer to initialize.

---

#### <span class="highlight-text">decode_from_latent_variable()</span>

Overrides: [moppy.interfaces.latent_decoder.LatentDecoder.decode_from_latent_variable](/interfaces/latent_decoder/#decode_from_latent_variable)

Decodes a latent variable into a tensor representing the trajectory state.

<h4>Method Signature</h4>

`decode_from_latent_variable(self, latent_variable: torch.Tensor, time: Union[torch.Tensor, float]) -> torch.Tensor`

<h4>Parameters:</h4>

- **latent_variable** (`torch.Tensor`): The latent variable to decode.
- **time** (`Union[torch.Tensor, float]`): The normalized time value.

<h4>Returns:</h4>

- **torch.Tensor**: The decoded trajectory state.

---

#### <span class="highlight-text">save_decoder()</span>

Saves the decoder model to a file, including the state dictionary and configuration.

**Method Signature**

`save_decoder(self, path: str = '', filename: str = "decoder_deep_pro_mp.pth")`

**Parameters:**

- **path** (`str`): The directory to save the model.
- **filename** (`str`): The name of the file to save (default: `"decoder_deep_pro_mp.pth"`).

---

#### <span class="highlight-text">save_model()</span>

Saves only the model's state dictionary.

**Method Signature**

`save_model(self, path: str = '', filename: str = "decoder_model_deep_pro_mp.pth")`

**Parameters:**

- **path** (`str`): The directory to save the model.
- **filename** (`str`): The name of the file to save (default: `"decoder_model_deep_pro_mp.pth"`).

---

#### <span class="highlight-text">load_model()</span>

Loads the model's state dictionary (`net.state_dict()`) from a file. The save_file can be created by created by [`save_model`](#save_model).

**Method Signature**

`load_model(self, path: str = '', filename: str = "decoder_model_deep_pro_mp.pth")`

**Parameters:**

- **path** (`str`): The directory containing the model file.
- **filename** (`str`): The name of the file to load (default: `"decoder_model_deep_pro_mp.pth"`).

---

#### <span class="highlight-text">forward()</span>

Defines the forward pass of the decoder.

**Method Signature**

`forward(self, latent_variable: torch.Tensor, time: Union[torch.Tensor, float])`

**Parameters:**

- **latent_variable** (`torch.Tensor`): The latent variable to decode.
- **time** (`Union[torch.Tensor, float]`): The normalized time value.

**Returns:**

- **torch.Tensor**: The decoded trajectory state.

## EncoderDeepProMP

`moppy.deep_promp.encoder_deep_pro_mp.EncoderDeepProMP(self, latent_variable_dimension: int, hidden_neurons: List[int], trajectory_state_class: Type[TrajectoryState] = JointConfiguration, activation_function: Type[nn.Module] = nn.ReLU, activation_function_params: dict = {})`

Bases: [`moppy.interfaces.movement_primitive.LatentEncoder`](/interfaces/latent_encoder)

The `EncoderDeepProMP` class implements a latent encoder architecture, extending `LatentEncoder` and `nn.Module`. It is designed to encode a trajectory into a latent variable.

### Parameters

- **latent_variable_dimension** (`int`): The dimension of the latent variable.
- **hidden_neurons** (`List[int]`): A list of integers representing the number of neurons in each hidden layer.
- **trajectory_state_class** (`Type[TrajectoryState]`): The class of the trajectory state (default: `JointConfiguration`).
- **activation_function** (`Type[nn.Module]`): The activation function to be used in the network (default: `nn.ReLU`).
- **activation_function_params** (`dict`): Parameters for the activation function.

<h4>Raises:</h4>

- **TypeError**: If `trajectory_state_class` is not a subclass of `TrajectoryState`.
- **ValueError**: If `latent_variable_dimension` is less than or equal to 0, if the number of neurons is less than 2, or if any neuron count is not greater than 0.

---

### Functions

#### <span class="highlight-text">load_from_save_file()</span>

Loads a model from a file and returns an `EncoderDeepProMP` instance. It uses a save file created by [`save_encoder`](#save_encoder).

<h4>Method Signature</h4>

`load_from_save_file(cls, path: str = '', file: str = "encoder_deep_pro_mp.pth") -> 'EncoderDeepProMP'`

<h4>Parameters:</h4>

- **path** (`str`): The path to the directory containing the model file.
- **file** (`str`): The name of the file to load (default: `"encoder_deep_pro_mp.pth"`).

<h4>Returns:</h4>

- **EncoderDeepProMP**: An instance of `EncoderDeepProMP`.

---

#### <span class="highlight-text">create_layers()</span>

Creates the layers of the encoder network based on the specified number of neurons.

**Returns:**

- **List[nn.Module]**: A list of layers for the neural network.

---

#### <span class="highlight-text">__init_weights()</span>

Initializes the weights and biases of the network using Xavier initialization.

<h4>Method Signature</h4>

`__init_weights(self, m) -> None`

<h4>Parameters:</h4>

- **m**: The layer to initialize.

---

#### <span class="highlight-text">encode_to_latent_variable()</span>

Encodes a trajectory into a mu and sigma (both with dimensions of `latent_variable_dimension`).

<h4>Method Signature</h4>

`encode_to_latent_variable(self, trajectory: Trajectory) -> tuple[Tensor, Tensor]`

<h4>Parameters:</h4>

- **trajectory** (`Trajectory`): The trajectory to encode.

<h4>Returns:</h4>

- **tuple[Tensor, Tensor]**: The resulting `mu` and `sigma` tensors, each with size `latent_variable_dimension`.

---

#### <span class="highlight-text">sample_latent_variable()</span>

Samples a latent variable `z` from a normal distribution specified by `mu` and `sigma`.

<h4>Method Signature</h4>

`sample_latent_variable(self, mu: torch.Tensor, sigma: torch.Tensor, percentage_of_standard_deviation=None) -> torch.Tensor`

<h4>Parameters:</h4>

- **mu** (`torch.Tensor`): The mean tensor.
- **sigma** (`torch.Tensor`): The standard deviation tensor.
- **percentage_of_standard_deviation** (`Optional[float]`): A percentage to scale the standard deviation (optional).

<h4>Returns:</h4>

- **torch.Tensor**: The sampled latent variable.

---

#### <span class="highlight-text">sample_latent_variables()</span>

Samples multiple latent variables from given `mu` and `sigma` tensors.

<h4>Method Signature</h4>

`sample_latent_variables(self, mu: torch.Tensor, sigma: torch.Tensor, size: int = 1) -> torch.Tensor`

<h4>Parameters:</h4>

- **mu** (`torch.Tensor`): The mean tensor.
- **sigma** (`torch.Tensor`): The standard deviation tensor.
- **size** (`int`): The number of samples to generate (default: `1`).

<h4>Returns:</h4>

- **torch.Tensor**: A tensor containing the sampled latent variables.

---

#### <span class="highlight-text">bayesian_aggregation()</span>

Performs Bayesian aggregation on `mu_points` and `sigma_points` to calculate aggregated `mu` and `sigma`.

<h4>Method Signature</h4>

`bayesian_aggregation(self, mu_points: torch.Tensor, sigma_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`

<h4>Parameters:</h4>

- **mu_points** (`torch.Tensor`): Tensor of `mu` points.
- **sigma_points** (`torch.Tensor`): Tensor of `sigma` points.

<h4>Returns:</h4>

- **tuple[torch.Tensor, torch.Tensor]**: Aggregated `mu_z` and `sigma_z_sq` tensors.

---

#### <span class="highlight-text">save_encoder()</span>

Saves the encoder model to a file, including the state dictionary and configuration.

**Method Signature**

`save_encoder(self, path: str = '', filename: str = "encoder_deep_pro_mp.pth")`

**Parameters:**

- **path** (`str`): The directory to save the model.
- **filename** (`str`): The name of the file to save (default: `"encoder_deep_pro_mp.pth"`).

---

#### <span class="highlight-text">save_model()</span>

Saves only the model's state dictionary.

**Method Signature**

`save_model(self, path: str = '', filename: str = "encoder_model_deep_pro_mp.pth")`

**Parameters:**

- **path** (`str`): The directory to save the model.
- **filename** (`str`): The name of the file to save (default: `"encoder_model_deep_pro_mp.pth"`).

---

#### <span class="highlight-text">load_model()</span>

Loads the model's state dictionary from a file.

**Method Signature**

`load_model(self, path: str = '', filename: str = "encoder_model_deep_pro_mp.pth")`

**Parameters:**

- **path** (`str`): The directory containing the model file.
- **filename** (`str`): The name of the file to load (default: `"encoder_model_deep_pro_mp.pth"`).

---

#### <span class="highlight-text">forward()</span>

Defines the forward pass of the encoder.

**Method Signature**

`forward(self, trajectory: Trajectory) -> tuple[Tensor, Tensor]`

**Parameters:**

- **trajectory** (`Trajectory`): The trajectory to encode.

**Returns:**

- **tuple[Tensor, Tensor]**: The encoded `mu` and `sigma` tensors.

---
