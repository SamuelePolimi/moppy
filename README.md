[![moppy-test-ci](https://github.com/SamuelePolimi/moppy/actions/workflows/run-tests.yml/badge.svg)](https://github.com/SamuelePolimi/moppy/actions/workflows/run-tests.yml)

# moppy

MoPPy is a python library that implements several movement primitives for robotics.

<!---
pyreverse -o png --colorized -k moppy
-->
![Uml structure of the library](classes.png)

## Getting started

Install the library with

```bash
pip install .
```

You can now use this library in another project!

## Examples

Take a look in the [example folder](examples/), where you can find multiple implementations showcasing the functionality of the library.

## DeepProMP

The library also includes a DeepProMP implementation, which is a probabilistic movement primitive that uses a neural network to model the distribution of the movement primitive.

The DeepProMP implementation is based on the paper "[Deep Probabilistic Movement Primitives with a Bayesian Aggregator](https://arxiv.org/pdf/2307.05141)".

### Example

In the [example folder](examples/) you can find an example of how to use the DeepProMP implementation.

The example [small_sinus_example.py](examples/small_examples/small_sinus_example.py) shows how to train a DeepProMP to learn a sinusoidal movement primitive. The trained DeepProMP is then used to generate new samples.
The resulting plot shows the training data, the learned DeepProMP, and the generated samples.

The result should look like this:

<p align="center">
  <img src="readme_images/all_losses.png" alt="Image 1" width="49%">
  <img src="readme_images/Trajectories_Comparison.png" alt="Image 2" width="49%">
</p>

```bash

```

### Usage

## Development

### Creating the conda enviroment

```bash
    conda env create -f environment.yml
    conda activate moppy
```

Then install the library in editable mode

```bash
pip install -e .
```

### Testing

- All the tests can be found in the [tests](/tests/) folder.
- There exists a ci for testings.
- For mor information on testing in moppy follow [README.md](/tests/README.md).
