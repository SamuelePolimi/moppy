[![moppy-test-ci](https://github.com/SamuelePolimi/moppy/actions/workflows/run-tests.yml/badge.svg)](https://github.com/SamuelePolimi/moppy/actions/workflows/run-tests.yml)
# moppy

MoPPy is a python library that implements several movement primitives for robotics.

## Getting started

Install the library with

```bash
pip install .
```
You can now use this library in another project!

## Examples

Take a look in the [example folder](examples/), where you can find multiple implementations showcasing the functionality of the library.

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

You can use [the small sinus example](examples/small_examples/small_sinus_example.py) to verify the code is still working fine. 

TODO sanity checks for trajectory and trajectory state classes
