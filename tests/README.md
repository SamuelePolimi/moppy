[![moppy-test-ci](https://github.com/SamuelePolimi/moppy/actions/workflows/run-tests.yml/badge.svg)](https://github.com/SamuelePolimi/moppy/actions/workflows/run-tests.yml)

# Testing

In this directory are all the tests for moppy located.

## Existing Tests

- [Trajectory](/tests/Trajectory/):
  - [Trajectory](/tests/Trajectory/test_trajectory.py)
  - [State](/tests/Trajectory/TrajectoryState/):
    - [SinusState](/tests/Trajectory/TrajectoryState/test_sinus_state.py)
    - [Joint Configuration](/tests/Trajectory/TrajectoryState/test_joint_configuration.py)
- [DeepProMP](/tests/deeppromp/):
  - [DeepProMP Encoder](/tests/deeppromp/test_encoder_pro_mp.py)
  - [DeepProMP Decoder](/tests/deeppromp/test_decoder_pro_mp.py)

## CI

The workflow [moppy-test-ci](https://github.com/SamuelePolimi/moppy/actions/workflows/run-tests.yml) will run all the test for moppy for:

- each **commit on the main**
- each **Pull-Request**

### Conda Cache

Since we are using a conda envirement the ci will create a conda envirement from **[/conda/environment.yml](/conda/environment.yml)**, install moppy in it and use that envirement to test. Since this takes some time, we use the **actions/cache@v3** to cache the the envirement and use it again for the next check, if there are no changes made to the envirement.yml file. Github offers **100 chaches per repo** and if full need to be deleted.

## TO FIX

At the moment the **caching does not work perfectly**. The cache will only be found for each action. So if a new PR is made is will not find a cache and create a new one, but the second run on the PR will find the new cache. Same for the main.
