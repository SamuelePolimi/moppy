from typing import List, Type

import os
import torch
import torch.nn as nn
from torch import Tensor


class RobosuiteDemoStartingPosition:
    """Class representing the starting position of a demonstration in the Robosuite environment.

    Attributes:
        input_dimension (int): The expected input dimension for the class.

    """

    input_dimension = 55

    def __init__(self,
                 robot0_joint_pos_cos: List[float],
                 robot0_joint_pos_sin: List[float],
                 robot0_joint_vel: List[float],
                 robot0_eef_pos: List[float],
                 robot0_eef_quat: List[float],
                 robot0_gripper_qpos: List[float],
                 robot0_gripper_qvel: List[float],
                 cubeA_pos: List[float],
                 cubeA_quat: List[float],
                 cubeB_pos: List[float],
                 cubeB_quat: List[float],
                 gripper_to_cubeA: List[float],
                 gripper_to_cubeB: List[float],
                 cubeA_to_cubeB: List[float]):
        assert len(robot0_joint_pos_cos) == 7, "Expected length of robot0_joint_pos_cos is 7"
        assert len(robot0_joint_pos_sin) == 7, "Expected length of robot0_joint_pos_sin is 7"
        assert len(robot0_joint_vel) == 7, "Expected length of robot0_joint_vel is 7"
        assert len(robot0_eef_pos) == 3, "Expected length of robot0_eef_pos is 3"
        assert len(robot0_eef_quat) == 4, "Expected length of robot0_eef_quat is 4"
        assert len(robot0_gripper_qpos) == 2, "Expected length of robot0_gripper_qpos is 2"
        assert len(robot0_gripper_qvel) == 2, "Expected length of robot0_gripper_qvel is 2"
        assert len(cubeA_pos) == 3, "Expected length of cubeA_pos is 3"
        assert len(cubeA_quat) == 4, "Expected length of cubeA_quat is 4"
        assert len(cubeB_pos) == 3, "Expected length of cubeB_pos is 3"
        assert len(cubeB_quat) == 4, "Expected length of cubeB_quat is 4"
        assert len(gripper_to_cubeA) == 3, "Expected length of gripper_to_cubeA is 3"
        assert len(gripper_to_cubeB) == 3, "Expected length of gripper_to_cubeB is 3"
        assert len(cubeA_to_cubeB) == 3, "Expected length of cubeA_to_cubeB is 3"

        self.robot0_joint_pos_cos = robot0_joint_pos_cos
        self.robot0_joint_pos_sin = robot0_joint_pos_sin
        self.robot0_joint_vel = robot0_joint_vel
        self.robot0_eef_pos = robot0_eef_pos
        self.robot0_eef_quat = robot0_eef_quat
        self.robot0_gripper_qpos = robot0_gripper_qpos
        self.robot0_gripper_qvel = robot0_gripper_qvel
        self.cubeA_pos = cubeA_pos
        self.cubeA_quat = cubeA_quat
        self.cubeB_pos = cubeB_pos
        self.cubeB_quat = cubeB_quat
        self.gripper_to_cubeA = gripper_to_cubeA
        self.gripper_to_cubeB = gripper_to_cubeB
        self.cubeA_to_cubeB = cubeA_to_cubeB

    @classmethod
    def from_array(cls, array):
        """
        Creates an instance of the class from a given array.
        Args:
            array (list or np.ndarray): Input array containing the values to initialize the class attributes.
                The array must have a length of 55 and should not be None.
        Returns:
            An instance of the class with attributes initialized from the input array.
        Raises:
            AssertionError: If the input array is None or its length is not equal to the expected input dimension.
        """

        assert array is not None, "Input array cannot be None"
        assert len(array) == cls.input_dimension, "Expected length of array is 55"
        return cls(
            robot0_joint_pos_cos=array[0:7],
            robot0_joint_pos_sin=array[7:14],
            robot0_joint_vel=array[14:21],
            robot0_eef_pos=array[21:24],
            robot0_eef_quat=array[24:28],
            robot0_gripper_qpos=array[28:30],
            robot0_gripper_qvel=array[30:32],
            cubeA_pos=array[32:35],
            cubeA_quat=array[35:39],
            cubeB_pos=array[39:42],
            cubeB_quat=array[42:46],
            gripper_to_cubeA=array[46:49],
            gripper_to_cubeB=array[49:52],
            cubeA_to_cubeB=array[52:55]
        )

    @classmethod
    def from_dict(cls, dict):
        """
        Creates an instance of the class from a given dictionary.
        Args:
            dict (dict): Input dictionary containing the values to initialize the class attributes.
            The dictionary must have keys corresponding to the class attributes.
        Returns:
            An instance of the class with attributes initialized from the input dictionary.
        Raises:
            AssertionError: If the input dictionary is None or does not contain the required keys.
        """
        assert dict is not None, "Input dictionary cannot be None"
        required_keys = [
            'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos',
            'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'cubeA_pos', 'cubeA_quat',
            'cubeB_pos', 'cubeB_quat', 'gripper_to_cubeA', 'gripper_to_cubeB', 'cubeA_to_cubeB'
        ]
        for key in required_keys:
            assert key in dict, f"Missing key '{key}' in input dictionary"

        return cls(
            robot0_joint_pos_cos=dict['robot0_joint_pos_cos'],
            robot0_joint_pos_sin=dict['robot0_joint_pos_sin'],
            robot0_joint_vel=dict['robot0_joint_vel'],
            robot0_eef_pos=dict['robot0_eef_pos'],
            robot0_eef_quat=dict['robot0_eef_quat'],
            robot0_gripper_qpos=dict['robot0_gripper_qpos'],
            robot0_gripper_qvel=dict['robot0_gripper_qvel'],
            cubeA_pos=dict['cubeA_pos'],
            cubeA_quat=dict['cubeA_quat'],
            cubeB_pos=dict['cubeB_pos'],
            cubeB_quat=dict['cubeB_quat'],
            gripper_to_cubeA=dict['gripper_to_cubeA'],
            gripper_to_cubeB=dict['gripper_to_cubeB'],
            cubeA_to_cubeB=dict['cubeA_to_cubeB']
        )

    def get_object_state_as_flatten_torch(self):
        """
        Returns the state of the objects as a flattened PyTorch tensor.

        The resulting tensor is built by concatenating the following components in order:
        - cubeA_pos: Position of cubeA (3 elements)
        - cubeA_quat: Orientation of cubeA as a quaternion (4 elements)
        - cubeB_pos: Position of cubeB (3 elements)
        - cubeB_quat: Orientation of cubeB as a quaternion (4 elements)
        - gripper_to_cubeA: Relative position between the gripper and cubeA (3 elements)
        - gripper_to_cubeB: Relative position between the gripper and cubeB (3 elements)
        - cubeA_to_cubeB: Relative position between cubeA and cubeB (3 elements)

        Returns:
            torch.Tensor: A 1D tensor containing the concatenated state information of the objects.
        """

        object_state = \
            self.cubeA_pos + \
            self.cubeA_quat + \
            self.cubeB_pos + \
            self.cubeB_quat + \
            self.gripper_to_cubeA + \
            self.gripper_to_cubeB + \
            self.cubeA_to_cubeB
        return torch.tensor(object_state, dtype=torch.float32)

    def get_robot0_proprio_state_as_flatten_torch(self):
        """
        Get the proprioceptive state of robot0 as a flattened torch tensor.

        The proprioceptive state includes:
        - Joint positions (cosine and sine components) (14 elements)
        - Joint velocities (7 elements)
        - End-effector position (3 elements)
        - End-effector orientation (quaternion) (4 elements)
        - Gripper positions (2 elements)
        - Gripper velocities (2 elements)

        Returns:
            torch.Tensor: The proprioceptive state of robot0 as a flattened tensor.
        """

        robot0_proprio_state = \
            self.robot0_joint_pos_cos + \
            self.robot0_joint_pos_sin + \
            self.robot0_joint_vel + \
            self.robot0_eef_pos + \
            self.robot0_eef_quat + \
            self.robot0_gripper_qpos + \
            self.robot0_gripper_qvel
        return torch.tensor(robot0_proprio_state, dtype=torch.float32)

    def get_as_flatten_torch(self):
        """
        Returns the concatenated tensor of object state and robot proprioceptive state.
        This method retrieves the object state and robot proprioceptive state as flattened
        tensors and concatenates them into a single tensor.

        The concatenated tensor contains the following components in order:
        - Object state (17 elements)
        - Robot proprioceptive state (32 elements)

        Here is the breakdown of the concatenated tensor:
        - cubeA_pos**: Position of cubeA (3 elements)
        - cubeA_quat: Orientation of cubeA as a quaternion (4 elements)
        - cubeB_pos: Position of cubeB (3 elements)
        - cubeB_quat: Orientation of cubeB as a quaternion (4 elements)
        - gripper_to_cubeA: Relative position between the gripper and cubeA (3 elements)
        - gripper_to_cubeB: Relative position between the gripper and cubeB (3 elements)
        - cubeA_to_cubeB: Relative position between cubeA and cubeB (3 elements)
        - robot0_joint_pos_cos: Joint positions (cosine components) of robot0 (7 elements)
        - robot0_joint_pos_sin: Joint positions (sine components) of robot0 (7 elements)
        - robot0_joint_vel: Joint velocities of robot0 (7 elements)
        - robot0_eef_pos: End-effector position of robot0 (3 elements)
        - robot0_eef_quat: End-effector orientation of robot0 as a quaternion (4 elements)
        - robot0_gripper_qpos: Gripper positions of robot0 (2 elements)
        - robot0_gripper_qvel: Gripper velocities of robot0 (2 elements)

        Returns:
            torch.Tensor: A concatenated tensor of object state and robot proprioceptive state.
        """

        object_state = self.get_object_state_as_flatten_torch()
        robot0_proprio_state = self.get_robot0_proprio_state_as_flatten_torch()
        return torch.cat((robot0_proprio_state, object_state))

    def __str__(self):
        ret = "RobosuiteDemoStartingPosition {\n"
        ret += f"\trobot0_joint_pos_cos: {self.robot0_joint_pos_cos}\n"
        ret += f"\trobot0_joint_pos_sin: {self.robot0_joint_pos_sin}\n"
        ret += f"\trobot0_joint_vel: {self.robot0_joint_vel}\n"
        ret += f"\trobot0_eef_pos: {self.robot0_eef_pos}\n"
        ret += f"\trobot0_eef_quat: {self.robot0_eef_quat}\n"
        ret += f"\trobot0_gripper_qpos: {self.robot0_gripper_qpos}\n"
        ret += f"\trobot0_gripper_qvel: {self.robot0_gripper_qvel}\n"
        ret += f"\tcubeA_pos: {self.cubeA_pos}\n"
        ret += f"\tcubeA_quat: {self.cubeA_quat}\n"
        ret += f"\tcubeB_pos: {self.cubeB_pos}\n"
        ret += f"\tcubeB_quat: {self.cubeB_quat}\n"
        ret += f"\tgripper_to_cubeA: {self.gripper_to_cubeA}\n"
        ret += f"\tgripper_to_cubeB: {self.gripper_to_cubeB}\n"
        ret += f"\tcubeA_to_cubeB: {self.cubeA_to_cubeB}\n"
        ret += "}"
        return ret

    def __repr__(self):
        return self.__str__()


class EncoderAsActor(nn.Module):
    """EncoderAsActor is a neural network model that encodes input data into a latent variable representation.
    Its resuting Neural Network is intendet to use as a starting actor for a Robosuite task.

    Attributes:
        activation_function (Type[nn.Module]): The activation function to use in the network layers.
        activation_function_params (dict): Parameters for the activation function.
        hidden_neurons (List[int]): List of integers representing the number of neurons in each hidden layer.
        latent_variable_dimension (int): The dimension of the latent variable output.
        input_dimension (int): The dimension of the input data.
        neurons (List[int]): List of integers representing the number of neurons in each layer,
            including input and output layers.
        net (nn.Sequential): The neural network model.
    """

    def __init__(self,
                 latent_variable_dimension: int,
                 hidden_neurons: List[int],
                 activation_function: Type[nn.Module] = nn.ReLU,
                 activation_function_params: dict = {}):
        nn.Module.__init__(self)

        self.activation_function = activation_function
        self.activation_function_params = activation_function_params
        self.hidden_neurons = hidden_neurons
        self.latent_variable_dimension = latent_variable_dimension
        self.input_dimension = RobosuiteDemoStartingPosition.input_dimension

        # create the neurons list, which is the list of the number of neurons in each layer of the network
        self.neurons = [self.input_dimension] + hidden_neurons + [latent_variable_dimension]

        # Check if the neurons list is empty or has fewer than 2 elements
        if not self.neurons or len(self.neurons) < 2:
            raise ValueError("The number of neurons must be at least 2. Got '%s'" % self.neurons)
        if not all(isinstance(neuron, int) for neuron in self.neurons):
            raise ValueError("All elements of neurons must be of type int. Got '%s'" % self.neurons)
        if not all(neuron > 0 for neuron in self.neurons):
            raise ValueError("All elements of neurons must be greater than 0. Got '%s'" % self.neurons)

        layers = self.create_layers()
        self.net = nn.Sequential(*layers).float()

        # Initialize the weights and biases of the network
        self.net.apply(self.__init_weights)

    @classmethod
    def load_from_save_file(cls, path: str = '', file: str = "encoder_as_actor.pth") -> 'EncoderAsActor':
        """Load a model from a file and return a EncoderDeepProMP instance."""
        file_path = os.path.join(path, file)
        # Load the model data
        model_data = torch.load(file_path)

        # Reconstruct the model using the saved configuration
        model = cls(
            latent_variable_dimension=model_data['latent_variable_dimension'],
            hidden_neurons=model_data['hidden_neurons'],
            activation_function=model_data['activation_function'],
            activation_function_params=model_data['activation_function_params']
        )

        # Load the model weights
        model.net.load_state_dict(model_data['state_dict'])

        return model

    def save_encoder(self, path: str = '', filename: str = "encoder_as_actor.pth"):
        """Save the encoder to a file, including the state_dict of the network and the configuration of the model.
        The configuration includes:

        - latent_variable_dimension
        - hidden_neurons
        - trajectory_state_class
        - activation function
        - activation function parameters

        Can be loaded using the `EncoderAsActor.load_from_save_file` method."""

        file_path = os.path.join(path, filename)
        model_data = {
            'state_dict': self.net.state_dict(),
            'latent_variable_dimension': self.latent_variable_dimension,
            'hidden_neurons': self.hidden_neurons,
            'activation_function': self.activation_function,
            'activation_function_params': self.activation_function_params
        }
        torch.save(model_data, file_path)

    def save_model(self, path: str = '', filename: str = "encoder_as_actor_model.pth"):
        file_path = os.path.join(path, filename)
        torch.save(self.net, file_path)

    def load_model(self, path: str = '', filename: str = "encoder_as_actor_model.pth"):
        file_path = os.path.join(path, filename)
        self.net.load_state_dict(torch.load(file_path))

    def create_layers(self):
        layers = []
        for i in range(len(self.neurons) - 2):
            layers += [nn.Linear(self.neurons[i], self.neurons[i + 1]),
                       self.activation_function(**self.activation_function_params)]
        layers += [nn.Linear(self.neurons[-2], self.neurons[-1])]
        return layers

    def __init_weights(self, m):
        """Initialize the weights and biases of the network using Xavier initialization and a bias"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                nn.init.constant_(m.bias, 0.05)

    def encode_to_latent_variable(self, input: RobosuiteDemoStartingPosition | List | Tensor) -> Tensor:
        """
        Encodes the input into a latent variable tensor.

        Args:
            input (RobosuiteDemoStartingPosition | List | Tensor): The input data to be encoded.
            It can be one of the following types:
                - RobosuiteDemoStartingPosition, List or Tensor with len `RobosuiteDemoStartingPosition.input_dimension`

        Returns:
            Tensor: The encoded latent variable tensor.
        """
        """"""
        input_tensor = None
        if isinstance(input, list):
            input_tensor = torch.tensor(input, dtype=torch.float32)
        elif isinstance(input, RobosuiteDemoStartingPosition):
            input_tensor = input.get_as_flatten_torch()
        elif isinstance(input, Tensor):
            input_tensor = input

        return self.net(input_tensor)

    def forward(self, input: RobosuiteDemoStartingPosition | List | Tensor) -> Tensor:
        return self.encode_to_latent_variable(input=input)

    def __str__(self):
        ret: str = 'EncoderAsActor {'
        ret += '\n\t' + f'neurons: {self.neurons}'
        ret += '\n\t' + f'input_dimension: {self.input_dimension}'
        ret += '\n\t' + f'hidden_neurons: {self.hidden_neurons}'
        ret += '\n\t' + f'latent_variable_dimension: {self.latent_variable_dimension}'
        ret += '\n\t' + f'activation_function: {self.activation_function}'
        ret += '\n\t' + f'net: {str(self.net)}'
        ret += '\n' + '}'
        return ret

    def __repr__(self):
        return f'EncoderAsActor(neurons={self.neurons})'
