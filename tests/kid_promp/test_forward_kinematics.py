import json

import torch
import json

from moppy.kid_promp.forward_kinematics import forward_kinematics, forward_kinematics_batch


def test_forward_kinematics():
    # expected output:
    # joint config [ 1.4336555 -1.2559798 -1.8131765 -1.9919585 -2.7769492  3.1733205 2.2967923]

    # link0 [0. 0. 0.] [0. 0. 0.] 1.0
    # link1 [0.    0.    0.333] [0.         0.         0.65699466] 0.7538952301455264
    # link2 [0.    0.    0.333] [-0.1584711  -0.68912039  0.06276929] 0.7043152814086875
    # link3 [-0.04107766 -0.29763846  0.43087735] [ 0.58666708  0.03086295 -0.15259722] 0.7947221582160214
    # link4 [ 0.037423   -0.31466087  0.41205889] [ 0.60331682  0.77300487 -0.19400412] -0.028890787157653183
    # link5 [ 0.41359188 -0.31698712  0.29912612] [-0.59138602  0.56361246 -0.32521138] 0.4762784204699823
    # link6 [ 0.41359188 -0.31698712  0.29912612] [0.16984533 0.07870821 0.7648305 ] 0.6164348171858808
    # link7 [ 0.3975476  -0.23165591  0.31344978] [ 0.77197275 -0.262593    0.48692373] -0.31306273005723173
    # racket [ 0.52581558 -0.19978824  0.26727861] [0.81369981 0.05281687 0.56966278] -0.10289461530856192

    dh_parameters = json.load(open("dh_params.json"))

    joint_configurations = torch.tensor(
        [[1.4336555, -1.2559798, -1.8131765, -1.9919585, -2.7769492, 3.1733205, 2.2967923],
         [0, 0, 0, 0, 0, 0, 0]])
    print(forward_kinematics_batch(dh_parameters, joint_configurations))
    pass


def test_fk_not_nan():
    dh_parameters = json.load(open("dh_params.json"))

    for i in range(10000):
        joint_configuration = torch.randn(7)

        pose = forward_kinematics(dh_parameters, joint_configuration)
        assert not torch.isnan(pose).any(), "Pose contains NaN values for joint configuration: " + str(
            joint_configuration)


#test_fk_not_nan()
#test_forward_kinematics()
