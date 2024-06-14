
from deep_promp.encoder_deep_pro_mp import EncoderDeepProMP
from trajectory.trajectory import Trajectory
from trajectory.joint_configuration_trajectory_state import JointConfigurationTrajectoryState
a = EncoderDeepProMP(9, [2, 3], 4)

print(a)

tr1 = Trajectory()

pt1 = JointConfigurationTrajectoryState(joint_configuration=[1, 2, 3, 4, 5, 6, 7],
                                        gripper_open=1,
                                        time=0.1)
pt2 = JointConfigurationTrajectoryState(joint_configuration=[10, 11, 12, 13, 14, 15, 16],
                                        gripper_open=0,
                                        time=0.1)

tr1.add_point(pt1)
tr1.add_point(pt2)

print(tr1.get_points())

a.encode_to_latent_variable(tr1)
