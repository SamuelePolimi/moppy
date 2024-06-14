
from deep_promp.encoder_deep_pro_mp import EncoderDeepProMP
from trajectory.trajectory import Trajectory, T
from trajectory.joint_configuration_trajectory_state import JointConfigurationTrajectoryState
a = EncoderDeepProMP(9, [2, 3], 4)

print(a)



tr1 = Trajectory()

pt1 = JointConfigurationTrajectoryState([1, 2, 3, 4, 5, 6, 7], 8, 0.1)
pt2 = JointConfigurationTrajectoryState([10, 11, 12, 13, 14, 15, 16], 17, 0.1)

tr1.add_point(pt1)
tr1.add_point(pt2)

print(tr1.get_points())

a.encode_to_latent_variable(tr1)