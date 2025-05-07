import pandas as pd
import numpy as np

ankle = pd.read_csv('ankle_angle_grad_myo.csv', header=None)
knee = pd.read_csv('knee_angle_grad_myo.csv', header=None)
hip = pd.read_csv('hip_angle_grad_myo.csv', header=None)

new_df  = pd.concat([176.99115*ankle, 176.99115*knee, 176.99115*hip], axis=1)

new_df.columns = ['ankle_angle','knee_angle','hip_angle']
new_df.to_csv('joint_angle_grad_data_2.csv', index=False)

# actual_angles = pd.read_csv('C:/Users/oddly/Documents/senior_research/depRL/joint_qpos.csv').to_numpy()
# # print(actual_angles.shape)
# ref_joint_angles = pd.read_csv("C:/Users/oddly/Documents/senior_research/custom_environments/joint_angle_data.csv").to_numpy()

# for i in range(200):
#     step = i
#     phase_var = (step / 100.0) % 1
#     l_index = int(np.floor(phase_var*200))
#     # Right side should have 50% phase difference from left side
#     r_index = int((l_index + 100) % 200)
#     # weights = np.array([self.])
#     target_angles = np.array([ref_joint_angles[r_index,2],ref_joint_angles[r_index,1]], dtype=np.float32)
#     print(np.linalg.norm(target_angles - actual_angles[step,:2]))