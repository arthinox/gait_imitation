import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

xpos_data = pd.read_csv('body_xpos.csv')

# bodies: pelvis, femur, tibia, talus, toes
# Create (timesteps x 3) shape array for each of the above
# pelvis to femur doesn't give true hip angle?

def get_angles(a, b):
    # Takes in array, each row is a vector
    ab_mag = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    dot_products = np.sum(a*b, axis=1)
    return np.arccos(dot_products / ab_mag.astype(float))

pelvis_index = xpos_data.columns.get_loc('pelvis_x')
femur_r_index = xpos_data.columns.get_loc('femur_r_x')
femur_l_index = xpos_data.columns.get_loc('femur_l_x')
tibia_r_index = xpos_data.columns.get_loc('tibia_r_x')
tibia_l_index = xpos_data.columns.get_loc('tibia_l_x')
talus_r_index = xpos_data.columns.get_loc('talus_r_x')
talus_l_index = xpos_data.columns.get_loc('talus_l_x')
toes_r_index = xpos_data.columns.get_loc('toes_r_x')
toes_l_index = xpos_data.columns.get_loc('toes_l_x')

pelvis_data = xpos_data.iloc[:, pelvis_index:(pelvis_index+3)].to_numpy()
femur_r_data = xpos_data.iloc[:, femur_r_index:(femur_r_index+3)].to_numpy()
femur_l_data = xpos_data.iloc[:, femur_l_index:(femur_l_index+3)].to_numpy()
tibia_r_data = xpos_data.iloc[:, tibia_r_index:(tibia_r_index+3)].to_numpy()
tibia_l_data = xpos_data.iloc[:, tibia_l_index:(tibia_l_index+3)].to_numpy()
talus_r_data = xpos_data.iloc[:, talus_r_index:(talus_r_index+3)].to_numpy()
talus_l_data = xpos_data.iloc[:, talus_l_index:(talus_l_index+3)].to_numpy()
toes_r_data = xpos_data.iloc[:, toes_r_index:(toes_r_index+3)].to_numpy()
toes_l_data = xpos_data.iloc[:, toes_l_index:(toes_l_index+3)].to_numpy()

# Vectors
pelvis_to_fem_r = femur_r_data - pelvis_data
fem_r_to_tib_r = tibia_r_data - femur_r_data
tib_r_to_tal_r = talus_r_data - tibia_r_data
tal_r_to_toes_r = toes_r_data - talus_r_data

pelvis_to_fem_l = femur_l_data - pelvis_data
fem_l_to_tib_l = tibia_l_data - femur_l_data
tib_l_to_tal_l = talus_l_data - tibia_l_data
tal_l_to_toes_l = toes_l_data - talus_l_data

# Angles
hip_r_angles = get_angles(fem_r_to_tib_r, -pelvis_to_fem_r)
hip_l_angles = get_angles(fem_l_to_tib_l, -pelvis_to_fem_l)
knee_r_angles = get_angles(tib_r_to_tal_r, -fem_r_to_tib_r)
knee_l_angles = get_angles(tib_l_to_tal_l, -fem_l_to_tib_l)
ankle_r_angles = get_angles(tal_r_to_toes_r, -tib_r_to_tal_r)
ankle_l_angles = get_angles(tal_l_to_toes_l, -tib_l_to_tal_l)

column_names = ['hip_r_angles','hip_l_angles','knee_r_angles','knee_l_angles','ankle_r_angles','ankle_l_angles']
all_angles = np.vstack((hip_r_angles, hip_l_angles, knee_r_angles, knee_l_angles, ankle_r_angles, ankle_l_angles)).T
# Take initial config. as reference angle
# all_angles = all_angles - all_angles[0]

# Compare with corresponding qpos entries


df = pd.DataFrame(all_angles, columns=column_names)
df.to_csv('angles_from_xpos.csv', index=False)