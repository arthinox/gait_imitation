import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt

smooth_kinematics = 0  # Uses median filter to remove outliers
kernel_size = 5

folder_path = "C:/Users/oddly/Documents/senior_research/downloaded_datasets/dne2_s087_stand_and_walk/openpose/"
out_fname = 'DNE_noisy_joint_angle_data.csv'

# Functions

def get_angle(edge1, edge2, edges, points):
    assert tuple(sorted(edge1)) in edges
    assert tuple(sorted(edge2)) in edges

    v1 = points[edge1[0]]-points[edge1[1]]
    v2 = points[edge2[0]]-points[edge2[1]]
    angle = np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    return angle

def get_angle_atan(edge1, edge2, edges, points):
    assert tuple(sorted(edge1)) in edges
    assert tuple(sorted(edge2)) in edges

    v1 = points[edge1[0]]-points[edge1[1]]
    v2 = points[edge2[0]]-points[edge2[1]]
    
    # Directed angle from first vector to second vector
    angle = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])

    while angle < 0:
        angle += 2*np.pi
    
    while angle > 2*np.pi:
        angle -= 2*np.pi

    return angle

def interpolate_many(lists):
    maxlen = 200
    interpolation_target = np.linspace(0, 1, maxlen)

    for lst in lists:
        x_values = np.linspace(0, 1, len(lst))
        yield np.interp(interpolation_target, x_values, lst)

# Points
# Hip (R): 9, Knee (R): 10, Ankle (R): 11
# Hip (L): 12, Knee (L): 13, Ankle (L): 14

# Angles
# Right ankle: (11, 10), (11, 22)
# Right knee: (10, 9), (10, 11)
# Right hip: (8, 1), (9, 10)
# Left ankle: (14, 13), (14, 19)
# Left knee: (13, 12), (13, 14)
# Left hip: (8, 1), (12, 13)

# Edges as defined in OpenPose
skeleton_edges = {(0, 1), (0, 15), (0, 16), (1, 2), (1, 5), (1, 8), (2, 3), (3, 4),
                  (5, 6), (6, 7), (8, 9), (8, 12), (9, 10), (10, 11), (11, 22), (11, 24),
                  (12, 13), (13, 14), (14, 19), (14, 21), (15, 17), (16, 18), (19, 20), (22, 23)}

# Initialize ndarray
keypoints_in_time = np.zeros((25, 2, 1850))
joint_angles = np.zeros((7, 1850)) # dim 0: frame #, l_ankle, l_knee, l_hip, r_ankle, r_knee, r_hip, dim 1: frame

for i in range(1850):
    num_str = str(i).zfill(12)
    file_str = 'stand_and_walk_rgb_' + num_str + '_keypoints.json'
    with open(folder_path + file_str, 'r', encoding='utf-8') as file:
        data = json.load(file)
        if len(data['people']) > 0:
            keypoints = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1,3)
             # Columns are (x, y, confidence) -> we only care about (x, y)
            kp = keypoints[:,:2]
            keypoints_in_time[:,:,i] = kp
            
            # Remember to apply offsets!
            joint_angles[0,i] = i
            
            joint_angles[1,i] = get_angle_atan((14, 13), (14, 19), skeleton_edges, keypoints)  # left ankle
            joint_angles[2,i] = get_angle_atan((13, 12), (13, 14), skeleton_edges, keypoints)  # left knee
            joint_angles[3,i] = get_angle_atan((8, 1), (12, 13), skeleton_edges, keypoints)    # left hip
            
            joint_angles[4,i] = get_angle_atan((11, 10), (11, 22), skeleton_edges, keypoints)  # right ankle
            joint_angles[5,i] = get_angle_atan((10, 9), (10, 11), skeleton_edges, keypoints)   # right knee
            joint_angles[6,i] = get_angle_atan((8, 1), (9, 10), skeleton_edges, keypoints)     # right hip
  
df = pd.DataFrame(joint_angles.T, columns=['frame','l_ankle', 'l_knee', 'l_hip', 'r_ankle', 'r_knee', 'r_hip'])

# fig, axes = plt.subplots(3,2)
# axes[0,0].plot(df['l_ankle'][87:392])
# axes[1,0].plot(df['l_knee'][87:392])
# axes[2,0].plot(df['l_hip'][87:392])

# axes[0,1].plot(df['r_ankle'][87:392])
# axes[1,1].plot(df['r_knee'][87:392])
# axes[2,1].plot(df['r_hip'][87:392])

# plt.show()

# The plan:
# Take averages of each joint separately across all gait cycles
# Use mean time difference between peaks (of three values) to get offset
# Use offset to merge two sides
# Make sign / angle offset adjustments as necessary 

# 1st set: [87, 392]
# 2nd set: [399, 680]
# 3rd set: [713, 1044]
# 4th set: [1079, 1350]
# 5th set: [1383, 1700]

# Gait cycles: [140:217], [217:303], negative [487:578], [744:839], [839:925], [925:1001],
#                                    negative [1174:1263], [1411:1501], [1501:1593], [1593:1664]

time_markers = [(140,217), (217,303), (487,578), (744,839), (839,925),
                (925,1001), (1174,1263), (1411,1501), (1501,1593), (1593,1664)]

la_dict = {}
lk_dict = {}
lh_dict = {}

ra_dict = {}
rk_dict = {}
rh_dict = {}

if smooth_kinematics:
    for i, times in enumerate(time_markers):
        if (i == 2) or (i == 6):
            la_dict[str(i)] = medfilt(-df['l_ankle'][times[0]:times[1]] + (2*np.pi), kernel_size=kernel_size)
            lk_dict[str(i)] = medfilt(-df['l_knee'][times[0]:times[1]] + (2*np.pi), kernel_size=kernel_size)
            lh_dict[str(i)] = medfilt(-df['l_hip'][times[0]:times[1]] + (2*np.pi), kernel_size=kernel_size)
            
            ra_dict[str(i)] = medfilt(-df['r_ankle'][times[0]:times[1]] + (2*np.pi), kernel_size=kernel_size)
            rk_dict[str(i)] = medfilt(-df['r_knee'][times[0]:times[1]] + (2*np.pi), kernel_size=kernel_size)
            rh_dict[str(i)] = medfilt(-df['r_hip'][times[0]:times[1]] + (2*np.pi), kernel_size=kernel_size)
        else:
            la_dict[str(i)] = medfilt(df['l_ankle'][times[0]:times[1]], kernel_size=kernel_size)
            lk_dict[str(i)] = medfilt(df['l_knee'][times[0]:times[1]], kernel_size=kernel_size)
            lh_dict[str(i)] = medfilt(df['l_hip'][times[0]:times[1]], kernel_size=kernel_size)
            
            ra_dict[str(i)] = medfilt(df['r_ankle'][times[0]:times[1]], kernel_size=kernel_size)
            rk_dict[str(i)] = medfilt(df['r_knee'][times[0]:times[1]], kernel_size=kernel_size)
            rh_dict[str(i)] = medfilt(df['r_hip'][times[0]:times[1]], kernel_size=kernel_size)
else:
    for i, times in enumerate(time_markers):
        if (i == 2) or (i == 6):
            la_dict[str(i)] = -df['l_ankle'][times[0]:times[1]] + (2*np.pi)
            lk_dict[str(i)] = -df['l_knee'][times[0]:times[1]] + (2*np.pi)
            lh_dict[str(i)] = -df['l_hip'][times[0]:times[1]] + (2*np.pi)
            
            ra_dict[str(i)] = -df['r_ankle'][times[0]:times[1]] + (2*np.pi)
            rk_dict[str(i)] = -df['r_knee'][times[0]:times[1]] + (2*np.pi)
            rh_dict[str(i)] = -df['r_hip'][times[0]:times[1]] + (2*np.pi)
        else:
            la_dict[str(i)] = df['l_ankle'][times[0]:times[1]]
            lk_dict[str(i)] = df['l_knee'][times[0]:times[1]]
            lh_dict[str(i)] = df['l_hip'][times[0]:times[1]]
            
            ra_dict[str(i)] = df['r_ankle'][times[0]:times[1]]
            rk_dict[str(i)] = df['r_knee'][times[0]:times[1]]
            rh_dict[str(i)] = df['r_hip'][times[0]:times[1]]

# Interpolate gait sequences for each joint angle
la_interp = interpolate_many(la_dict.values())
la_merged = [np.mean(values) for values in zip(*la_interp)]

lk_interp = interpolate_many(lk_dict.values())
lk_merged = [np.mean(values) for values in zip(*lk_interp)]

lh_interp = interpolate_many(lh_dict.values())
lh_merged = [np.mean(values) for values in zip(*lh_interp)]

ra_interp = interpolate_many(ra_dict.values())
ra_merged = [np.mean(values) for values in zip(*ra_interp)]

rk_interp = interpolate_many(rk_dict.values())
rk_merged = [np.mean(values) for values in zip(*rk_interp)]

rh_interp = interpolate_many(rh_dict.values())
rh_merged = [np.mean(values) for values in zip(*rh_interp)]

# Align right joints with left joints and merge
ankle_merged = np.mean(np.array([la_merged, np.roll(ra_merged,100)]), axis=0)
knee_merged = np.mean(np.array([lk_merged, np.roll(rk_merged,100)]), axis=0)
hip_merged = np.mean(np.array([lh_merged, np.roll(rh_merged,100)]), axis=0)

ankle_merged = np.roll(-ankle_merged + (2*np.pi), 160)
knee_merged = np.roll(knee_merged, 160)
hip_merged = np.roll(-hip_merged + (2*np.pi), 160)

# Apply offset
ankle_merged -= (3*np.pi)/2.0
knee_merged -= np.pi
hip_merged -= np.pi

# Save as .csv
dne_joint_angles = pd.DataFrame(np.array([ankle_merged, knee_merged, hip_merged]).T,
                                columns=['ankle_angle', 'knee_angle', 'hip_angle'])

dne_joint_angles.to_csv(out_fname, index=False)

# Plot
fig, axes = plt.subplots(3,1)
axes[0].plot(hip_merged*57.2958)
axes[1].plot(knee_merged*57.2958)
axes[2].plot(ankle_merged*57.2958)

axes[0].title.set_text('Hip')
axes[1].title.set_text('Knee')
axes[2].title.set_text('Ankle')

axes[0].set_xticks([])
axes[1].set_xticks([])

plt.show()

# fig, axes = plt.subplots(3,2)
# axes[0,0].plot(la_merged)
# axes[1,0].plot(lk_merged)
# axes[2,0].plot(lh_merged)

# axes[0,1].plot(ra_merged)
# axes[1,1].plot(rk_merged)
# axes[2,1].plot(rh_merged)

# axes[0,0].title.set_text('Left Ankle')

# axes[1,0].title.set_text('Left Knee')
# axes[2,0].title.set_text('Left Hip')

# axes[0,1].title.set_text('Right Ankle')
# axes[1,1].title.set_text('Right Knee')
# axes[2,1].title.set_text('Right Hip')

# axes[0,0].set_xticks([])
# axes[1,0].set_xticks([])
# axes[0,1].set_xticks([])
# axes[1,1].set_xticks([])

# plt.show()