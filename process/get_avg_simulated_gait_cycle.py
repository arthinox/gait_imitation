from statistics import mean
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

qpos_path = "joint_qpos_16.csv"
out_fname = '16_joint_angles.csv'

# Left knee angle peaks
# Trial 18
# time_markers = [(319,430), (430,544), (544,657), (657,772), (772,882)]
# Trial 19
time_markers = [(216,322),(322,429),(429,547),(547,655),(655,771),(771,883)]

def interpolate_many(lists):
    maxlen = 200
    interpolation_target = np.linspace(0, 1, maxlen)

    for lst in lists:
        x_values = np.linspace(0, 1, len(lst))
        yield np.interp(interpolation_target, x_values, lst)

# df2 = pd.read_csv("C:/Users/oddly/Documents/senior_research/custom_environments/joint_angle_data.csv")
# print(df.columns.tolist())

la_dict = {}
lk_dict = {}
lh_dict = {}

ra_dict = {}
rk_dict = {}
rh_dict = {}

df = pd.read_csv(qpos_path)

lh = df['hip_flexion_l']
rh = df['hip_flexion_r']
lk = df['knee_angle_l']
rk = df['knee_angle_r']
la = df['ankle_angle_l']
ra = df['ankle_angle_r']

# plt.plot(lk)      # Used to find time_markers
# plt.show()

for i, times in enumerate(time_markers):
    la_dict[str(i)] = df['ankle_angle_l'][times[0]:times[1]]
    lk_dict[str(i)] = df['knee_angle_l'][times[0]:times[1]]
    lh_dict[str(i)] = df['hip_flexion_l'][times[0]:times[1]]

    ra_dict[str(i)] = df['ankle_angle_r'][times[0]:times[1]]
    rk_dict[str(i)] = df['knee_angle_r'][times[0]:times[1]]
    rh_dict[str(i)] = df['hip_flexion_r'][times[0]:times[1]]

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

out_df = pd.DataFrame(np.vstack((la_merged, lk_merged, lh_merged, ra_merged, rk_merged, rh_merged)).T,
                      columns=['l_ankle_angle','l_knee_angle','l_hip_angle','r_ankle_angle','r_knee_angle','r_hip_angle'])

out_df.to_csv(out_fname, index=False)

# van der zee
# left indices: start -> 201, 304, 404, 495, 598, end -> 706
# right indices: start -> 160, 257, 361, 463, 560, end -> 661

# DNE
# left indices: 319, 430, 544, 657, 772, 882
# r_idx = np.array([160,257,361,463,560,661])
# new_r_idx = r_idx


fig, axes = plt.subplots(3,2)     
axes[0,0].plot(np.roll(la_merged,150))
axes[1,0].plot(np.roll(lk_merged,150))
axes[2,0].plot(np.roll(lh_merged,150))

axes[0,1].plot(np.roll(ra_merged,50))
axes[1,1].plot(np.roll(rk_merged,50))
axes[2,1].plot(np.roll(rh_merged,50))

axes[0,0].title.set_text('Left Ankle')
axes[1,0].title.set_text('Left Knee')
axes[2,0].title.set_text('Left Hip')

axes[0,1].title.set_text('Right Ankle')
axes[1,1].title.set_text('Right Knee')
axes[2,1].title.set_text('Right Hip')

axes[0,0].set_xticks([])
axes[1,0].set_xticks([])
axes[0,1].set_xticks([])
axes[1,1].set_xticks([])

plt.show()