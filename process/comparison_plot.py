import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

lk = pd.read_csv('sim_left_knee.csv')
rk = pd.read_csv('sim_right_knee.csv')
la = pd.read_csv('sim_left_ankle.csv')
ra = pd.read_csv('sim_right_ankle.csv')
lh = pd.read_csv('sim_left_hip.csv')
rh = pd.read_csv('sim_right_hip.csv')

ref = pd.read_csv("C:/Users/oddly/Documents/senior_research/custom_environments/joint_angle_data.csv")

fig, (hip, knee, ankle) = plt.subplots(3)
fig.set_figheight(7)
fig.set_figwidth(3.5)

gc = np.linspace(0,100,num=200)
x_labels = ['0%','20%','40%','60%','80%','100%']

hip.set_title('Hip')
# hip.plot(gc, 57.2958*lh, label="Left")
# hip.plot(gc, 57.2958*rh, label="Right")
hip.plot(gc, 57.2958*ref["hip_angle"], label="Reference")
hip.set_ylabel('Angle (deg)')
hip.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
          ncol=3)

knee.set_title('Knee')
# knee.plot(gc, 57.2958*lk, label="Left")
# knee.plot(gc, 57.2958*rk, label="Right")
knee.plot(gc,57.2958*ref["knee_angle"], label="Reference")
knee.set_ylabel('Angle (deg)')

ankle.set_title('Ankle')
# ankle.plot(gc,57.2958*la, label="Left")
# ankle.plot(gc,57.2958*ra, label="Right")
ankle.plot(gc,57.2958*ref["ankle_angle"], label="Reference")
ankle.set_xticks([0, 19, 39, 59, 79, 99], labels=x_labels)
ankle.set_ylabel('Angle (deg)')
ankle.set_xlabel('Gait cycle')

# lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# fig.legend(lines, labels)
# labels = ['Left','Right','Reference']
# fig.legend(labels, loc='lower right', bbox_to_anchor=(1,-0.1), ncol=len(labels), bbox_transform=fig.transFigure)

for ax in fig.get_axes():
    ax.label_outer()
   
plt.show()