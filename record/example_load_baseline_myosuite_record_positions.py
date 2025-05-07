# This example requires the installation of myosuite
# pip install myosuite

import time
import pandas as pd
import numpy as np

from myosuite.utils import gym

import deprl
from deprl import env_wrappers

import matplotlib.pyplot as plt

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

path = "C:/Users/oddly/Documents/senior_research/cig_04_files/imitate_walk_train_17_1e8/myoLeg/250420.004857/"
env = gym.make("imitateWalk-v0", reset_type="random")
env = env_wrappers.GymWrapper(env)
policy = deprl.load(path, env)

# Set by the user
EP_NUM = 0          # Set episode to record observation for
tot_steps = 1001
# inspect_step = 1000
csv_fname1 = 'body_xpos_17.csv'
csv_fname2 = 'joint_qpos_17.csv'

count = 0

model = env.sim.model
data = env.sim.data

xpos_dict = {}
qpos_dict = {}
labels = []
qpos_labels = ['hip_flexion_r','knee_angle_r','ankle_angle_r','hip_flexion_l','knee_angle_l','ankle_angle_l']

env.seed(0)
for ep in range(10):
    ep_steps = 0
    ep_tot_reward = 0
    state = env.reset()

    # Setup dataframe
    if ep == EP_NUM:
        # xpos
        for i in range(data.xpos.shape[0]):
            body = model.id2name(i, 1)
            key_x = body + "_x"
            key_y = body + "_y"
            key_z = body + "_z"
            
            if i > 0:
                labels.append(body)

            # Instantiate arrays
            xpos_dict[key_x] = np.zeros(tot_steps)
            xpos_dict[key_y] = np.zeros(tot_steps)
            xpos_dict[key_z] = np.zeros(tot_steps)
            
            xpos_dict[key_x][0] = data.xpos[i][0]
            xpos_dict[key_y][0] = data.xpos[i][1]
            xpos_dict[key_z][0] = data.xpos[i][2]

        # qpos
        for i in range(len(qpos_labels)):
            id = model.jnt_qposadr[model.name2id(qpos_labels[i], 3)]
            qpos_dict[qpos_labels[i]] = np.zeros(tot_steps)
            qpos_dict[qpos_labels[i]][0] = data.qpos[id]
            
    while True:
        # samples random action
        action = policy(state)
        # applies action and advances environment by one step
        state, reward, done, info = env.step(action)

        ep_steps += 1
        ep_tot_reward += reward
        env.mj_render()
        time.sleep(0.01)

        # Update dataframe
        
        if ep == EP_NUM:
            if ep_steps == 125:
                data.qpos = model.qpos0
            
            for i in range(data.xpos.shape[0]):
                body = model.id2name(i, 1)
                key_x = body + "_x"
                key_y = body + "_y"
                key_z = body + "_z"
                
                xpos_dict[key_x][ep_steps] = data.xpos[i][0]
                xpos_dict[key_y][ep_steps] = data.xpos[i][1]
                xpos_dict[key_z][ep_steps] = data.xpos[i][2]
            
            # qpos
            for i in range(len(qpos_labels)):
                id = model.jnt_qposadr[model.name2id(qpos_labels[i], 3)]
                qpos_dict[qpos_labels[i]][ep_steps] = data.qpos[id]
            
        # check if done
        if done or (ep_steps >= tot_steps - 1):

            # Write dataframe to csv
            if ep == EP_NUM:
                pd.DataFrame(xpos_dict).to_csv(csv_fname1, index=False)
                pd.DataFrame(qpos_dict).to_csv(csv_fname2, index=False)
                
                # Plot xpos of all bodies at inspect_step
                # positions_from_csv = np.genfromtxt(csv_fname1, delimiter=',')
                # num_bodies = int(positions_from_csv.shape[1] / 3)
                
                # x_coord = np.zeros(num_bodies - 1)
                # y_coord = np.zeros(num_bodies - 1)
                # z_coord = np.zeros(num_bodies - 1)
                
                # for i in range(1, num_bodies):          # Ignore world body
                #     x_index = 3*i
                #     x_coord[i - 1] = positions_from_csv[inspect_step][x_index]
                #     y_coord[i - 1] = positions_from_csv[inspect_step][x_index+1]
                #     z_coord[i - 1] = positions_from_csv[inspect_step][x_index+2]
                
                
                # fig = plt.figure()
                # xpos_plot = fig.add_subplot(111, projection='3d')
                # xpos_plot.set_box_aspect([1.0, 1.0, 1.0])

                # # Plot the points
                # xpos_plot.scatter(x_coord, y_coord, z_coord, c='r', marker='o')

                # # Label points
                # for i, label in enumerate(labels):
                #     xpos_plot.text(x_coord[i], y_coord[i], z_coord[i], label)
                    
                # # Set labels for axes
                # xpos_plot.set_xlabel('X Axis')
                # xpos_plot.set_ylabel('Y Axis')
                # xpos_plot.set_zlabel('Z Axis')
                
                # set_axes_equal(xpos_plot)
                # plt.show()
            
            print(
                f"Episode {ep} ending; steps={ep_steps}; reward={ep_tot_reward:0.3f};"
            )
            env.reset()
            break
