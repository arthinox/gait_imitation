# This example requires the installation of myosuite
# pip install myosuite

import time
import pandas as pd
import numpy as np

import myosuite  # noqa
from myosuite.utils import gym

import deprl
from deprl import env_wrappers

# imitate walk train 0
path = "C:/Users/oddly/Documents/senior_research/cig_04_files/imitate_walk_train_5e7/myoLeg/250409.074250/"
# path = "C:/Users/oddly/Documents/senior_research/cig_04_files/baselines_DEPRL_pelv_75_percent/myoLeg/250222.084142/"
export_fname = 'imitate_walk_train_1_obs.csv'

env = gym.make("imitateWalk-v0", reset_type="random")
env = env_wrappers.GymWrapper(env)
policy = deprl.load(path, env)

EP_NUM = 0          # Set episode to record observation for
count = 0
# column_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

env.seed(0)
for ep in range(5):
    ep_steps = 0
    ep_tot_reward = 0
    state = env.reset()

    # Setup dataframe
    if ep == EP_NUM:
        obs_dict = env.get_obs_dict(env.sim)
        keys_row = []                           # Row of keys (strings)
        init_data = []
        for key, value in obs_dict.items():
            if hasattr(value, "__len__"):
                for i, obs_type in enumerate(value):
                    if hasattr(obs_type, "__len__"):
                        for j, row in enumerate(obs_type):
                            keys_row.append(key + "_" + str(i) + "_" + str(j))
                            init_data.append(row)
                    else:
                        keys_row.append(key + "_" + str(i))
                        init_data.append(obs_type)
            else:
                keys_row.append(key)
                init_data.append(value)

        row1 = [init_data]
        df = pd.DataFrame(row1, columns=keys_row)

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
            obs_dict = env.get_obs_dict(env.sim)
            new_row = []
            for key, value in obs_dict.items():
                if hasattr(value, "__len__"):
                    for i, obs_type in enumerate(value):
                        if hasattr(obs_type, "__len__"):
                            for j, row in enumerate(obs_type):
                                new_row.append(row)
                        else:
                            new_row.append(obs_type)
                else:
                    new_row.append(value)
            df.loc[len(df)] = new_row

        # check if done
        if done or (ep_steps >= 1000):

            # Write dataframe to csv
            if 'df' in globals():
                # try:
                #     times = pd.read_csv(export_fname)
                # except FileNotFoundError: 
                    # times = pd.DataFrame(0, index=np.arange(1004), columns=column_names) 
                df.to_csv(export_fname, index=False)
                # times.insert(count, str(count), df['t_0'], True)
                
                # times.to_csv(export_fname, index=False)
                
            count += 1

            print(
                f"Episode {ep} ending; steps={ep_steps}; reward={ep_tot_reward:0.3f};"
            )
            env.reset()
            break
