# This example requires the installation of myosuite
# pip install myosuite

import time
import pandas as pd
import numpy as np

import myosuite  # noqa
from myosuite.utils import gym

import deprl
from deprl import env_wrappers

path = "C:/Users/oddly/Documents/senior_research/cig_04_files/imitate_walk_train_14_5e7/myoLeg/250416.062918/"
# path = "C:/Users/oddly/Documents/senior_research/cig_04_files/baselines_DEPRL_pelv_75_percent/myoLeg/250222.084142/"
export_fname = 'imitateWalk14_GRF.csv'

env = gym.make("imitateWalk-v0", reset_type="random")
env = env_wrappers.GymWrapper(env)
policy = deprl.load(path, env)

# env = gym.make("myoLegWalk-v0", reset_type="random")
# env = env_wrappers.GymWrapper(env)
# policy = deprl.load_baseline(env)

# Set by the user
EP_NUM = 0          # Set episode to record observation for
tot_steps = 1001

count = 0

model = env.sim.model
data = env.sim.data

# all_steps = []
# Dictionary to store all contacts. Each key stores a numpy array (1 row -> 1 timestep, 3 columns for 3 forces)
all_contacts = {}

env.seed(0)
for ep in range(10):
    ep_steps = 0
    ep_tot_reward = 0
    state = env.reset()

    # Setup dataframe
    if ep == EP_NUM:
        init_data = []
        contacts = data.contact
        # init_data.append(len(contacts))
        for index, contact in enumerate(contacts):
            # Check if column set exists
            id_1 = int(contact.geom1)
            id_2 = int(contact.geom2)
            includes_foot_geom = (id_2 >= 54 and id_2 <= 60) or (id_2 >= 85 or id_2 <= 91)
            if int(contact.geom1) == 0 and includes_foot_geom:
                geom_name_1 = model.id2name(id_1, 5)
                geom_name_2 = model.id2name(id_2, 5)
                key_1 = geom_name_1 + "_" + geom_name_2 + "_x"
                key_2 = geom_name_1 + "_" + geom_name_2 + "_y"
                key_3 = geom_name_1 + "_" + geom_name_2 + "_z"
                
                # Instantiate arrays
                all_contacts[key_1] = np.zeros(tot_steps)
                all_contacts[key_2] = np.zeros(tot_steps)
                all_contacts[key_3] = np.zeros(tot_steps)
                # import ipdb; ipdb.set_trace()
                
                # Get forces and convert from contact frame to world frame
                contact_xmat = contact.frame.reshape(3,3).T
                forces = contact_xmat @ data.contact_force(index)[0]      # 2 x 3 contact wrench array
                
                # Key -> contact geoms; Index -> timestep
                all_contacts[key_1][0] = forces[0]
                all_contacts[key_2][0] = forces[1]
                all_contacts[key_3][0] = forces[2]

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
            # new_contacts = []
            contacts = data.contact
            # new_contacts.append(len(contacts))
            for index, contact in enumerate(contacts):
                id_1 = int(contact.geom1)
                id_2 = int(contact.geom2)
                includes_foot_geom = (id_2 >= 54 and id_2 <= 60) or (id_2 >= 85 or id_2 <= 91)
                if int(contact.geom1) == 0 and includes_foot_geom:
                    geom_name_1 = model.id2name(id_1, 5)
                    geom_name_2 = model.id2name(id_2, 5)
                    key_1 = geom_name_1 + "_" + geom_name_2 + "_x"
                    key_2 = geom_name_1 + "_" + geom_name_2 + "_y"
                    key_3 = geom_name_1 + "_" + geom_name_2 + "_z"
                    
                    if key_1 not in all_contacts:
                        # Instantiate arrays
                        all_contacts[key_1] = np.zeros(tot_steps)
                        all_contacts[key_2] = np.zeros(tot_steps)
                        all_contacts[key_3] = np.zeros(tot_steps)

                    # Get forces and convert from contact frame to world frame
                    contact_xmat = contact.frame.reshape(3,3).T
                    forces = contact_xmat @ data.contact_force(index)[0]      # 2 x 3 contact wrench array
                    
                    # Key -> contact geoms; Index -> timestep
                    all_contacts[key_1][ep_steps] = forces[0]
                    all_contacts[key_2][ep_steps] = forces[1]
                    all_contacts[key_3][ep_steps] = forces[2]
            
        # check if done
        if done or (ep_steps >= tot_steps - 1):

            # Write dataframe to csv
            if ep == EP_NUM:
                contacts_df = pd.DataFrame(all_contacts)
                # all_contacts = pd.DataFrame([x if isinstance(x, list) else [x] for x in all_steps])
                
                
                contacts_df['r_foot_y'] = (contacts_df['floor_r_bofoot_col1_y'] + contacts_df['floor_r_bofoot_col2_y'] 
                                           + contacts_df['floor_r_foot_col1_y'] + contacts_df['floor_r_foot_col3_y']
                                           + contacts_df['floor_r_foot_col4_y'])
                
                contacts_df['r_foot_z'] = (contacts_df['floor_r_bofoot_col1_z'] + contacts_df['floor_r_bofoot_col2_z'] 
                                           + contacts_df['floor_r_foot_col1_z'] + contacts_df['floor_r_foot_col3_z']
                                           + contacts_df['floor_r_foot_col4_z'])
                
                contacts_df.to_csv(export_fname, index=False)
                

            print(
                f"Episode {ep} ending; steps={ep_steps}; reward={ep_tot_reward:0.3f};"
            )
            env.reset()
            break
