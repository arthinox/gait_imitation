# Imitation Learning for Simulating High-Dimensional Human Locomotion
Author: Ethan Ran

## Overview
This repository contains the most relevant files and scripts I created while working on my senior thesis research project, which aimed to use imitation learning to achieve natural-looking gait in a high-dimensional musculoskeletal (MSK) model. Training and simulation were performed using the [DEP-RL](https://deprl.readthedocs.io/en/latest/) and [MyoSuite](https://myosuite.readthedocs.io/en/latest/) libraries, as well as the [MuJoCo physics engine](https://mujoco.readthedocs.io/en/stable/overview.html).

## Directories
### config_files
Example training configuration files for specifying hyperparameters and reward conditions. 
### envs
Custom MyoSuite environment and registration file.
### misc_data
Useful data to help understand the myoLeg MSK model.
### process
Files used to perform two main tasks:
	1. Process motion capture or pose estimation data to be used by reward function
	2. Process simulated angles for gait analysis
### record
Files used to record simulation data.
### reference_kinematics
Reference kinematic data, already processed for use by reward function.
## Installation
(This section is a work in progress.)
