import mujoco
import myosuite  # noqa
from myosuite.utils import gym

import deprl
from deprl import env_wrappers

# path = "C:/Users/oddly/Documents/senior_research/cig_04_files/baselines_DEPRL_pelv_half/myoLeg/250217.072319/"
# path = "C:/Users/oddly/Documents/senior_research/cig_04_files/baselines_DEPRL_pelv_75_percent/myoLeg/250222.084142/"

# env = gym.make("myoLegWalk-v0", reset_type="random")
# env = env_wrappers.GymWrapper(env)
# policy = deprl.load(path, env)

model_path = "C:/Users/oddly/Documents/senior_research/depRL/.venv/Lib/site-packages/myosuite/simhive/myo_sim/leg/myolegs.xml"
model0 = mujoco.MjModel.from_xml_path(model_path)
mujoco.mj_printModel(model0, 'myoleg_printedmodel_from_xml.txt')
