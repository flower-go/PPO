from stable_baselines.ppo2.ppo2 import PPO2
from overcooked_ai_py.mdp.overcooked_env import *

mdp = OvercookedGridworld.from_layout_name("cramped_room")
overcooked_env = OvercookedEnv.from_mdp(mdp, horizon=400)

params = {}
params["RUN_TYPE"] == "joint_ppo"

gym_env = get_vectorized_gym_env(
    overcooked_env, 'Overcooked-v0', agent_idx=0, featurize_fn=lambda x: mdp.lossless_state_encoding(x), **params
)
gym_env.update_reward_shaping_param(1.0)  # Start reward shaping from 1