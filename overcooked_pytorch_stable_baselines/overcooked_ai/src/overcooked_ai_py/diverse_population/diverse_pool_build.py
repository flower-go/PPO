from stable_baselines3.ppo.ppo import PPO
from overcooked_ai.src.overcooked_ai_py.mdp.overcooked_env import OvercookedGridworld, OvercookedEnv, get_vectorized_gym_env

mdp = OvercookedGridworld.from_layout_name("cramped_room")
overcooked_env = OvercookedEnv.from_mdp(mdp, horizon=400)

params = {}
params["RUN_TYPE"] = "joint_ppo"
params["sim_threads"] = 8

if __name__ == "__main__":
    feature_fn = lambda _, state: overcooked_env.featurize_state_mdp(state) # TODO: maybe lossless encoding?
    gym_env = get_vectorized_gym_env(
        overcooked_env, 'Overcooked-v0', agent_idx=0, featurize_fn=feature_fn, **params
    )
    # gym_env.update_reward_shaping_param(1.0)  # Start reward shaping from 1 # TODO: apply reward shaping wrapper

    model1 = PPO("MlpPolicy", gym_env, device="cpu", verbose=1, tensorboard_log="./diverse_population/logs/a", n_steps=400, batch_size=3200,ent_coef=0.1)

    model2 = PPO("MlpPolicy", gym_env, device="cpu")
    agent_idxs = [0,0,0,0,1,1,1,1]
    gym_env.remote_set_agent_idx(agent_idxs)
    gym_env.other_agent_model = model2

    model1.learn(5000000)    # TODO: it seems to work, now is time to git this basic version and then start trying inject agent/population logic

