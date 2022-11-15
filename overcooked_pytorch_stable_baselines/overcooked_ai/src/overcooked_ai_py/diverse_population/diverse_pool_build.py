from stable_baselines3.ppo.ppo import PPO
from overcooked_ai.src.overcooked_ai_py.mdp.overcooked_env import OvercookedGridworld, OvercookedEnv, get_vectorized_gym_env

mdp = OvercookedGridworld.from_layout_name("cramped_room")
overcooked_env = OvercookedEnv.from_mdp(mdp, horizon=400)

params = {}
params["RUN_TYPE"] = "joint_ppo"
params["sim_threads"] = 8

args = {}
args["ent_coef_start"] = 0.0
args["ent_coef_horizon"] = 2e6
args["ent_coef_end"] = 0.000

args["sparse_r_coef_horizon"] = 1e6

args["total_timesteps"] = 5e6   # TODO: can i state "with little loss of generality" throughout experiments??

args["action_prob_diff_reward_coef"] = 0.1

args["action_prob_diff_loss_coef"] = 0.1

# TODO: should i study effect of annealing of entropy_coef, sparse_r_coef, learning_rate???, ...
# TODO: should i consider RNN, or at least frame stacking?

if __name__ == "__main__":
    feature_fn = lambda _, state: overcooked_env.featurize_state_mdp(state) # TODO: maybe lossless encoding?
    gym_env = get_vectorized_gym_env(
        overcooked_env, 'Overcooked-v0', agent_idx=0, featurize_fn=feature_fn, **params
    )

    model1 = PPO("MlpPolicy", gym_env, device="cpu", verbose=1, tensorboard_log="./diverse_population/logs/a", n_steps=400, batch_size=400)


    agent_idxs = [0,0,0,0,1,1,1,1] # TODO: do better
    gym_env.remote_set_agent_idx(agent_idxs)
    gym_env.other_agent_model = model1



    population = []
    for _ in range(5):
        population.append(PPO("MlpPolicy", gym_env, device="cpu", verbose=1, tensorboard_log="./diverse_population/logs/a", n_steps=400, batch_size=400))

    gym_env.population = population

    model1.learn(args["total_timesteps"], args=args)

    # model1.save("models/test")
    # model2 = PPO("MlpPolicy", gym_env, device="cpu")

    # model2 = PPO.load("models/test", env=gym_env, device="cpu")
    # model2.learn(10000)

