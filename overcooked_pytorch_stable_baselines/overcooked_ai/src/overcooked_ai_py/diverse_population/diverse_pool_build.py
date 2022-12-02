import random

from stable_baselines3.ppo.ppo import PPO
from overcooked_ai.src.overcooked_ai_py.mdp.overcooked_env import OvercookedGridworld, OvercookedEnv, get_vectorized_gym_env
from overcooked_ai.src.overcooked_ai_py.agents.agent import AgentPair, AgentFromStableBaselinesPolicy
from overcooked_ai.src.overcooked_ai_py.mdp.actions import Action
from overcooked_ai.src.overcooked_ai_py.agents.benchmarking import AgentEvaluator
from datetime import datetime
from experiments_params import ExperimentsParamsHolder
import numpy as np
from visualisation.visualisation import heat_map



args = {}
args["num_workers"] = 2
args["map"] = "cramped_room"
args["trained_models"] = 10

args["ent_coef_start"] = 0.0
args["ent_coef_horizon"] = 2e6
args["ent_coef_end"] = 0.000

args["sparse_r_coef_horizon"] = 1e6

args["total_timesteps"] = 3e6   # TODO: can i state "with little loss of generality" throughout experiments??

args["action_prob_diff_reward_coef"] = 0.0

args["action_prob_diff_loss_coef"] = 0.0

# TODO: should i study effect of annealing of entropy_coef, sparse_r_coef, learning_rate???, ...
# TODO: should i consider RNN, or at least frame stacking? -> no need

def init_gym_env(args):
    mdp = OvercookedGridworld.from_layout_name(args["map"])
    overcooked_env = OvercookedEnv.from_mdp(mdp, horizon=400)
    feature_fn = lambda _, state: overcooked_env.featurize_state_mdp(state) # TODO: maybe lossless encoding?
    gym_env = get_vectorized_gym_env(
        overcooked_env, 'Overcooked-v0', agent_idx=0, featurize_fn=feature_fn, **params
    )

    agent_idxs = [ int(x < args["num_workers"] / 2) for x in range(args["num_workers"])]
    gym_env.remote_set_agent_idx(agent_idxs)

    return gym_env

def load_or_train_models(args, env, checkpoints = [4,7,10]):
    # TODO: assert

    directory = "diverse_population/models/" + args["map"] + "/" + args["mode"] + "/"
    models = {} if checkpoints is not None else []
    env.population_mode = False
    for n in range(args["trained_models"]):
        model = None
        if checkpoints is not None:
            for checkpoint in [4,7,10]:
                try:
                    model_name = directory + str(checkpoint) + 'M_' + str(n).zfill(2)
                    model = PPO.load(model_name, env=env, device="cpu")
                    print(f"model {model_name} loaded")
                except:
                    now = datetime.now()
                    if model is None:
                        model = PPO("MlpPolicy", env, device="cpu", verbose=0,
                                    tensorboard_log=f"./diverse_population/logs/{args['mode']}/{str(checkpoint) + 'M_' + str(n).zfill(2)}", n_steps=400,
                                    batch_size=400, seed=now.microsecond + now.hour)
                    env.other_agent_model = model
                    if checkpoint == 4:
                        num_steps = 4e6
                    else:
                        num_steps = 3e6
                    model.learn(num_steps, args=args, reset_num_timesteps=False)
                    model.save(model_name)
                    print(f"model {model_name} learned")

                if checkpoint not in models:
                    models[checkpoint] = []
                models[checkpoint].append(model)

        else:
            try:
                model_name = directory + str(n).zfill(2)
                model = PPO.load(model_name, env=env, device="cpu")
                print(f"model {model_name} loaded")
            except:
                now = datetime.now()
                if model is None:
                    model = PPO("MlpPolicy", env, device="cpu", verbose=0,
                                tensorboard_log=f"./diverse_population/logs/{args['mode']}/{str(n).zfill(2)}",
                                n_steps=400,
                                batch_size=400, seed=now.microsecond + now.hour)
                env.other_agent_model = model
                num_steps = args["total_timesteps"]
                model.learn(num_steps, args=args, reset_num_timesteps=False)
                model.save(model_name)
                print(f"model {model_name} learned")

            models.append(model)

    return models

def load_or_train_population_models(args, env):
    # TODO: assert

    directory = "diverse_population/models/" + args["map"] + "/" + args["mode"] + "/"
    models = {}
    env.population_mode = True
    for n in range(args["trained_models"]):
        model = None
        for checkpoint in [4,8]:
            try:
                model_name = directory + str(checkpoint) + 'M_' + str(n).zfill(2)
                model = PPO.load(model_name, env=gym_env, device="cpu")
                print(f"model {model_name} loaded")
            except:
                now = datetime.now()
                env.population = models
                if model is None:
                    model = PPO("MlpPolicy", env, device="cpu", verbose=0,
                                tensorboard_log=f"./diverse_population/logs/{args['mode']}/{str(checkpoint) + 'M_' + str(n).zfill(2)}", n_steps=400,
                                batch_size=400, seed=now.microsecond + now.hour)

                env.population.append(model)

                env.other_agent_model = model # this will be changed every episode if population is present
                model.learn(4e6, args=args, reset_num_timesteps=False)

                model.save(model_name)
                print(f"model {model_name} learned")

            if checkpoint not in models:
                models[checkpoint] = []
            models[checkpoint].append(model)

    return models

def evaluate_agent_sets(set_0, set_1, start_state_fn, feature_fn, params_holder, num_games = 40, file_name=None, deterministic=True):
    evaluator = AgentEvaluator.from_layout_name(params_holder.args, {"horizon": 400, "start_state_fn": start_state_fn})
    global_returns = []

    result_matrix = np.zeros((len(set_0), len(set_1)))
    for i in range(len(set_0)):
        for j in range(len(set_1)):
            agent0 = AgentFromStableBaselinesPolicy(set_0[i].policy, feature_fn, deterministic=deterministic)
            agent1 = AgentFromStableBaselinesPolicy(set_1[j].policy, feature_fn, deterministic=deterministic)
            pair = AgentPair(agent0, agent1)

            agent0.agent_index = 0
            agent1.agent_index = 1


            evaluation = evaluator.evaluate_agent_pair(pair, num_games // 2)
            avg_returns0 = np.mean(evaluation["ep_returns"])
            # print(avg_returns0)
            # return

            agent0.agent_index = 1
            agent1.agent_index = 0

            # print(agent0.agent_index)

            evaluation = evaluator.evaluate_agent_pair(pair, num_games // 2)
            # print(agent0.agent_index)
            # return
            avg_returns1 = np.mean(evaluation["ep_returns"])
            # print(avg_returns1)

            result_matrix[i,j] = np.mean([avg_returns0, avg_returns1])

            # return


        # ep_final_infos = [ep_info[-1]["episode"] for ep_info in evaluation["ep_infos"]]

    if file_name is not None:
        np.savetxt("diverse_population/evaluation_results/" + file_name, np.array(result_matrix))
    return np.array(result_matrix)

params_holder = ExperimentsParamsHolder(args)

mdp = OvercookedGridworld.from_layout_name(params_holder.args["layout_name"])
overcooked_env = OvercookedEnv.from_mdp(mdp, horizon=400)


if __name__ == "__main__":
    # gym_env = init_gym_env(args)

    feature_fn = lambda _, state: overcooked_env.featurize_state_mdp(state)
    start_state_fn = mdp.get_random_start_state_fn(random_start_pos=True)
    gym_env = get_vectorized_gym_env(
        overcooked_env, 'Overcooked-v0', agent_idx=0, featurize_fn=feature_fn, start_state_fn=start_state_fn, **args
    )
    agent_idxs = [ int(x < args["num_workers"] / 2) for x in range(args["num_workers"])]
    gym_env.remote_set_agent_idx(agent_idxs)
    gym_env.population = []




    # default_population_BU_models = load_or_train_population_models(args, gym_env)
    params_holder.set_default_population_BU_RS()
    # default_BU_RS_models = load_or_train_population_models(args, gym_env)

    params_holder.set_default_SP_RS()
    pretrained_entropy_SP_RS = load_or_train_models(args, gym_env)


    models = pretrained_entropy_SP_RS[4]

    # eval_name = "4M_deterministic_radnom_start_Eval_entropy_SP_RS"
    # eval_table = evaluate_agent_sets(models, models, start_state_fn, feature_fn, params_holder, num_games=30, file_name="cramped_room/default_SP_RS/" + eval_name, deterministic=True)
    # heat_map(eval_table, eval_name, eval_name)


    # eval_name = "4M_stochastic_radnom_start_Eval_entropy_SP_RS"
    # eval_table = evaluate_agent_sets(models, models, start_state_fn, feature_fn, params_holder, num_games=30, file_name="cramped_room/default_SP_RS/" + eval_name, deterministic=False)
    # # eval_table = np.loadtxt("diverse_population/evaluation_results/cramped_room/default_population_BU_RS/" + eval_name).transpose()
    # heat_map(eval_table, eval_name, eval_name)

    models = pretrained_entropy_SP_RS[10]

    # eval_name = "10M_deterministic_radnom_start_Eval_entropy_SP_RS"
    # eval_table = evaluate_agent_sets(models, models, start_state_fn, feature_fn, params_holder, num_games=30, file_name="cramped_room/default_SP_RS/" + eval_name, deterministic=True)
    # # eval_table = np.loadtxt("diverse_population/evaluation_results/cramped_room/default_population_BU_RS/" + eval_name).transpose()
    # heat_map(eval_table, eval_name, eval_name)





    eval_name = "10M_stochastic_radnom_start_Eval_entropy_SP_RS"
    eval_table = evaluate_agent_sets(models, models, start_state_fn, feature_fn, params_holder,
                                     num_games=30, file_name="cramped_room/default_SP_RS/" + eval_name,
                                     deterministic=False)
    # eval_table = np.loadtxt("diverse_population/evaluation_results/cramped_room/default_SP_RS/" + eval_name).transpose()
    heat_map(eval_table, eval_name, eval_name)
    #
    # eval_name = "8M_stochastic_radnom_start_Eval_entropy_SP_set"
    # # eval_table = evaluate_agent_sets(models, pretrained_entropy_SP, start_state_fn, feature_fn, params_holder,
    # #                                  num_games=60, file_name="cramped_room/default_population_BU_RS/" + eval_name,
    # #                                  deterministic=False)
    # eval_table = np.loadtxt("diverse_population/evaluation_results/cramped_room/default_SP_RS/" + eval_name).transpose()
    # heat_map(eval_table, eval_name, eval_name)




    # args["mode"] = "R0d1_population_BU"
    # args["ent_coef_start"] = 0.1
    # args["action_prob_diff_reward_coef"] = 0.1
    # args["ent_coef_end"] = 0.01
    #
    # load_or_train_population_models(args, gym_env)
    #
    # args["mode"] = "R0d2_population_BU"
    # args["ent_coef_start"] = 0.1
    # args["action_prob_diff_reward_coef"] = 0.2
    # args["ent_coef_end"] = 0.01
    #
    # load_or_train_population_models(args, gym_env)
    #
    # args["mode"] = "R0d4_population_BU"
    # args["ent_coef_start"] = 0.1
    # args["action_prob_diff_reward_coef"] = 0.4
    # args["ent_coef_end"] = 0.01
    #
    # load_or_train_population_models(args, gym_env)
    #
    # args["mode"] = "R0d8_population_BU"
    # args["ent_coef_start"] = 0.1
    # args["action_prob_diff_reward_coef"] = 0.8
    # args["ent_coef_end"] = 0.01
    #
    # load_or_train_population_models(args, gym_env)
    #
    # args["mode"] = "R1d6_population_BU"
    # args["ent_coef_start"] = 0.1
    # args["action_prob_diff_reward_coef"] = 1.6
    # args["ent_coef_end"] = 0.01
    #
    # load_or_train_population_models(args, gym_env)


