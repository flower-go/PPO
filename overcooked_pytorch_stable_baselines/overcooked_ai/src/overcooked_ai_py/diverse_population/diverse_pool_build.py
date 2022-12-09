import random

from stable_baselines3.ppo.ppo import PPO
from overcooked_ai.src.overcooked_ai_py.mdp.overcooked_env import OvercookedGridworld, OvercookedEnv, get_vectorized_gym_env
from overcooked_ai.src.overcooked_ai_py.agents.agent import AgentPair, AgentFromStableBaselinesPolicy
from overcooked_ai.src.overcooked_ai_py.mdp.actions import Action
from overcooked_ai.src.overcooked_ai_py.agents.benchmarking import AgentEvaluator
from datetime import datetime
from experiments_params import ExperimentsParamsManager, ALL_LAYOUTS
import numpy as np
from visualisation.visualisation import heat_map
from evaluation.evaluation import Evaluator



args = {}

# TODO: should i study effect of annealing of entropy_coef, sparse_r_coef, learning_rate???, ...
# TODO: should i consider RNN, or at least frame stacking? -> no need

def init_gym_env(args):
    mdp = OvercookedGridworld.from_layout_name(args["map"])
    overcooked_env = OvercookedEnv.from_mdp(mdp, horizon=400)
    feature_fn = lambda _, state: overcooked_env.featurize_state_mdp(state) # TODO: maybe lossless encoding?
    gym_env = get_vectorized_gym_env(
        overcooked_env, 'Overcooked-v0', agent_idx=0, featurize_fn=feature_fn
    )

    agent_idxs = [ int(x < args["num_workers"] / 2) for x in range(args["num_workers"])]
    gym_env.remote_set_agent_idx(agent_idxs)

    return gym_env

def load_or_train_models(args, env, checkpoints = [4,7,10]):
    # TODO: assert

    directory = "diverse_population/models/" + args["layout_name"] + "/" + args["exp"] + "/"
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
                                    tensorboard_log=f"./diverse_population/logs/{args['layout_name']}/{args['exp']}/{str(checkpoint) + 'M_' + str(n).zfill(2)}", n_steps=400,
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
                if model is None:
                    model = train_model(n, env, args)
                model.save(model_name)
                print(f"model {model_name} learned")

            models.append(model)

    return models

def load_or_train_population_models(args, env, checkpoints = [4,8]):
    test_state = env.reset()
    assert_set = []
    # TODO: assert

    directory = "diverse_population/models/" + args["map"] + "/" + args["exp"] + "/"
    models = {}
    env.population_mode = True
    for n in range(args["trained_models"]):
        model = None
        if checkpoints is not None:
            for checkpoint in checkpoints:
                try:
                    model_name = directory + str(n).zfill(2)
                    model = PPO.load(model_name, env=gym_env, device="cpu")
                    print(f"model {model_name} loaded")
                except:
                    now = datetime.now()
                    env.population = models
                    if model is None:
                        model = PPO("MlpPolicy", env, device="cpu", verbose=0,
                                    tensorboard_log=f"./diverse_population/logs/{args['exp']}/{str(n).zfill(2)}", n_steps=400,
                                    batch_size=400, seed=now.microsecond + now.hour)

                    env.other_agent_model = model # this will be changed every episode if population is present
                    model.learn(4e6, args=args, reset_num_timesteps=False)
                    model(test_state)

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
                                tensorboard_log=f"./diverse_population/logs/{args['exp']}/{str(n).zfill(2)}",
                                n_steps=400,
                                batch_size=400, seed=now.microsecond + now.hour)
                env.other_agent_model = model
                num_steps = args["total_timesteps"]
                model.learn(num_steps, args=args, reset_num_timesteps=False)
                model.save(model_name)
                print(f"model {model_name} learned")

            models.append(model)

    return models


def train_model(n, env, args, checkpoint=None):
    now = datetime.now()
    found = False
    while not found:
        try:
            model = PPO("CnnPolicy", env, device=args["device"], verbose=0,
                        tensorboard_log=f"./diverse_population/logs/{args['layout_name']}/{args['exp']}/{str(n).zfill(2)}",
                        n_steps=400,
                        batch_size=400, seed=now.microsecond + now.hour,
                        # learning_rate=1e-3, gae_lambda=0.98, clip_range=0.05, max_grad_norm = 0.1
                        )
            env.other_agent_model = model
            num_steps = args["total_timesteps"]
            model.learn(num_steps, args=args, reset_num_timesteps=False)
            found = True
        finally:
            print("found divergent solution")
            found = False

    return model

params_manager = ExperimentsParamsManager(args)

mdp = OvercookedGridworld.from_layout_name(params_manager.args["layout_name"])
overcooked_env = OvercookedEnv.from_mdp(mdp, horizon=400)


if __name__ == "__main__":


    # feature_fn = lambda _, state: overcooked_env.featurize_state_mdp(state)
    feature_fn = lambda _, state: overcooked_env.lossless_state_encoding_mdp(state, debug=False)

    start_state_fn = mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh = args["rnd_obj_prob_thresh"]) if args["random_start"] == True else None
    gym_env = get_vectorized_gym_env(
        overcooked_env, 'Overcooked-v0', agent_idx=0, featurize_fn=feature_fn, start_state_fn=start_state_fn, **args
    )
    agent_idxs = [ int(x < args["num_workers"] / 2) for x in range(args["num_workers"])]
    gym_env.remote_set_agent_idx(agent_idxs)
    gym_env.population = []


    evaluator = Evaluator(gym_env, args, deterministic=True, device="cpu")






    exp = "CNN_SP_E0_01"
    layouts = ALL_LAYOUTS
    layouts = ["cramped_room"]


    for layout in layouts:
        params_manager.args["layout_name"] = layout
        params_manager.init_base_args_for_layout(layout)
        params_manager.init_exp_specific_args(exp)
        models = load_or_train_models(args, gym_env, None)

    for layout in layouts:
        params_manager.args["layout_name"] = layout
        params_manager.init_base_args_for_layout(layout)
        params_manager.init_exp_specific_args(exp)
        models = load_or_train_models(args, gym_env, None)
        eval_table = evaluator.evaluate(models, models, 10, args["exp"])
        heat_map(eval_table, args["exp"], args["exp"], args)








