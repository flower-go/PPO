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
            print(f"Learning {args['layout_name']}/{args['exp']}")
            model = PPO("CnnPolicy", env, device=args["device"], verbose=0,
                        tensorboard_log=f"./diverse_population/logs/{args['layout_name']}/{args['exp']}/{str(n).zfill(2)}",
                        n_steps=args["n_steps"],
                        seed=now.microsecond + now.hour,
                        batch_size=args["batch_size"],
                        n_epochs=args["n_epochs"],
                        learning_rate=args["learning_rate"],
                        gae_lambda=0.98,
                        clip_range=args["clip_range"],
                        max_grad_norm = args["max_grad_norm"],
                        vf_coef=args["vf_coef"],
                        target_kl=0.006
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

    start_state_fn = mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh = args["rnd_obj_prob_thresh"]) if args["random_start"] == True else mdp.get_standard_start_state
    # start_state_fn = mdp.get_standard_start_state

    gym_env = get_vectorized_gym_env(
        overcooked_env, 'Overcooked-v0', agent_idx=0, featurize_fn=feature_fn, start_state_fn=start_state_fn, **args
    )
    agent_idxs = [ int(x < args["num_workers"] / 2) for x in range(args["num_workers"])]
    gym_env.remote_set_agent_idx(agent_idxs)
    gym_env.population = []


    evaluator = Evaluator(gym_env, args, deterministic=True, device="cpu")






    exp = "CNN_CUDA_RS"
    # exp = "CNN_HARL"

    layouts = ALL_LAYOUTS
    layout = "cramped_room"


    def vf_coef(val):
        args["vf_coef"] = val

    def max_grad_norm(val):
        args["max_grad_norm"] = val

    def clip_range(val):
        args["clip_range"] = val

    def learning_rate(val):
        args["learning_rate"] = val

    def batch_size(val):
        args["batch_size"] = val

    def entropy(val):
        args["ent_coef_start"] = val

    modifications = {"VF0_0001": (vf_coef, 0.0001),
                     "VF0_001": (vf_coef, 0.001),
                     # "VF0_01" : (vf_coef, 0.01),
                     "VF0_1"  : (vf_coef, 0.1),
                     "MGN_0_3": (max_grad_norm, 0.3),
                     "MGN_0_1": (max_grad_norm, 0.1),
                     "CR_0_1" : (clip_range, 0.1),
                     "CR_0_05": (clip_range, 0.05),
                     "LR_6_04": (learning_rate, 6e-4),
                     "LR_1_04": (learning_rate, 1e-4),
                     # "BS_100": (batch_size, 100),
                     "BS_400": (batch_size, 400),
                     "BS_800": (batch_size, 800),
                     "E_0_05": (entropy, 0.05),
                     "E_0_2": (entropy, 0.2),

                     }

    def set_random_params():
        args["vf_coef"] = np.round(np.random.uniform(0.0001, 0.5),5)
        args["max_grad_norm"] = np.round(np.random.uniform(0.1,0.5),5)
        args["clip_range"] = np.round(np.random.uniform(0.05,0.2),5)
        args["learning_rate"] = np.round(np.random.uniform(0.00001,0.003),5)
        args["batch_size"] = np.random.choice([800,1600,6400,9600])
        args["ent_coef_start"] = np.round(np.random.uniform(0.01,0.2),5)
        args["ent_coef_end"] = np.random.uniform(0.00, np.max([args["ent_coef_start"], 0.1]))
        # args["ent_coef_start"] = 0.2
        # args["ent_coef_horizon"] = np.random.choice([0.5e6, 1e6, 2e6, 3e6])
        args["ent_coef_horizon"] = np.random.randint(0.5e6,2e6)
        # args["ent_coef_horizon"] = 1.5e5
        args["n_steps"] =  np.random.choice([400,800,1200])
        args["n_epochs"] = np.random.choice([8,10,12])
        args["sparse_r_coef_horizon"] = np.random.randint(2.5e6,5e6)

    def get_name():
        full_name = exp + "_VF" + str(args["vf_coef"])
        full_name = full_name + "_MGN" + str(args["max_grad_norm"])
        full_name = full_name + "_CR" + str(args["clip_range"])
        full_name = full_name + "_LR" + str(args["learning_rate"])
        full_name = full_name + "_ES" + str(args["ent_coef_start"])
        full_name = full_name + "_SRC" + str(args["sparse_r_coef_horizon"])
        full_name = full_name + "_EP" + str(int(args["n_epochs"]))
        full_name = full_name + "_EH" + str(int(args["ent_coef_horizon"]))
        full_name = full_name + "_BS" + str(int(args["batch_size"]))
        full_name = full_name + "_NS" + str(int(args["n_steps"]))
        full_name = full_name + "_NW" + str(args["num_workers"])
        full_name = full_name + "_TS" + str(int(args["total_timesteps"]))

        args["exp"] = full_name


    for _ in range(1):
        params_manager.args["layout_name"] = layout
        params_manager.init_base_args_for_layout(layout)
        params_manager.init_exp_specific_args(exp)
        # set_random_params()
        get_name()
        models = load_or_train_models(args, gym_env, None)
        # exit()

    # for modification_key in modifications.keys():
    #     params_manager.args["layout_name"] = layout
    #     params_manager.init_base_args_for_layout(layout)
    #     params_manager.init_exp_specific_args(exp)
    #     args["exp"] = exp + "_" + modification_key
    #     modifications[modification_key][0](modifications[modification_key][1])
    #     models = load_or_train_models(args, gym_env, None)


    eval_table = evaluator.evaluate(models, models, 2, args["exp"])
    heat_map(eval_table, args["exp"], args["exp"], args)








