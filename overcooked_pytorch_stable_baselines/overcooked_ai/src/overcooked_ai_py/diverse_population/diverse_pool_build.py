import random
from stable_baselines3.ppo.ppo import PPO
import overcooked_ai_py as oai
from overcooked_ai_py.mdp.overcooked_env import OvercookedGridworld, OvercookedEnv, get_vectorized_gym_env
from overcooked_ai_py.agents.agent import AgentPair, AgentFromStableBaselinesPolicy
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from datetime import datetime
from experiments_params import ExperimentsParamsManager
import numpy as np
from visualisation.visualisation import heat_map
from evaluation.evaluation import Evaluator
import sys

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

    directory = "diverse_population/models/" + args["layout_name"] + "/" + args["mode"] + "/"
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
    directory = "diverse_population/models/" + args["map"] + "/" + args["mode"] + "/"
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
                                    tensorboard_log=f"./diverse_population/logs/{args['mode']}/{str(n).zfill(2)}", n_steps=400,
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

def train_model(n, env, args, checkpoint=None):
    now = datetime.now()
    found = False
    print("~~~train model metoda", file=sys.stderr)
    while not found:
        try:
            model = PPO("MlpPolicy", env, device=args["device"], verbose=0,
                        tensorboard_log=f"./diverse_population/logs/{args['mode']}/{str(n).zfill(2)}",
                        n_steps=400,
                        batch_size=400, seed=now.microsecond + now.hour)
            env.other_agent_model = model
            num_steps = args["total_timesteps"]
            print("~~~jdem hledat model", file=sys.stderr)
            model.learn(num_steps, args=args, reset_num_timesteps=False)
            found = True
            print("~~~nalezen model", file=sys.stderr)
            print("nalezeno = " + str(found), file=sys.stderr)
        except:
            print("~~~found divergent solution", file=sys.stderr)
            exit(1)
            sys.stdout.flush()
            found = False

    print("koncime",file=sys.stderr)
    return model

params_manager = ExperimentsParamsManager(args)

mdp = OvercookedGridworld.from_layout_name(params_manager.args["layout_name"])
overcooked_env = OvercookedEnv.from_mdp(mdp, horizon=400)


if __name__ == "__main__":
    print("~~~metoda main")
    sys.stdout.flush()
    print("~~~metoda main e", file=sys.stderr)
    params_manager.set_SP_RS_E0()

    #state functions for env and env
    feature_fn = lambda _, state: overcooked_env.featurize_state_mdp(state)
    start_state_fn = mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh = args["rnd_obj_prob_thresh"]) if args["random_start"] == True else None
    gym_env = get_vectorized_gym_env(
        overcooked_env, 'Overcooked-v0', agent_idx=0, featurize_fn=feature_fn, start_state_fn=start_state_fn, **args
    )

    print("~~~vytvoreno rpostredi e", file=sys.stderr)
    agent_idxs = [ int(x < args["num_workers"] / 2) for x in range(args["num_workers"])]
    gym_env.remote_set_agent_idx(agent_idxs)


    gym_env.population = []

    evaluator = Evaluator(gym_env, args, deterministic=True, device="cpu")

    #TODO neco z toho se asi nepouziva
    #modes = [params_manager.set_SP_RS_E0, params_manager.set_SP_RS_E0_01, params_manager.set_SP_RS_E0_02, params_manager.set_SP_RS_E0_05, params_manager.set_SP_RS_E0_05_Drop]
    modes = [params_manager.set_SP_RS_E0_01_Drop]
   
    print("jdeme trenovat")
    print("jdeme ttrenovat (e)",file=sys.stderr)
    sys.stdout.flush()
    #creation of models if they do not existed
    for mode_fn in modes:
        mode_fn()
        models = load_or_train_models(args, gym_env, None)
    print("~~~jdem na evaluaci", file=sys.stderr)
    #evaluation of previously created models
    for mode_fn in modes:
        mode_fn()
        models = load_or_train_models(args, gym_env, None)
        eval_table = evaluator.evaluate(models, models, 8, args["mode"])
        heat_map(eval_table, args["mode"], args["mode"], args)








