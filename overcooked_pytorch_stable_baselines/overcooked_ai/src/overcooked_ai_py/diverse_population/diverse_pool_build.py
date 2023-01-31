import sys
import os

codedir = os.environ["CODEDIR"]
#codedir = /home/premek/DP/
projdir = os.environ["PROJDIR"]
#projdir = /home/premek/DP/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_pytorch
sys.path.append(codedir + "/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src")
sys.path.append(codedir + "/PPO/overcooked_pytorch_stable_baselines/stable-baselines3")
sys.path.append(codedir + "/PPO/overcooked_pytorch_stable_baselines")
# print(sys.path)
# exit()
from stable_baselines3.ppo.ppo import PPO
from overcooked_ai.src.overcooked_ai_py.mdp.overcooked_env import OvercookedGridworld, OvercookedEnv, get_vectorized_gym_env
from datetime import datetime
from experiments_params import ExperimentsParamsManager, ALL_LAYOUTS
import numpy as np
from visualisation.visualisation import heat_map
from evaluation.evaluation import Evaluator
import DivergentSolutionException
from divergent_solution_exception import divergent_solution_exception

EVAL_SET_SIZE = 30


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--layout_name", default="forced_coordination", type=str, help="Layout name.")
parser.add_argument("--trained_models", default=8, type=int, help="Number of models to train in experiment.")
parser.add_argument("--mode", default="POP", type=str, help="Mode of experiment: Self-play ('SP') or Population ('POP').")
parser.add_argument("--kl_diff_reward_coef", default=0.0, type=float, help="Coeficient for kl div population policies difference.")
parser.add_argument("--kl_diff_reward_clip", default=0.0, type=float, help="")
parser.add_argument("--cross_entropy_loss_coef", default=0., type=float, help="Coeficient for cross-entropy loss of population policies.")
parser.add_argument("--delay_shared_reward", default=False, action="store_true", help="Whether to delay shared rewards.")
parser.add_argument("--pop_bonus_ts", default=1e5, type=int, help="Number of bonus train time steps for each consecutive individual in population.")
parser.add_argument("--exp", default="CNN_CUDA_RS", type=str, help="Experiment name.")
parser.add_argument("--ent_coef_start", default=0.1, type=float, help="Coeficient for cross-entropy loss of population policies.")
parser.add_argument("--ent_coef_end", default=0.03, type=float, help="Coeficient for cross-entropy loss of population policies.")
parser.add_argument("--ent_coef_horizon", default=1.5e6, type=int, help="Coeficient for cross-entropy loss of population policies.")
parser.add_argument("--total_timesteps", default=5.5e6, type=int, help="Coeficient for cross-entropy loss of population policies.")
parser.add_argument("--vf_coef", default=0.1, type=float, help="Coeficient for cross-entropy loss of population policies.")
parser.add_argument("--batch_size", default=2000, type=int, help="Coeficient for cross-entropy loss of population policies.")
parser.add_argument("--device", default="cuda", type=str, help="Coeficient for cross-entropy loss of population policies.")
parser.add_argument("--max_grad_norm", default=0.3, type=float, help="Coeficient for cross-entropy loss of population policies.")
parser.add_argument("--clip_range", default=0.1, type=float, help="Coeficient for cross-entropy loss of population policies.")
parser.add_argument("--learning_rate", default=0.0004, type=float, help="Coeficient for cross-entropy loss of population policies.")
parser.add_argument("--n_steps", default=400, type=int, help="Coeficient for cross-entropy loss of population policies.")
parser.add_argument("--n_epochs", default=8, type=int, help="Coeficient for cross-entropy loss of population policies.")
parser.add_argument("--shaped_r_coef_horizon", default=2.5e6, type=int, help="Annealing horizont for shaped partial rewards")
parser.add_argument("--divergent_check_timestep", default=3e6, type=int, help="Coeficient for cross-entropy loss of population policies.")
parser.add_argument("--training_percent_start_eval", default=0.0, type=float, help="Coeficient for cross-entropy loss of population policies.")
parser.add_argument("--init_SP_agents", default=1, type=int, help="Number of self-play agents trained to initialize population.")
parser.add_argument("--pop_train_play_argmax", default=False, action="store_true", help="Whether trained partners from population play argmax for episodes sampling")



args = parser.parse_args([] if "__file__" not in globals() else None)
# args = {}

# TODO: should i study effect of annealing of entropy_coef, sparse_r_coef, learning_rate???, ... values found such that works for all maps

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

def load_or_train_models(args, env):
    directory = projdir + "/diverse_population/models/" + args.layout_name + "/" + args.exp + "/"
    models = []
    env.population_mode = False
    for n in range(args.trained_models):
        model = None
        try:
            model_name = directory + str(n).zfill(2)
            print(f"Looking for file {model_name}")
            model = PPO.load(model_name, env=env, device="cuda")
            model.custom_id = n
            print(f"model {model_name} loaded")
        except:
            if model is None:
                model = train_model(n, env, args)
            model.save(model_name)
            print(f"model {model_name} learned")

        models.append(model)

        # First init_SP_agents models are always self-play
        if (n + 1) >= args.init_SP_agents:
            env.population_mode = args.mode == "POP"
        env.population = models

    return models

def train_model(n, env, args):
    now = datetime.now()
    found = False
    while not found:
        try:
            print(f"Learning {args.layout_name}/{args.exp}")
            model = PPO("CnnPolicy", env, device=args.device, verbose=0,
                        tensorboard_log=f"./diverse_population/logs/{args.layout_name}/{args.exp}/{str(n).zfill(2)}",
                        n_steps=args.n_steps,
                        seed=now.microsecond + now.hour,
                        batch_size=args.batch_size,
                        n_epochs=args.n_epochs,
                        learning_rate=args.learning_rate,
                        gae_lambda=0.98,
                        clip_range=args.clip_range,
                        max_grad_norm = args.max_grad_norm,
                        vf_coef=args.vf_coef
                        )
            model.custom_id = n
            env.other_agent_model = model
            num_steps = args.total_timesteps
            num_steps += n * args.pop_bonus_ts
            model.learn(num_steps, args=args, reset_num_timesteps=False)
            found = True
        except divergent_solution_exception.divergentSolutionException:
            print("found divergent solution")
            found = False

    return model

params_manager = ExperimentsParamsManager(args)

mdp = OvercookedGridworld.from_layout_name(params_manager.args.layout_name)
overcooked_env = OvercookedEnv.from_mdp(mdp, horizon=400)


if __name__ == "__main__":

    feature_fn = lambda _, state: overcooked_env.lossless_state_encoding_mdp(state, debug=False)
    start_state_fn = mdp.get_random_start_state_fn(random_start_pos=True, rnd_obj_prob_thresh = args.rnd_obj_prob_thresh) if args.random_start == True else mdp.get_standard_start_state
    gym_env = get_vectorized_gym_env(
        overcooked_env, 'Overcooked-v0', agent_idx=0, featurize_fn=feature_fn, start_state_fn=start_state_fn, args=args
    )
    agent_idxs = [0 for _ in range(args.num_workers)]
    gym_env.remote_set_agent_idx(agent_idxs)
    gym_env.population = []


    evaluator = Evaluator(gym_env, args, deterministic=True, device="cpu")

    def get_name(sp=False, extended=False):
        full_name = args.exp
        if sp:
            full_name = full_name + "_ROP" + str(args.rnd_obj_prob_thresh)
        else:
            if extended:
                full_name = full_name + "_VF" + str(args["vf_coef"])
                full_name = full_name + "_MGN" + str(args["max_grad_norm"])
                full_name = full_name + "_CR" + str(args["clip_range"])
                full_name = full_name + "_LR" + str(args["learning_rate"])
                full_name = full_name + "_ES" + str(args["ent_coef_start"])
                full_name = full_name + "_EE" + str(args["ent_coef_end"])
                full_name = full_name + "_SRC" + str(args["sparse_r_coef_horizon"])
                full_name = full_name + "_EP" + str(int(args["n_epochs"]))
                full_name = full_name + "_EH" + str(int(args["ent_coef_horizon"]))
                full_name = full_name + "_BS" + str(int(args["batch_size"]))
                full_name = full_name + "_NS" + str(int(args["n_steps"]))
                full_name = full_name + "_NW" + str(args["num_workers"])
                full_name = full_name + "_TS" + str(int(args["total_timesteps"]))
            full_name = full_name + "_ROP" + str(args.rnd_obj_prob_thresh)
            full_name = full_name + "_M" + str(args.mode)
            full_name = full_name + "_DR" + str(args.kl_diff_reward_coef)
            full_name = full_name + "_DRC" + str(args.kl_diff_reward_clip)
            full_name = full_name + "_DL" + str(args.cross_entropy_loss_coef)
            full_name = full_name + "_DSR" + str(args.delay_shared_reward)
        args.exp = full_name



    params_manager.args.layout_name = args.layout_name
    params_manager.init_base_args_for_layout(args.layout_name)
    # params_manager.init_exp_specific_args(args.exp)
    get_name()
    models = load_or_train_models(args, gym_env)

    if args.mode == "POP":
        population_name = args.exp
        args.exp = "SP_EVAL"
        get_name(sp=True)
        args.trained_models = EVAL_SET_SIZE
        eval_models = load_or_train_models(args, gym_env)
        eval_table = evaluator.evaluate(models, eval_models, 1, args.layout_name, population_name)
        heat_map(eval_table, population_name, population_name, args.layout_name)
    else:
        eval_table = evaluator.evaluate(models, models, 1, args.layout_name, args.exp)
        heat_map(eval_table, args.exp, args.exp, args.layout_name)
