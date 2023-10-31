import copy
import random
import sys
import os

codedir = os.environ["CODEDIR"]
#codedir = /home/premek/DP/
projdir = os.environ["PROJDIR"]
#projdir = /home/premek/DP/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_pytorch
sys.path.append(codedir + "/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src")
sys.path.append(codedir + "/PPO/overcooked_pytorch_stable_baselines/stable-baselines3")
sys.path.append(codedir + "/PPO/overcooked_pytorch_stable_baselines")

from stable_baselines3.ppo.ppo import PPO
from overcooked_ai.src.overcooked_ai_py.mdp.overcooked_env import OvercookedGridworld, OvercookedEnv, get_vectorized_gym_env
from experiments_params import set_layout_params
from visualisation.visualisation import heat_map
from evaluation.evaluation import Evaluator
from divergent_solution_exception import divergent_solution_exception
os.environ['CUDA_VISIBLE_DEVICES'] = "0"   #Due to Metacentrum computing reasons


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--layout_name", default="forced_coordination", type=str, help="Layout name ('forced_coordination','cramped_room', 'counter_circuit_o_1order', 'coordination_ring'")
parser.add_argument("--trained_models", default=11, type=int, help="Number of models to train during run")
parser.add_argument("--init_SP_agents", default=3, type=int, help="Number of self-play agents trained to initialize population")
parser.add_argument("--eval_agents", default=30, type=int, help="Number of agents from evaluation set used for evaluation")
parser.add_argument("--mode", default="POP", type=str, help="Mode of experiment: Self-play ('SP') or Population ('POP')")
parser.add_argument("--eval_mode", default="SP", type=str, help="Mode used by eval set: Self-play ('SP') or Population ('POP')")
parser.add_argument("--kl_diff_bonus_reward_coef", default=0.0, type=float, help="Coeficient for KL div population policies difference bonus")
parser.add_argument("--kl_diff_bonus_reward_clip", default=0.0, type=float, help="Clipping value for KL div population policies difference bonus")
parser.add_argument("--kl_diff_loss_coef", default=0., type=float, help="Coeficient for KL div term of population policies")
parser.add_argument("--kl_diff_loss_clip", default=0., type=float, help="Clipping value for KL div term of population policies")
parser.add_argument("--delay_shared_reward", default=False, action="store_true", help="Whether to delay shared rewards during early phase")
parser.add_argument("--exp", default="POP_SP_INIT", type=str, help="Experiment name")
parser.add_argument("--eval_set_name", default="SP_EVAL2_ROP0.0", type=str, help="Exact name of evaluation set")
parser.add_argument("--base_eval_name", default="SP_EVAL2_ROP0.0", type=str, help="Base name of evaluation set")
parser.add_argument("--execute_final_eval", default=False, action="store_true", help="Whether to do final evaluation")
parser.add_argument("--final_eval_games_per_worker", default=5, type=int, help="Number of games per worker for pair in final evaluation")
parser.add_argument("--n_sample_partners", default=-1, type=int, help="Number of sampled partners from population for data collection")
parser.add_argument("--frame_stacking", default=4, type=int, help="Number of frames stacked considered for temporal information")
parser.add_argument("--frame_stacking_mode", default="channels", type=str, help="Whether to stack previous frames as other channels ('channels'), or as tuple of individual states ('tuple').")



parser.add_argument("--partner_action_deterministic", default=False, action="store_true", help="Whether trained partners from population play argmax for episodes sampling")
parser.add_argument("--random_switch_start_pos", default=False, action="store_true", help="Whether initial player locations are switched")
parser.add_argument("--rnd_obj_prob_thresh_env", default=0.0, type=float, help="Random object generation probability of environment on environment reset")
parser.add_argument("--rnd_obj_prob_thresh_agent", default=0.0, type=float, help="Random object generation probability during training on environment reset")
parser.add_argument("--static_start", default=False, action="store_true", help="Initial positions are the same on every reset")

# Now moreless fixed training hyperparameters
parser.add_argument("--ent_coef_start", default=0.1, type=float, help="Coeficient for entropy bonus term at the start")
parser.add_argument("--ent_coef_end", default=0.03, type=float, help="Coeficient for entropy bonus term after ent_coef_horizon steps")
parser.add_argument("--ent_coef_horizon", default=1.5e6, type=int, help="Number of steps for annealing entropy bonus coeficient")
parser.add_argument("--total_timesteps", default=5.5e6, type=int, help="Total number of environment steps during training")
parser.add_argument("--vf_coef", default=0.1, type=float, help="Coeficient for critic term in PPO objective")
parser.add_argument("--batch_size", default=2000, type=int, help="Batch size")
parser.add_argument("--max_grad_norm", default=0.3, type=float, help="Maximal gradient norm value")
parser.add_argument("--clip_range", default=0.1, type=float, help="Clipping range")
parser.add_argument("--learning_rate", default=0.0004, type=float, help="Learning rate")
parser.add_argument("--n_steps", default=400, type=int, help="Number of steps of the environment taken during training")
parser.add_argument("--n_epochs", default=8, type=int, help="Number of learning epochs")
parser.add_argument("--shaped_r_coef_horizon", default=2.5e6, type=int, help="Annealing horizont for shaped partial rewards")
parser.add_argument("--divergent_check_timestep", default=3e6, type=int, help="Timestep of the check for divergent solution")
parser.add_argument("--num_workers", default=30, type=int, help="Number of parallel environment")
parser.add_argument("--eval_interval", default=10, type=int, help="Number of steps interval to perform evaluation during training")
parser.add_argument("--evals_num_to_threshold", default=2, type=int, help="Number of more evaluation to get exact value")
parser.add_argument("--device", default="cuda", type=str, help="Device ('cuda' or 'cpu')")
parser.add_argument("--training_percent_start_eval", default=0.5, type=float, help="Ratio of total training steps when evaluation starts")
parser.add_argument("--tensorboard_log", default=False, action="store_true", help="Whether to do tensorboard logging")
parser.add_argument("--seed", default=42, type=int, help="Random seed value")
parser.add_argument("--behavior_check", default=False, action="store_true",help="if true, logs actions and states, stops after divergent found" )
parser.add_argument("--log_dir", default="log_dir", help="where to store actions, state etc.")


args = parser.parse_args([] if "__file__" not in globals() else None)

import numpy as np
random.seed(args.seed)
np.random.seed(args.seed)


def load_or_train_models(args, env):
    """
    Tries to load trained models. Models are trained if not found.
    There are args.trained_models standard models trained.
    If the experiment is in POP mode there are two additional final population agents trained.
    """
    directory = projdir + "/diverse_population/models/" + args.layout_name + "/"
    models = []
    env.population = []
    env.population_mode = False
    for n in range(args.trained_models):
        model = load_or_train_model(directory, n, env, args)
        models.append(model)
        env.population.append(model)

        # First init_SP_agents models are always self-play
        if (n + 1) >= args.init_SP_agents:
            env.population_mode = args.mode == "POP"

    #At the end of the population training two more final agents are trained
    if args.mode == "POP":
        #First final agents is trained with all previos population agents
        final_model = train_final_model(directory, n+1, env, args)
        models.append(final_model)

        #Second final agent is trained without the initial self-play agents
        env.population = models[args.init_SP_agents:-1]
        final_model = train_final_model(directory, n+2, env, args)
        models.append(final_model)
    return models


def load_or_train_model(directory, n, env, args):
    """
    Tries to load one trained model. Model is trained if not found.
    """
    model = None
    if args.mode == "SP" or n < args.init_SP_agents:
        exp_part = args.exp
    else:
        exp_part = args.full_exp_name
    model_name = directory + exp_part + "/" + str(n).zfill(2)
    try:
        print(f"Looking for file {model_name}")
        model = PPO.load(model_name, env=env, device="cuda")
        model.custom_id = n
        print(f"model {model_name} loaded")
    except:
        if model is None:
            model = train_model(n, env, args)
        model.save(model_name)
        print(f"model {model_name} learned")

    return model


def train_model(n, env, args):
    """
    Trains one model using population-extended stable baselines PPO algorithm
    """
    found = False
    while not found:
        try:
            print(f"Learning {args.layout_name}/{args.exp}")
            model = PPO("CnnPolicy", env, device=args.device, verbose=0,

                        tensorboard_log=f"./diverse_population/logs/{args.layout_name}/{args.exp}/{str(n).zfill(2)}" if args.tensorboard_log else None,
                        n_steps=args.n_steps,
                        seed=args.seed,
                        batch_size=args.batch_size,
                        n_epochs=args.n_epochs,
                        learning_rate=args.learning_rate,
                        gae_lambda=0.98,
                        clip_range=args.clip_range,
                        max_grad_norm = args.max_grad_norm,
                        vf_coef=args.vf_coef,
                        policy_kwargs={"features_extractor_kwargs": {"frame_stacking": args.frame_stacking, "frame_stacking_mode": args.frame_stacking_mode}, "normalize_images": False }
                        )
            model.custom_id = n
            # Initially, the environment partner is a copy of itself (self-play)
            # If population mode is detected, this will change during training
            env.other_agent_model = model
            num_steps = args.total_timesteps
            model.learn(num_steps, args=args, reset_num_timesteps=False)
            found = True
        except divergent_solution_exception.divergent_solution_exception:
            print("found divergent solution")
            if (args.behavior_check):
                break;
            found = False

    return model


def train_final_model(directory, n, env, args):
    """
    Modifies final agent training parameters by extending training time and elimination all diversity pressures.
    Trains final population agent.
    """
    final_args = copy.deepcopy(args)

    # Reset all population diversification techniques
    final_args.total_timesteps = 1.5 * final_args.total_timesteps
    final_args.kl_diff_bonus_reward_coef = 0.
    final_args.kl_diff_bonus_reward_clip = 0.
    final_args.kl_diff_loss_coef = 0.
    final_args.kl_diff_loss_clip = 0.

    return load_or_train_model(directory, n, env, final_args)

def get_eval_models(args, gym_env, mode):
    """
    Trains or loads existing trained evaluation set models.
    """
    eval_args = copy.deepcopy(args)
    eval_args.exp = args.base_eval_name
    eval_args.full_exp_name = args.eval_set_name
    eval_args.trained_models = args.eval_agents
    eval_args.mode = mode
    return load_or_train_models(eval_args, gym_env)

def models_are_same(model1, model2):
    """
    Check if two PPO are the same by comparing all their network parameters values
    """
    for p1, p2 in zip(model1.policy.parameters(), model2.policy.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def get_name(name, args, sp=False, extended=False):
    """
    Builds experiment name based on give parameters.
    """
    full_name = name
    if sp:
        pass
    else:
        if extended:
            full_name = full_name + "_VF" + str(args.vf_coef)
            full_name = full_name + "_MGN" + str(args.max_grad_norm)
            full_name = full_name + "_CR" + str(args.clip_range)
            full_name = full_name + "_LR" + str(args.learning_rate)
            full_name = full_name + "_ES" + str(args.ent_coef_start)
            full_name = full_name + "_EE" + str(args.ent_coef_end)
            full_name = full_name + "_SRC" + str(args.shaped_r_coef_horizon)
            full_name = full_name + "_EP" + str(int(args.n_epochs))
            full_name = full_name + "_EH" + str(int(args.ent_coef_horizon))
            full_name = full_name + "_BS" + str(int(args.batch_size))
            full_name = full_name + "_NS" + str(int(args.n_steps))
            full_name = full_name + "_NW" + str(args.num_workers)
            full_name = full_name + "_TS" + str(int(args.total_timesteps))
        full_name = full_name + "_ROP" + str(args.rnd_obj_prob_thresh_agent)
        full_name = full_name + "_M" + str(args.mode)
        full_name = full_name + "_BRCoef" + str(args.kl_diff_bonus_reward_coef)
        full_name = full_name + "_BRClip" + str(args.kl_diff_bonus_reward_clip)
        full_name = full_name + "_LCoef" + str(args.kl_diff_loss_coef)
        full_name = full_name + "_LClip" + str(args.kl_diff_loss_clip)
        full_name = full_name + "_DSR" + str(args.delay_shared_reward)
        full_name = full_name + "_PAD" + str(args.partner_action_deterministic)
        full_name = full_name + "_NSP" + str(args.n_sample_partners)
    return full_name


def print_args():
    print("all args:")
    for arg in vars(args):
        print (arg, getattr(args, arg))

# Overcooked MDP and environment is loaded
mdp = OvercookedGridworld.from_layout_name(args.layout_name)
if (args.behavior_check):
    overcooked_env = OvercookedEnv.from_mdp(mdp, horizon=400, info_level=4)
#    print("grid:" + str(overcooked_env.mdp.mdp_params["terrain"]))
else:
    overcooked_env = OvercookedEnv.from_mdp(mdp, horizon=400)




if __name__ == "__main__":
    # State representaion and environment reset functions set
    os.makedirs(args.log_dir)
    feature_fn = lambda _, state: overcooked_env.lossless_state_encoding_mdp(state, debug=False)
    start_state_fn = mdp.get_random_start_state_fn(random_start_pos=True, # TODO: set Default True
                                                   rnd_obj_prob_thresh = args.rnd_obj_prob_thresh_env,# TODO: set Default args.rnd_obj_prob_thresh_env,
                                                   random_switch_start_pos = args.random_switch_start_pos) if args.static_start == False else mdp.get_standard_start_state
    print("pred gym env vytvvorenim")
    print_args()
    # Vectorized overcooked environments are initialized
    gym_env = get_vectorized_gym_env(
        overcooked_env, 'Overcooked-v0', agent_idx=0, featurize_fn=feature_fn, start_state_fn=start_state_fn, args=args
    )
    print("\n mame gym env\n")

    #Agent indices are set
    agent_idxs = [0 for _ in range(args.num_workers)]
    gym_env.remote_set_agent_idx(agent_idxs)
    gym_env.population = []

    #Frame stacking method is initialized
    gym_env.frame_stacking = args.frame_stacking
    gym_env.frame_stacking_mode = args.frame_stacking_mode

    evaluator = Evaluator(gym_env, args, deterministic=True, device="cpu")


    set_layout_params(args)
    args.full_exp_name = get_name(args.exp, args, sp=args.mode == "SP")
    print("budu volat load or train")
    print_args()
    #Models training
    models = load_or_train_models(args, gym_env)

    #Final evaluation
    if args.execute_final_eval:
        random.seed(args.seed)
        np.random.seed(args.seed)

        eval_env = "_ENVROP" + str(args.rnd_obj_prob_thresh_env)
        if args.mode == "POP":
            models_name = args.full_exp_name
            evals_name = args.eval_set_name
            group_name = models_name + "_X_" + evals_name

            #Evaluation set models are loaded
            eval_models = get_eval_models(args, gym_env, mode=args.eval_mode)

            #Sets of models are cross-play evaluated
            eval_table = evaluator.evaluate(models, eval_models, args.final_eval_games_per_worker, args.layout_name, group_name, eval_env = eval_env, mode=args.mode, deterministic=True)
            heat_map(eval_table, group_name, args.layout_name, eval_env = eval_env, deterministic=True)
        else:
            print("vypisuji v SP mode")
            group_name = args.full_exp_name + "_X_" + args.full_exp_name
            eval_table = evaluator.evaluate(models, models, args.final_eval_games_per_worker, args.layout_name, group_name, eval_env = eval_env, mode=args.mode, deterministic=True)
            print("eval table " + str(eval_table))
            print("eval_env " + str(eval_env))
            heat_map(eval_table, group_name, args.layout_name, eval_env = eval_env, deterministic=True)

