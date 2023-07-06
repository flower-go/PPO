START_QSUB = "sh ./coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/scripts/run_uni "
SCRIPT_PATH = "./coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/scripts/script.sh"
layouts_onions = [
           "small_corridor",
           "five_by_five",
           "schelling",
           "centre_pots",
           "corridor",
           "pipeline",
           "scenario1_s",
           "large_room",
           "asymetric_advantages", #tady
           "schelling_s",
            "coordination_ring", #tady
           "counter_circuit_o1_order", #tady
           "long_cook_time",
           "cramped_room", #tady
           "forced_coordination", #tady
           "m_shaped_s",
           "unident",
           "simple_o",
           "centre_objects",
           "scenario2_s",
           "scenario3",
           "scenario2",
           "scenario4",
           "bottleneck",
           #"diagonal",
           #"long_forced",
           "tutorial_0"]
origin_params = {
    "discount":0.99,
    "GAE":0.98,
    "lr":1e-3,
    "vf": 0.5,
    "clip": 0.05,
    "max_grad_norm": 0.1,
    "grad_steps": 8,
    "minibatch_size": 2000,
    "num_par_envs": 30,
    "timesteps": 6e6,
    "entropy_bonus_start": 0.1,
    "entropy_bonus_end": 0.1,
    "entropy_horizon": None,
    "preemt_divergence_step": 3e6,
    "initial_positions_random": True #oni mely false, ale to nechcem testovat
}

novel_params = {
    "discount":0.98,
    "GAE":0.95,
    "lr":4e-4,
    "vf": 0.1,
    "clip": 0.1,
    "max_grad_norm": 0.3,
    "grad_steps": 8,
    "minibatch_size": 2000,
    "num_par_envs": 30,
    "timesteps": 5.5e6,
    "entropy_bonus_start": 0.1,
    "entropy_bonus_end": 0.03,
    "entropy_horizon": 1.5e6,
    "preemt_divergence_step": 3e6,
    "initial_positions_random": True
}

seeds = [79, 18, 17, 67, 63]
frame_stacking = {"channels":("channels",4),
                  "tuple":("tuple",4),
                  "nostack":("channels",1) #effectively no stacking
                  }


def SP_ref(layout_name, seed, stacking = "nostack"):
    #sh ./coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/scripts/run_uni exp="SP30_ref_nostack" layout_name="forced_coordination" seed=18 file=./coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/scripts/script.sh mode="SP" n_sample_partners=-1 trained_models=20 frame_stacking=1
    exp = stacking[0:4] + "_"  + layout_name + "_ref_30"
    end = ' mode="SP" n_sample_partners=-1 trained_models=30 file="' + SCRIPT_PATH + '"'
    stack = frame_stacking[stacking]
    stack_mode = stack[0]
    stack_number = stack[1]
    result = START_QSUB + 'exp="' + exp + '"' + ' layout_name="' + layout_name + '"'
    result += ' frame_stacking=' + str(stack_number) + ' frame_stacking_mode="' + stack_mode + '"'
    result += ' seed=' + str(seed) + ' execute_final_eval=True walltime=72:00:00' + end

    return result,exp

def SP_epochs(layout_name, seed, stacking = "nostack"):
    exp = "log1_" + stacking[0:4] + "_"  + layout_name + "_ref_30"
    end = ' mode="SP" n_sample_partners=-1 trained_models=30 file="' + SCRIPT_PATH + '"'
    stack = frame_stacking[stacking]
    stack_mode = stack[0]
    stack_number = stack[1]
    result = START_QSUB + 'exp="' + exp + '"' + ' layout_name="' + layout_name + '"'
    result += ' frame_stacking=' + str(stack_number) + ' frame_stacking_mode="' + stack_mode + '"'
    result += ' seed=' + str(seed) + end

    return result,exp


def ref_populations(names_only=False):
    #jenom ref populace bez stackingu
    seed = seeds[0]
    for stacking in frame_stacking:
        print("\n" + "#stacking: " + stacking + "\n")
        for layout_name in layouts_onions:
            res,exp = SP_ref(layout_name, seed,stacking)
            if not names_only:
                print(res)
            else:
                print(exp)


#this will run just 1 epoch of training with logging --> good for qualitative analysis of behaviour
# jenom ref populace bez stackingu
def one_epoch():
    seed = seeds[0]
    for stacking in frame_stacking:
        print("\n" + "#stacking: " + stacking + "\n")
        for layout_name in layouts_onions:
            res, exp = SP_epochs(layout_name, seed, stacking)
            print(res)

def print_exps_ref_pop():
    ref_populations(True)

#print_exps_ref_pop()

#this will generate qsubs for ref population training
#ref_populations(False)
one_epoch()










