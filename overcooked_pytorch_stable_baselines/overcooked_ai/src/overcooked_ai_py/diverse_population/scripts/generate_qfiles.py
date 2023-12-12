import json
vis_maps =[
"five_by_five",
"schelling",
"centre_pots",
"scenario1_s",
"large_room",
"schelling_s",
"coordination_ring",
"counter_circuit_o_1order",
"cramped_room",
"forced_coordination",
"m_shaped_s",
"unident",
"simple_o",
"centre_objects",
"scenario2_s",
"scenario3",
"scenario2",
"scenario4",
"bottleneck"
]
layouts_onions = [
           "small_corridor",
           "five_by_five",
           "schelling",
           "centre_pots",
           "corridor",
           "pipeline",
           "scenario1_s",
           "large_room",
           "asymmetric_advantages", #tady
           "schelling_s",
            "coordination_ring", #tady
           "counter_circuit_o_1order", #tady
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
           "tutorial_0"]

seeds = [79, 18, 17, 67, 63]
frame_stacking = {"chan":("channels",4),
                  "tupl":("tuple",4),
                  "nost":("channels",1) #effectively no stacking
                  }

def gen_ref_30_common(result_dict):
    result_dict["execute_final_eval"] = True
    result_dict["mode"] ="SP"
    result_dict["n_sample_partners"] = -1
    result_dict["seed"] = seeds[0]
    result_dict["behavior_check"] = True
    return result_dict



def generate_whole_ref_pop(map_list = layouts_onions, step = None):
    result_dict={}
    res = gen_ref_30_common(result_dict)
    for stack_type in frame_stacking:
        fs = frame_stacking[stack_type]
        res["frame_stacking_mode"] = fs[0]
        res["frame_stacking"] = fs[1]

        for map in map_list:
            res["exp"] = fs[0] + "_" + map + "_" + "ref-30"
            res["layout_name"] = map
            prefix = ""
            if step is not None:
                res["checkp_step"] = step
                prefix = "steps" + str(step) + "_"
            with open('./hyperparams/' + prefix + stack_type + "_" + map + "_" + "ref-30" + '.json', 'w') as f:
                json.dump(res, f)

def generate_steps():
    steps = [1377030, 2754060, 4131090]
    counts = {
        "five_by_five": 2,
        "schelling": 3,
        "centre_pots": 3,
        "scenario1_s": 3,
        "large_room": 3,
        "asymmetric_advantages": 3,
        "schelling_s": 3,
        "coordination_ring": 3,
        "counter_circuit_o_1order": 3,
        "cramped_room": 3,
        "forced_coordination": 2,
        "m_shaped_s": 2,
        "unident": 2,
        "simple_o": 2,
        "centre_objects": 3,
        "scenario2_s": 3,
        "scenario3": 3,
        "scenario2": 2,
        "scenario4": 3,
        "bottleneck": 3
    }
    for i, s in enumerate(steps):
        print(f"# steps {steps[i]}")
        step_map_list = []
        for m in vis_maps:
            if counts[m] >= i + 1:
                step_map_list.append(m)
        generate_whole_ref_pop(step_map_list,step=s)


generate_whole_ref_pop()
generate_steps()









