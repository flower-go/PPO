import json
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
frame_stacking = {"channels":("channels",4),
                  "tuple":("tuple",4),
                  "nostack":("channels",1) #effectively no stacking
                  }

def gen_ref_30_common(result_dict):
    result_dict["execute_final_eval"] = True
    result_dict["mode"] ="SP"
    result_dict["n_sample_partners"] = -1
    result_dict["seed"] = seeds[0]
    return result_dict


def generate_whole_ref_pop(map_list = layouts_onions):
    result_dict={}
    res = gen_ref_30_common(result_dict)
    for stack_type in frame_stacking:
        fs = frame_stacking[stack_type]
        res["frame_stacking_mode"] = fs[0]
        res["frame_stacking"] = fs[1]

        for map in map_list:
            res["exp"] = fs[0] + "_" + map + "_" + "ref_30"
            res["layout_name"] = map
            with open('./hyperparams/' + res["exp"] + '.json', 'w') as f:
                json.dump(res, f)



generate_whole_ref_pop()








