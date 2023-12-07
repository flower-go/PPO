import os
import sys
import datetime
codedir = os.environ["CODEDIR"]
#codedir = /home/premek/DP/
projdir = os.environ["PROJDIR"]
#projdir = /home/premek/DP/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_pytorch
sys.path.append(codedir + "/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src")
sys.path.append(codedir + "/PPO/overcooked_pytorch_stable_baselines/stable-baselines3")
sys.path.append(codedir + "/PPO/overcooked_pytorch_stable_baselines")

results = {}

maps = [
    # "small_corridor",
    "five_by_five",
    "schelling",
    "centre_pots",
    # "corridor",
    # "pipeline",
    "scenario1_s",
    "large_room",
    "asymmetric_advantages",  # tady
    "schelling_s",
    "coordination_ring",  # tady
    "counter_circuit_o_1order",  # tady
    #"long_cook_time",
    "cramped_room",  # tady
    "forced_coordination",  # tady
    "m_shaped_s",
    "unident",
    "simple_o",
    "centre_objects",
    "scenario2_s",
    "scenario3",
    "scenario2",
    "scenario4",
    "bottleneck",
    #"tutorial_0"
]


def check_models(name_1, name_2 = None, count = 30):
    is_ok = True
    for map in maps:
        name = name_1 + map + name_2
        path = "./diverse_population/models/" + map + "/" + name + "/"
        print(f"searching models in: {path}")
        try:
            files = [f.path for f in os.scandir(path) if f.is_file()]
        except:
            print(f"Path to model  {path} probably not found")
        results[map] = {}
        num_models = len(files)
        results[map]["model_count"] = num_models
        if num_models != count:
            is_ok= False
            print(f"Model count is wrong. Desired: {count} Actual:{num_models}")
        results[map]["model_files"] = [(f, os.path.getmtime(f)) for f in files]
    return results, is_ok

def check_maps(name_1, name_2 = None):
    is_ok = True
    for map in maps:
        name = name_1 + map + name_2
        path = "./diverse_population/visualisation/" + map + "/" 
        print(f"searching models in: {path}")
        try:
            files = [f.path for f in os.scandir(path) if f.is_file() and f.name.startswith(name_1)]
            results[map] = {}
            results[map]["map_file"] = [(f, os.path.getmtime(f)) for f in files][0]
        except Exception as error:
            print(f"Visual not found  {path} probably not found. Error {error}")
    return results, is_ok

import shutil
def copy_maps(filenames, dest_dir):
    for f in filenames:
        shutil.copy2(f, dest_dir) 

if __name__ == "__main__":
    res, ok = check_models(name_1 = "nost1_", name_2 = "_ref_30")
    res_m, ok = check_maps(name_1 = "nost1_", name_2 = "_ref_30")
#    print(res)
    no_map = []
    yes_map = []
    print("*************************************")
    print("Visualisation matrices are for maps:")
    for map in maps:
        map_file = res_m[map].get("map_file")

        if map_file is not None:
            print(f"{map}")
            yes_map.append(map_file[0])
        else:
            no_map.append(map)
    print("*************************************")
    print("no visualisation file for:")
    print(*no_map, sep="\n")
 #   print("yes maps")
#    print(yes_map)
    dir_name = projdir + "/diverse_population/visualisation/matrices" + f"{datetime.datetime.now():%Y_%m_%d_%H_%M_%S}"
    print(dir_name)
    os.makedirs(dir_name)
    copy_maps(yes_map,dir_name)
