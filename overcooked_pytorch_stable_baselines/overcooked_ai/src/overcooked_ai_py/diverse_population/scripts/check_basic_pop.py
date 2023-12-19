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
sys.path.append(projdir + "/diverse_population")
from visualisation.maps.maps_to_pdf import createPDF

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, help="Name of the resulting pdf file")
parser.add_argument("--title", type=str, help="Header of the pdf")
parser.add_argument("--columns", default=4, type=int, help="number of columns")
parser.add_argument("--prefix", default="", type=str, help="prefix ot be removed")
parser.add_argument("--postfix", default="", type=str, help="postfix to be removed")
args = parser.parse_args([] if "__file__" not in globals() else None)


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


for map in maps:
    results[map] = {}

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
        num_models = len(files)
        results[map]["model_count_" + name_1] = num_models
        if num_models != count:
            is_ok= False
            print(f"Model count is wrong. Desired: {count} Actual:{num_models}")
        results[map]["model_files_" + name_1] = [(f, os.path.getmtime(f)) for f in files]
    return results, is_ok

def check_maps(name_1, name_2 = None):
    is_ok = True
    for map in maps:
        name = name_1 + map + name_2
        path = "./diverse_population/visualisation/" + map + "/" 
        print(f"searching models in: {path}")
        try:
            files = [f.path for f in os.scandir(path) if f.is_file() and f.name.startswith(name_1)]
            results[map]["map_file_" + name_1] = [(f, os.path.getmtime(f)) for f in files][0]
        except Exception as error:
            print(f"Visual not found  {path} probably not found. Error {error}")
    return results, is_ok



def check_checkpoints(name_1, name_2 = None):
    is_ok = True
    for map in maps:
        name = name_1 + map + name_2
        path = "./diverse_population/checkpoints/" + map + "/" + name + "/"
        print(f"searching models in: {path}")
        for i in range(30):
            min_num = 3
            try:
                new_path = path + f"{i:02d}"
                files = [f.path for f in os.scandir(new_path) if f.is_file()]
                if len(files) < min_num:
                    min_num = len(files)
            except:
                print(f"Path to model  {path} probably not found")
        results[map]["checkpoint_count_"+ name_1] = min_num
        results[map]["checkpoint_files_" + name_1] = files
    return results, is_ok

import shutil
def copy_maps(filenames, dest_dir):
    for f in filenames:
        shutil.copy2(f, dest_dir) 

def print_models(name_1="nost1_", name_2="_ref_30"):
    print("*************************************")
    print(f"Printing models for {name_1} a {name_2}")
    res, ok = check_models(name_1=name_1, name_2=name_2)
    return res


def print_eval(name_1 = "nost1_", name_2 = "_ref_30"):
    print("*************************************")
    print(f"Printing visualisation matrices for {name_1} a {name_2}")
    res_m, ok = check_maps(name_1 = name_1, name_2 = name_2)
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
    return yes_map

def prepare_pdf(yes_map,args):
    dir_name = projdir + "/diverse_population/visualisation/matrices" + f"{datetime.datetime.now():%Y_%m_%d_%H_%M_%S}"
    print(dir_name)
    os.makedirs(dir_name)
    copy_maps(yes_map,dir_name)
    createPDF(dir_name, args.filename, args.title, 4, args)

def print_checkpoints():
    res_c, ok_c = check_checkpoints(name_1="nost1_", name_2="_ref_30")
    print("*************************************")
    print("Checkpoint counts:")
    for map in maps:
       print(f'"{map}": {res_c[map]["checkpoint_count"]},')

def print_table():
    for map in maps:
        values = results[map]
        line = ""
        for key, value in values.items():
            if "files" not in key:
                if ""
                line = line + f" {value} "
        print(line)
        exit()
if __name__ == "__main__":
    print_models()
    yes_maps = print_eval()
    yes_maps + print_eval("tupl_", "ref-30") + print_eval("chan_", "ref-30")
    #print_checkpoints()
    #prepare_pdf(yes_maps,args)
    print_table()




