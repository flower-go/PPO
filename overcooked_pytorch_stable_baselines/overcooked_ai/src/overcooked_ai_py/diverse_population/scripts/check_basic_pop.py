import os
import sys
import datetime
import pandas as pd
codedir = os.environ["CODEDIR"]
#codedir = "/Users/petravysusilova/Documents/TR/research/coding"
#codedir = /home/premek/DP/
#projdir = os.environ["PROJDIR"]
projdir = codedir + "/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py"
#projdir = /home/premek/DP/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_pytorch
sys.path.append(codedir + "/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src")
sys.path.append(codedir + "/PPO/overcooked_pytorch_stable_baselines/stable-baselines3")
sys.path.append(codedir + "/PPO/overcooked_pytorch_stable_baselines")
sys.path.append(projdir + "/diverse_population")
#from visualisation.maps.maps_to_pdf import createPDF

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default="mapy.pdf", help="Name of the resulting pdf file")
parser.add_argument("--title", type=str, default="header", help="Header of the pdf")
parser.add_argument("--columns", default=4, type=int, help="number of columns")
parser.add_argument("--prefix", default="chan_", type=str, help="prefix ot be removed")
parser.add_argument("--postfix", default="ref-30", type=str, help="postfix to be removed")
args = parser.parse_args([] if "__file__" not in globals() else None)


results = {}
res_table = pd.DataFrame(columns=['exp_name', "stacking", "layout",'num_models', 'has_map', 'map_file', 'checkp_count', 'checkp_structure' ])


maps = [
    "small_corridor",
    "five_by_five",
    "schelling",
    "centre_pots",
    "corridor",
    "pipeline",
    "scenario1_s",
    "large_room",
    "asymmetric_advantages",  # tady
    "schelling_s",
    "coordination_ring",  # tady
    "counter_circuit_o_1order",  # tady
    "long_cook_time",
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
    "tutorial_0"
]

def fill_table():
    exp_names1 = ["chan", "tupl", "nost"]
    names_dict = []
    layout_dict= []
    stacking_dict = []
    for map in maps:
        for pref in exp_names1:
            names_dict.append(pref + "_" + map)
            layout_dict.append(map)
            stacking_dict.append(pref)
    res_table["exp_name"] = names_dict
    res_table["layout"] = layout_dict
    res_table["stacking"] = stacking_dict
    res_table.set_index("exp_name", inplace=True)




for map in maps:
    results[map] = {}

def check_models(name_1, name_2 = None, count = 30):
    is_ok = True
    for map in maps:
        name = name_1 + map + name_2
        path = projdir + "/diverse_population/models/" + map + "/" + name + "/"
        #print(f"searching models in: {path}")
        try:
            files = [f.path for f in os.scandir(path) if f.is_file()]
        except:
            print(f"Path to models  {path} probably not found", file=sys. stderr)
            res_table.loc[name_1 + map]["num_models"] = 0
            continue
        num_models = len(files)
        #results[map]["model_count_" + name_1] = num_models
        if num_models != count:
            is_ok= False
            print(f"Model count is wrong. Desired: {count} Actual:{num_models}")
        res_table.loc[name_1 + map]["num_models"] = len(files)
    return results, is_ok

def check_maps(name_1, name_2 = None):
    is_ok = True
    for map in maps:
        name = name_1 + map + name_2
        path = "./diverse_population/visualisation/" + map + "/" 
        print(f"searching models in: {path}")
        try:
            files = [f.path for f in os.scandir(path) if f.is_file() and f.name.startswith(name_1)]
            potential_files = [(f, os.path.getmtime(f)) for f in files]
            latest_file = max(files, key=os.path.getctime)
            results[map]["map_file_" + name_1] = (latest_file, os.path.getmtime(latest_file))
            print(f"file se jmenuje {results[map]['map_file_' + name_1]}")
        except Exception as error:
            print(f"Visual not found  {path} probably not found. Error {error}")
    return results, is_ok



def check_checkpoints(name_1, name_2 = None):
    is_ok = True
    for map in maps:
        name = name_1 + map + name_2
        path = "./diverse_population/checkpoints/" + map + "/" + name + "/"
        print(f"searching models in: {path}")
        num_checkps = 30
        checkp_structure = []
        for i in range(30):
            try:
                new_path = path + f"{i:02d}"
                files = [f.path for f in os.scandir(new_path) if f.is_file()]
            except:
                print(f"Path to model  {path} probably not found")
                files = []
                num_checkps -= 1
            checkp_structure.append(files)
        #tady chchi pocty pro vsech 30 modelu
        res_table.loc[name_1 + map]["checkp_count"] = num_checkps
        res_table.loc[name_1 + map]["checkp_structure"] = checkp_structure
        #results[map]["checkp_files"] = files
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


def print_eval(name_1 = "nost_", name_2 = "_ref_30"):
    res_m, ok = check_maps(name_1 = name_1, name_2 = name_2)
    no_map = []
    yes_map = []
    for map in maps:
        map_file = res_m[map].get("map_file_" + name_1)

        if map_file is not None:
            print(f"{map}")
            res_table.loc[name_1 + map]["map_file"] = map_file[0]
            res_table.loc[name_1 + map]["has_map"] = 1
            yes_map.append(map_file[0])
        else:
            no_map.append(map)
    print(*no_map, sep="\n")
    return yes_map

def prepare_pdf(yes_map,args):
    #tady potrebuju vypocitat jmena pro vysledky a tri stepy checkpoints
    dir_name = projdir + "/diverse_population/visualisation/matrices" + f"{datetime.datetime.now():%Y_%m_%d_%H_%M_%S}"
    print(dir_name)
    os.makedirs(dir_name)
    copy_maps(yes_map,dir_name)
    # tady je treba upravit vypisovani aby to pro jednu mapu vypsalo vsechny ty files <-- idealne vyuzit ze to bude ulozene v te pandas tabulce pohromade√
    createPDF(dir_name, args.filename, args.title, 5, args)

def print_checkpoints(name_1="nost_", name_2="_ref-30"):
    res_c, ok_c = check_checkpoints(name_1=name_1, name_2=name_2)
    print("*************************************")
    print("Checkpoint counts:")
    for map in maps:
       print(f'"{map}": {res_c[map]["checkpoint_count"]},')
       res_table.loc[name_1 + map]["checkp_count"] = res_c[map]["checkpoint_count"]

if __name__ == "__main__":
    fill_table()

    #trained models check
    check_models("nost_","_ref-30")
    check_models("tupl_","_ref-30")
    check_models("chan_","_ref-30")
    
    #eval table check
    print_eval()
    print_eval("chan_", "_ref-30")
    print_eval("tupl_","_ref-30")


    #check checkpoints
    check_checkpoints("nost_" ,"_ref-30")
    check_checkpoints("chan_", "_ref-30")
    check_checkpoints("tupl_", "_ref-30")

    #print table
    #res_table.groupby(["layout"]).apply(print)
    #print(res_table.to_string())
    r2 = res_table.set_index(["layout", "stacking"])
    print(r2.to_string())
    with open("res_table.txt", "w") as text_file:
        text_file.write(r2.to_string())
    
    print("end")
#   print_models()
    #yes_maps = print_eval()
    #yes_maps =  print_eval("chan_", "ref-30")
    #print_checkpoints()
    #prepare_pdf(yes_maps,args)
    #print_table()




