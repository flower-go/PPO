import os
import sys
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
    "pipeline",
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
        path = f"../models/{map}/{name}/"
        print("searching models in: {path}")
        subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
        results[map] = {}
        num_models = len(subfolders)
        results[map]["model_folders"] = num_models
        if num_models != count:
            is_ok= False
            print(f"Model count is wrong. Desired: {count} Actual: {num_models}")
        for sub in subfolders:
            model_files = [(f,os.path.getmtime(f)) for f in os.scandir(sub) if f.is_file()]
            if len(model_files) > 1:
                print(f"V adresari je vice model souboru. Dir: {sub}")
            else:
                file, date = model_files[0]
                results["map"]["model_file"] = (file, date)
                print(f"model {file} created {date}")




