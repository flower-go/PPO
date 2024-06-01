#import jinja2
#environment = jinja2.Environment()

#template = environment.from_string("Hello, {{ name }}!")
#a = template.render(name="World")
#print(a)
import os
# write_messages.py
#CODE_PATH = "C:/Users/PetraVysušilová/Documents/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/"
CODE_PATH = "C:/Users/PetraVysušilová/DOCUME~1/coding/PPO/OVERCO~1/OVERCO~1/src/OVERCO~1/DIVERS~1/"
MDP_PATH = "C:/Users/PetraVysušilová/Documents/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/mdp/"
heat_path = ("visualisation/")
import win32api

from jinja2 import Environment, FileSystemLoader
#from overcooked_pytorch_stable_baselines.overcooked_ai.src.overcooked_ai_py.diverse_population.evaluation.compute_metrics import metrics as m_list

res_dict = {}


max_score = 100
test_name = "Python Challenge"
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

metrics = {"SDAO": "diag_average_average_out",
           "SDMO": "diag_average_max_out",
           "STD error": "std_error"}

metrics_sp = [
    "std_error_sp",
    "max_on_diag",
    "min_on_diag",
    "average_diag"
]

frame_stacking = {"chan":"channels",
                  "tupl":"tuple",
                  "nost":"nostack", #effectively no stacking
                  }

exp_names = ["R0","R1", "R2", "L0", "L1", "L2", "R0L0", "R1L1"]



for i in layouts_onions:
    res_dict[i] = {}
    for s in frame_stacking:
        res_dict[i][frame_stacking[s]] = {}
    res_dict[i]["general"] = {}

def heat_maps():
    result = []
    path = CODE_PATH + heat_path
    for map in layouts_onions:
        for s in frame_stacking:
            file = f"{path}/{map}/{s}_{map}_ref-30_reordered.png"
            res_dict[map][frame_stacking[s]]["heat"] = file
            #result.append(file)

def heat_steps():
    path = CODE_PATH + heat_path
    for map in layouts_onions:
        for s in frame_stacking:
            file = f"{path}/{map}/steps2754060_{s}_{map}_ref-30_reordered.png"
            res_dict[map][frame_stacking[s]]["heat_s2"] = file
            file = f"{path}/{map}/steps1377030_{s}_{map}_ref-30_reordered.png"
            res_dict[map][frame_stacking[s]]["heat_s1"] = file
def map_image():
    path = CODE_PATH +  "visualisation/maps"
    for map in layouts_onions:
        for s in frame_stacking:
            file = f"{path}/{map}.png"
            res_dict[map]["general"]["layout"] = file

def get_metrics():
    for m in metrics:
        name = m
        append = metrics[m]
        path = CODE_PATH + "evaluation/metrics/" + append + ".txt"
        with open(path) as file:
            lines = [line.rstrip() for line in file]
        for line in lines:
            if line.startswith("*"):
                break
            l = line.split(",")
            stack = frame_stacking[l[0]]
            map = l[1]
            value = l[3]
            res_dict[map][stack][name] = round(float(value),2)

def collect_R(r="R0"):
    path = CODE_PATH + heat_path
    for map in layouts_onions:
        for s in frame_stacking:
            file = f"{path}{map}/{s}_{map}_{r}_X_{s}_{map}_ref-30_ENVROP0.0.png"
            #short_file_name = win32api.GetShortPathName(file)
            #print(short_file_name)
            #exit()
            res_dict[map][frame_stacking[s]][r] = file

def get_quant(suffix):
    path = CODE_PATH + heat_path
    for map in layouts_onions:
        for s in frame_stacking:
            file = f"{path}{map}/{s}_{map}_quant15_{suffix}.png"
            # short_file_name = win32api.GetShortPathName(file)
            # print(short_file_name)
            # exit()
            res_dict[map][frame_stacking[s]][f"q{suffix}"] = file

def load_sp_metrics():
    for m in metrics_sp:
        name = m
        append = m
        path = CODE_PATH + "evaluation/metrics/" + append + "_SP" +  ".txt"
        with open(path) as file:
            lines = [line.rstrip() for line in file]
        for line in lines:
            if line.startswith("*") or len(line) == 0:
                break
            l = line.split(",")
            print(l)
            stack = frame_stacking[l[0]]
            map = l[1]
            value = l[3]
            res_dict[map][stack][name] = round(float(value),2)


def rename_R(r="R0"):
    path = CODE_PATH + heat_path
    for map in layouts_onions:
        for s in frame_stacking:
            file = f"{path}{map}/{s}_{map}_{r}_X_SP_EVAL2_ROP0.0_ENVROP0.0.png"
            if os.path.exists(file):
                print("existuje")
                new_name = f"{path}{map}/{s}_{map}_{r}_X_{s}_{map}_ref-30_ENVROP0.0.png"
                #print(new_name)
                os.rename(file,new_name)


def remove_empty_maps(dict):
    maps = []
    for map in dict:
        if "SDAO" in res_dict[map]["nostack"].keys():
            maps.append(map)
    return maps

def all_results():
    environment = Environment(loader=FileSystemLoader(
        "C:\\Users\\PetraVysušilová\\Documents\\coding\\PPO\\overcooked_pytorch_stable_baselines\\overcooked_ai\\src\\overcooked_ai_py\\diverse_population\\scripts\\html_rendering\\templates"))
    template = environment.get_template("results.txt")
    heat_maps()
    heat_steps()
    map_image()
    get_metrics()
    for e in exp_names:
        collect_R(e)

    for q in ["all", "best"]:
        get_quant(q)
    with open("results.html", mode="w", encoding="utf-8") as results:
        results.write(template.render(maps=layouts_onions, res = res_dict, exps = exp_names, metrics=metrics))
        print(f"... wrote {results}")

def compute_cov():
    for map in layouts_onions:
        try:
            std = res_dict[map]["nostack"]["std_error_sp"]
            avg = res_dict[map]["nostack"]["average_diag"]
            res_dict[map]["nostack"]["CoV"] = round(std/avg,2)
        except:
            pass
def sp_difficulty():
    load_sp_metrics()



    environment = Environment(loader=FileSystemLoader(
        "C:\\Users\\PetraVysušilová\\Documents\\coding\\PPO\\overcooked_pytorch_stable_baselines\\overcooked_ai\\src\\overcooked_ai_py\\diverse_population\\scripts\\html_rendering\\templates"))
    template = environment.get_template("results_tables.txt")


    with open("results_tables.html", mode="w", encoding="utf-8") as results:
        results.write(template.render(maps=layouts_onions, res = res_dict, exps = exp_names, metrics=metrics_sp))
        print(f"... wrote {results}")


def sp_sort_basic():
    heat_maps()
    #chci jenom pro kazdou mapu average a odchylku a serazeno dle average
    load_sp_metrics()
    compute_cov()
    for s in frame_stacking:
        stack = frame_stacking[s]
        rs = dict(sorted(res_dict.items(), key=lambda item: item[1][stack]["average_diag"] if len(item[1][stack]) > 1 else 0, reverse=True ))
        #[v[1]["nostack"]["average_diag"] if len(v[1]["nostack"]) > 0 else None for v in res_dict.items()]



        environment = Environment(loader=FileSystemLoader(
            "C:\\Users\\PetraVysušilová\\Documents\\coding\\PPO\\overcooked_pytorch_stable_baselines\\overcooked_ai\\src\\overcooked_ai_py\\diverse_population\\scripts\\html_rendering\\templates"))
        template = environment.get_template("results_sp_sorted.txt")


        with open(f"results_sort_basic_{stack}.html", mode="w", encoding="utf-8") as results:
            results.write(template.render(maps=list(rs.keys()), res = rs, exps = exp_names, metrics=metrics_sp, stack=stack))
            print(f"... wrote {results}")

#TODO blbe a empty list rika ty co nejsou prazdne
def sp_res_off_diag():
    heat_maps()
    get_metrics()
    for s in frame_stacking:
        stack = frame_stacking[s]
        empty_list = [k for k in res_dict.keys() if "SDAO" not in res_dict[k][stack].keys()]
        rd = {key:res_dict[key] for key in sorted(res_dict.keys())}
        sorted_layouts = {}
        for metric in metrics:
            sorted_layouts[metric] = remove_empty_maps(list(dict(sorted(res_dict.items(),
                         key=lambda item: item[1][stack][metric] if len(item[1][stack]) > 1 else 0,
                         reverse=True)).keys()))


        environment = Environment(loader=FileSystemLoader(
            "C:\\Users\\PetraVysušilová\\Documents\\coding\\PPO\\overcooked_pytorch_stable_baselines\\overcooked_ai\\src\\overcooked_ai_py\\diverse_population\\scripts\\html_rendering\\templates"))
        template = environment.get_template("results_off_diag.txt")

        with open(f"results_off_diag{stack}.html", mode="w", encoding="utf-8") as results:
            results.write(template.render(maps=list(rd.keys()), res=res_dict, exps=exp_names, metrics=metrics,
                                          sorted_maps=sorted_layouts, empty_list = empty_list,stack=stack))
            print(f"... wrote {results}")

#sp_difficulty()
sp_sort_basic()
#sp_res_off_diag()

