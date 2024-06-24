#import jinja2
#environment = jinja2.Environment()

#template = environment.from_string("Hello, {{ name }}!")
#a = template.render(name="World")
#print(a)
import os
from os import listdir
from os.path import isfile, join

import numpy as np

# write_messages.py
#CODE_PATH = "C:/Users/PetraVysušilová/Documents/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/"
CODE_PATH = "C:/Users/PETRAV~1/PYCHAR~1/coding/PPO/OVERCO~1/OVERCO~1/src/OVERCO~1/DIVERS~1/"
MDP_PATH = "C:/Users/PetraVysušilová/PycharmProjects/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/mdp/"
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
           "STD error": "std_error",
           "avgs out to diag": "avg_out_to_avg_diag",
           "cov out to cov diag": "cov_out_to_cov_diag"}

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

def get_rank(input_vec):
    ranks_order = sorted(np.array(range(0, 9)), key=lambda x: input_vec[x])
    ranks = np.full(len(ranks_order), -1)
    for i in range(len(ranks_order)):
        ranks[ranks_order[i]] = i
    return ranks

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
        "C:/Users/PetraVysušilová/PycharmProjects/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/scripts/html_rendering/templates"))
    template = environment.get_template("results.txt")
    heat_maps()
    heat_steps()
    map_image()
    get_metrics()
    for e in exp_names:
        collect_R(e)

    for q in ["all", "best"]:
        get_quant(q)
    with open("./pages/all_results.html", mode="w", encoding="utf-8") as results:
        results.write(template.render(maps=layouts_onions, res = res_dict, exps = exp_names, metrics=metrics))
        print(f"... wrote {results}")

def compute_cov():
    for map in layouts_onions:
        for stack in frame_stacking:
            s = frame_stacking[stack]
            try:
                std = res_dict[map][s]["std_error_sp"]
                avg = res_dict[map][s]["average_diag"]
                res_dict[map][s]["CoV"] = round(std/avg,2)
            except:
                pass
def sp_difficulty():
    load_sp_metrics()



    environment = Environment(loader=FileSystemLoader(
        "C:/Users/PetraVysušilová/PycharmProjects/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/scripts/html_rendering/templates"))
    template = environment.get_template("results_tables.txt")


    with open("./pages/results_tables.html", mode="w", encoding="utf-8") as results:
        results.write(template.render(maps=layouts_onions, res = res_dict, exps = exp_names, metrics=metrics_sp))
        print(f"... wrote {results}")


def sp_sort_basic():
    heat_maps()
    #chci jenom pro kazdou mapu average a odchylku a serazeno dle average
    load_sp_metrics()
    compute_cov()
    stacks = {}
    for s in frame_stacking:
        stack = frame_stacking[s]

        rs = dict(sorted(res_dict.items(), key=lambda item: item[1][stack]["average_diag"] if len(item[1][stack]) > 1 else 0, reverse=True ))
        #[v[1]["nostack"]["average_diag"] if len(v[1]["nostack"]) > 0 else None for v in res_dict.items()]

        stacks[stack] = rs
    environment = Environment(loader=FileSystemLoader(
        "C:/Users/PetraVysušilová/PycharmProjects/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/scripts/html_rendering/templates"))
    template = environment.get_template("results_sp_sorted.txt")


    with open(f"./pages/results_sort_basic_allstacks.html", mode="w", encoding="utf-8") as results:
        results.write(template.render(maps=list(stacks["nostack"].keys()), stacks = stacks, exps = exp_names, metrics=metrics_sp))
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
            "C:/Users/PetraVysušilová/PycharmProjects/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/scripts/html_rendering/templates"))
        template = environment.get_template("results_off_diag.txt")

        with open(f"./pages/results_off_diag{stack}.html", mode="w", encoding="utf-8") as results:
            results.write(template.render(maps=list(rd.keys()), res=res_dict, exps=exp_names, metrics=metrics,
                                          sorted_maps=sorted_layouts, empty_list = empty_list,stack=stack))
            print(f"... wrote {results}")

def get_pages():
    mypath = "./pages/"
    files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f != "main_page.html"]
    result = []
    for f in files:
        result.append((f,f"./{f}"))
    return result
def update_menu():
    pages = get_pages()
    environment = Environment(loader=FileSystemLoader(
        "C:/Users/PetraVysušilová/PycharmProjects/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/scripts/html_rendering/templates"))
    template = environment.get_template("rozcestnik.txt")

    with open(f"./pages/main_page.html", mode="w", encoding="utf-8") as results:
        results.write(template.render(links=pages))

def comp_avg_order(matrix):
    matrix = np.transpose(matrix)
    result = [np.unique(matrix[i], return_counts=True) for i in range(len(matrix))]
    avgs = []
    for i in range(3):
        a= result[0][1][np.where(result[0][0] == i)]
        b = result[1][1][np.where(result[1][0] == i)]
        c = result[2][1][np.where(result[2][0] == i)]
        avg = (3*a+2*b+1*c)/(a + b + c)
        avgs.append(avg)
    return result,avgs

def get_sorted_stack():
    r = {}
    mat = []
    for map in layouts_onions:
        missing = []
        l = []
        i=-1
        for stack in frame_stacking:

            i += 1
            try:
                ad = res_dict[map][frame_stacking[stack]]["average_diag"]
                l.append(ad)
            except:
                missing.append(i)
                l.append(-1)
        if len(missing) < 3:
            r[map] = np.argsort(l)
            mat.append(r[map])
            for i in missing:
                r[map][np.where(r[map] == i)] = -1

    res,a = comp_avg_order(mat)

    return r, res,a
def stack_influence():
    load_sp_metrics()
    sorted_stacks,counts,avg_order = get_sorted_stack()

    environment = Environment(loader=FileSystemLoader(
        "C:/Users/PetraVysušilová/PycharmProjects/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/scripts/html_rendering/templates"))
    template = environment.get_template("sorted_stack_np.txt")

    with open(f"./pages/sorted_stack_np.html", mode="w", encoding="utf-8") as results:
        results.write(template.render(results=sorted_stacks,avgs=avg_order))


stacking = ["chan", "tupl", "nost"]
exp_type = ["SP", "L0", "L1", "L2", "R0", "R1", "R2", "R0L0", "R1L1"]


def get_index(stack, exp, layout):
    col = exp_type.index(exp)
    row = stacking.index(stack) + layouts_onions.index(layout) * 3
    return ([row, col])


def get_rank_matrix(input_matrix):
    res = []
    # TODO je treba osetrit cisla co chybi
    for i in range(len(input_matrix)):
        row = input_matrix[i]
        # ranks = np.argsort(row)
        ranks_order = sorted(np.array(range(0, 9)), key=lambda x: row[x], reverse=True)
        ranks = np.full(len(ranks_order), -1)
        for i in range(len(ranks_order)):
            ranks[ranks_order[i]] = i
        res.append(ranks)
    return np.array(res)


def get_empy_rows(input_matrix):
    zero_rows = []
    for i in range(len(input_matrix)):
        row = input_matrix[i]
        if np.sum(row) == 0:
            zero_rows.append(i)
    return zero_rows


def remove_zeros(rank, zeros):
    res = []
    for i in range(len(rank)):
        row = rank[i]
        if not i in zeros:
            res.append(row)
    return np.array(res)


def column_average(i_m):
    return np.mean(i_m, axis=0)


def eval_auc(filename):
    res_matrix = np.zeros((len(stacking) * len(layouts_onions), len(exp_type)))

    with open(file=filename, mode='r') as res_file:
        for line in res_file:
            if len(line) > 0:
                splitted = line.split(",")
                stack = splitted[0]
                layout = splitted[1]
                e_type = splitted[2]
                index = get_index(stack, e_type, layout)
                res_matrix[index[0], index[1]] = splitted[3]

    rank_matrix = get_rank_matrix(res_matrix)
    zero_rows = get_empy_rows(res_matrix)
    without_zeros = remove_zeros(rank_matrix, zero_rows)
    avg_rank = column_average(without_zeros)
    # je to blbe - radi to od nejmensiho a jeste nevim jestli ty cisla jsou fakt poradi
    return res_matrix, rank_matrix, avg_rank, without_zeros, zero_rows

def generate_names(zero_rows):
    names = []
    i = 0
    for l in layouts_onions:
        for s in stacking:
            if i not in zero_rows:
                names.append(f"{s}_{l}")
            i+=1
    return names

def population_avg_rank():
    input_dict= {}
    for perc in [15,30]:
        for best in ["","_best"]:

            res_mat, rank_mat, avg_rank,no_zeros, zero_rows = eval_auc(
                f"C:/Users/PetraVysušilová/PycharmProjects/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/evaluation/metrics/auc{best}_{perc}.0.txt")
            sorted_avg_rank = get_rank(avg_rank)
            input_dict[f"{perc}_{best}"] = {}
            input_dict[f"{perc}_{best}"]["res_mat"] = res_mat
            input_dict[f"{perc}_{best}"]["rank_mat"] = no_zeros
            input_dict[f"{perc}_{best}"]["avg_rank"] = np.round(avg_rank,2)
            input_dict[f"{perc}_{best}"]["sorted_avg_rank"] = sorted_avg_rank


    environment = Environment(loader=FileSystemLoader(
        "C:/Users/PetraVysušilová/PycharmProjects/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/scripts/html_rendering/templates"))
    template = environment.get_template("pop_avg_rank.txt")

    with open(f"./pages/pop_avg_rank.html", mode="w", encoding="utf-8") as results:
        #results.write(template.render(res_mat=res_mat,rank_mat=no_zeros,avg_rank=np.round(avg_rank,2), sorted_avg_rank = sorted_avg_rank, color_range=["Yellow","Light-Green","Green","Teal","Cyan","Blue","Indigo","Purple","Black"], exp_names = exp_type, names=generate_names(zero_rows)))
        results.write(template.render(input_dict = input_dict, color_range=["yellow","orange","light-green","green","teal","cyan","blue","indigo","purple"], exp_names = exp_type, names=generate_names(zero_rows)))

#all_results()
#sp_difficulty()
#sp_sort_basic()
#sp_res_off_diag()
#stack_influence()
population_avg_rank()
#update_menu()

