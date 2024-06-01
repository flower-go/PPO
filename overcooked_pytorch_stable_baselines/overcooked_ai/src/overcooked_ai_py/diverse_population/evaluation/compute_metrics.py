import numpy as np
import os
from statistics import mean
import pandas as pd
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
#a = np.loadtxt("coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/evaluation/five_by_five/nost_coordination_ring_ref-30_ROP0.0_MSP_BRCoef0.0_BRClip0.0_LCoef0.0_LClip0.0_DSRFalse_PADFalse_NSP-1_X_nost_coordination_ring_ref-30_ROP0.0_MSP_BRCoef0.0_BRClip0.0_LCoef0.0_LClip0.0_DSRFalse_PADFalse_NSP-1_ENVROP0.0")
#a = np.loadtxt("coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/evaluation/coordination_ring/nost_coordination_ring_ref-30_ROP0.0_MSP_BRCoef0.0_BRClip0.0_LCoef0.0_LClip0.0_DSRFalse_PADFalse_NSP-1_X_nost_coordination_ring_ref-30_ROP0.0_MSP_BRCoef0.0_BRClip0.0_LCoef0.0_LClip0.0_DSRFalse_PADFalse_NSP-1_ENVROP0.0")
import scipy
import scipy.cluster.hierarchy as sch
#PATH_PREFIX = "/storage/plzen1/home/ayshi/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/evaluation/"
PATH_PREFIX = "./overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/evaluation/"


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
           #"diagonal",
           #"long_forced",
           "tutorial_0"]

def comp_std_error(matrix):
    return np.std(matrix), 0

def comp_diag_average_average_out(matrix):
    diag_array =  []
    out_array = []
    diag_zeros = 0
    for i in range(len(matrix[0])):
         diag_item = matrix[i][i]
         if diag_item == 0:
             diag_zeros += 1
             continue
         diag_array.append(1)
         out_row = []
         for j in range(len(matrix[i])):
             if i != j:
                 out_row.append(matrix[i][j]/diag_item)
         out_array.append(out_row)

    max_out = [mean(r) for r in out_array]
    result = sum(np.array(diag_array) - np.array(max_out))
    return result, diag_zeros
def comp_diag_average_max_out(matrix):
    diag_array =  []
    diag_zeros = 0
    out_array = []
    for i in range(len(matrix[0])):
         diag_item = matrix[i][i]
         if diag_item == 0:
             diag_zeros += 1
             continue
         diag_array.append(1)
         out_row = []
         for j in range(len(matrix[i])):
             if i != j:
                 out_row.append(matrix[i][j]/diag_item)
         out_array.append(out_row)

    max_out = [max(r) for r in out_array]
    result = sum(np.array(diag_array) - np.array(max_out))
    return result, diag_zeros

metrics = {
    "diag_average_average_out": comp_diag_average_average_out,
    "diag_average_max_out": comp_diag_average_max_out,
    "std_error": comp_std_error
}

def max_on_diag(matrix):
    diag = [matrix[i,i] for i in range(len(matrix))]
    return np.max(diag)

def min_on_diag(matrix):
    diag = [matrix[i,i] for i in range(len(matrix))]
    return np.min(diag)

def average_diag(matrix):
    diag = [matrix[i,i] for i in range(len(matrix))]
    return np.average(diag)

def comp_std_error_diag(matrix):
    diag = [matrix[i, i] for i in range(len(matrix))]
    return np.std(diag)

metrics_sp = {
    "std_error_sp": comp_std_error_diag,
    "max_on_diag": max_on_diag,
    "min_on_diag": min_on_diag,
    "average_diag": average_diag
}

def comp_pop_metrics():
    for m in metrics:
        res_string = ""
        for map in layouts_onions:
        #for map in ["pipeline"]:
           for stack in ["chan","tupl","nost"]:
                path = f"{PATH_PREFIX}{map}/{stack}_{map}_ref-30"
                if os.path.isfile(path):
                    print(f"nalezeno {path}")
                    try:
                        a = np.loadtxt(path)
                        comp1= metrics[m](a)
                        res_string += f"{stack},{map},ref-30,{comp1}\n"
                    except Exception as e:
                        print("chyba")
                        print(e)
                else:
                    print("not found")
        print(res_string)
        with open(f'{PATH_PREFIX}metrics/{m}.txt', 'w') as f:
            print(res_string, file=f)
        print("end")

# spocitat metriky pro to jak jsou tezke jednotlive mapy viz obsidian
def comp_sp_metrics():
    for m in metrics_sp:
        res_string = ""
        z = 0
        for map in layouts_onions:
            for stack in ["chan", "tupl", "nost"]:
                path = f"./{map}/{stack}_{map}_ref-30"
                if os.path.isfile(path):
                    print(f"nalezeno {path}")
                    try:
                        a = np.loadtxt(path)
                        comp1 = metrics_sp[m](a)
                        res_string += f"{stack},{map},ref-30,{comp1}\n"
                    except Exception as e:
                        print("chyba")
                        print(e)
                else:
                    print("not found")
        print(res_string)
        with open(f'./metrics/{m}_SP.txt', 'w') as f:
            print(res_string, file=f)
        print("end")

comp_sp_metrics()