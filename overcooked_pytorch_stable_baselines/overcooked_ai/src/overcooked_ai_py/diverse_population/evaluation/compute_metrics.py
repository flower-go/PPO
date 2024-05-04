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
PATH_PREFIX = "/storage/plzen1/home/ayshi/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/evaluation/"


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


def comp_diag_average_average_out(matrix):
    print("diag average average out")
    diag_array =  []
    out_array = []
    for i in range(len(matrix[0])):
         diag_item = matrix[i][i]
         diag_array.append(1)
         out_row = []
         for j in range(len(matrix[i])):
             if i != j:
                 #print(f"matrix[i][j] is {matrix[i][j]} and diag item is {diag_item}")
                 out_row.append(matrix[i][j]/diag_item)
         out_array.append(out_row)

    max_out = [mean(r) for r in out_array]
    #print(f"max out is {max_out}")
    result = sum(np.array(diag_array) - np.array(max_out))
    return result
def comp_diag_average_max_out(matrix):
    print("diag_average_max_out")
    diag_array =  []
    out_array = []
    for i in range(len(matrix[0])):
         diag_item = matrix[i][i]
         diag_array.append(1)
         out_row = []
         for j in range(len(matrix[i])):
             if i != j:
                 #print(f"matrix[i][j] is {matrix[i][j]} and diag item is {diag_item}")
                 out_row.append(matrix[i][j]/diag_item)
         out_array.append(out_row)

    max_out = [max(r) for r in out_array]
    #print(f"max out is {max_out}")
    result = sum(np.array(diag_array) - np.array(max_out))
    return result
res_string = ""
for map in layouts_onions:
    for stack in ["chan","tupl","nost"]:
        path = f"{PATH_PREFIX}{map}/{stack}_{map}_ref-30"
        if os.path.isfile(path):
            print(f"nalezeno {path}")
            try:
                a = np.loadtxt(path)

                comp1 =comp_diag_average_max_out(a)

                res_string += f"{stack},{map},ref-30,{comp1}\n"
            except Exception as e:
                print("chyba")
                print(e)
        else:
            print(f"not found: {path}")
print(res_string)
