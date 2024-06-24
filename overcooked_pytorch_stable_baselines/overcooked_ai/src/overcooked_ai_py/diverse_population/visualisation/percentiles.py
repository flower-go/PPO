import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from scipy import integrate

def comp_area_under_curve(table, quantile = 0.15):
    table = np.around(table, decimals=2)
    table = np.sort(table)
    x = range(len(table[0]))
    y = np.quantile(table, quantile, axis=0)

    #pouzit metodu trapezoids ze scikit
    integrated = round(integrate.trapezoid(y, x=x),2)
    return integrated
    #TODO ulozit do spravneho souboru

size = 2000

def scale_matrix(matrix):
  extended_matrix = []
  for row in matrix:
    new_row = np.array([])
    row = np.sort(row)
    for pos in range(len(row) - 1):
      new_row = np.concatenate([new_row, np.linspace(row[pos], row[pos+1], size//len(row))])
    extended_matrix.append(new_row)
  return np.array(extended_matrix)

def get_sorted_pairwise_best(row1, row2):
  row1_sorted = np.sort(row1)
  row2_sorted = np.sort(row2)
  if np.sum(row1_sorted >= row2_sorted) > np.sum(row2_sorted >= row1_sorted):
    return row1
  else:
    return row2

def remove_daigonal(table):
  return table[~np.eye(table.shape[0],dtype=bool)].reshape(table.shape[0],-1)

def show_sorted_cross_play(name,matrices, legends, title="", remove_diag=False, quantile=0.15,):
    labels = []
    auc = []
    for matrix, legend in zip(matrices, legends):
        l = name.split('\\')
        auc_name = f"{l[1].split("_")[0]},{l[0]},{legend}"
        auc_res = comp_area_under_curve(matrix,quantile)
        auc.append(f"{auc_name},{auc_res}")
        table = np.around(matrix, decimals=2)
        if remove_diag:
            table = remove_daigonal(table)
        table = np.sort(table)

        color = None
        if "SP" in legend:
            color = 'black'
            # ax = plt.plot(range(len(table[0])), np.mean(table, axis=0), label=legend, color=color)
            ax = plt.plot(range(len(table[0])), np.quantile(table, quantile, axis=0), label=legend, color=color)
            continue

        # ax = plt.plot(range(len(table[0])), np.mean(table, axis=0), label=legend, color=color)
        ax = plt.plot(range(len(table[0])), np.quantile(table, quantile, axis=0), label=legend, color=color)

        #plt.fill_between(x=range(len(table[0])), y1=np.quantile(table,0.25, axis=0), y2=np.quantile(table,0.75, axis=0), alpha=0.25)
        plt.fill_between(x=range(len(table[0])), y1=np.quantile(table, 0.0, axis=0),y2=np.quantile(table, 0.30, axis=0), alpha=0.25)

    plt.xlabel("ordered agents")
    percentil = np.arange(0, 1.01, 0.1)
    plt.xticks((len(table[0]) - 1) * percentil, map(str, np.round(percentil, 1)))
    plt.ylabel("Average cummulative reward")
    plt.legend()

    plt.title(title)
    print(f"ukladam {name}")
    #plt.savefig(f"{name}.png", dpi=300)
    plt.clf()
    #plt.show()
    return auc

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

frame_stacking = {"chan":"channels",
                  "tupl":"tuple",
                  "nost":"nostack", #effectively no stacking
                  }

exp_names = ["R0","R1", "R2", "L0", "L1", "L2", "R0L0", "R1L1"]
#eval_path = "..\\evaluation\\"
eval_path = "C:/Users/PETRAV~1/PYCHAR~1/coding/PPO/OVERCO~1/OVERCO~1/src/OVERCO~1/DIVERS~1/evaluation/"

def print_all(percentile=0.15):
    auc_all = []
    for layout in layouts_onions:
        for s in frame_stacking:
            matrices = []
            labels = []
            try:
                sp = np.loadtxt(f"{eval_path}{layout}\\{s}_{layout}_ref-30")
                print("SP found")
            except:
                print("nenalezeno")
                continue
            for e in exp_names:
                file = f"{eval_path}{layout}/{s}_{layout}_{e}_X_{s}_{layout}_ref-30_ENVROP0.0"
                try:
                    m = np.loadtxt(file)
                except Exception as x:
                    print(f"not found {layout}{s}{e}")
                    print(f"{file}")
                    print(x)
                    continue
                m = m[3:] #neuvazuju init SP agenty
                m = scale_matrix(m) #pridelam body at mam hladkou primku
                matrices.append(m)
                labels.append(e)
            sp = remove_daigonal(sp) #TODO potreba mit diagonalu take na vypocet pod krivkou
            sp = scale_matrix(sp)
            matrices.append(sp)
            labels.append("SP")

            auc = show_sorted_cross_play(name=f"{layout}\\{s}_{layout}_quant{percentile}_all",matrices=matrices, legends=labels, title="Ordered evaluation results", remove_diag=False, quantile=percentile)
            auc_all.extend(auc)
    return auc_all
def print_best_final(percentile=0.15):
    auc_best = []
    for layout in layouts_onions:
        for s in frame_stacking:
            matrices = []
            labels = []
            try:
                sp = np.loadtxt(f"{eval_path}{layout}\\{s}_{layout}_ref-30")
                print("SP found")
            except:
                print(f"nenalezeno SP {s}{layout}")
                continue
            for e in exp_names:
                file = f"{eval_path}{layout}\\{s}_{layout}_{e}_X_{s}_{layout}_ref-30_ENVROP0.0"
                try:
                    m = np.loadtxt(file)
                except Exception as x:
                    print(f"not found {layout}{s}{e}")
                    print(file)
                    print(x)
                    continue
                m = get_sorted_pairwise_best(m[[11]], m[[12]])
                m = scale_matrix(m) #pridelam body at mam hladkou primku
                matrices.append(m)
                labels.append(e)
            sp = remove_daigonal(sp)
            sp = scale_matrix(sp)
            matrices.append(sp)
            labels.append("SP")

            auc = show_sorted_cross_play(name=f"{layout}\\{s}_{layout}_quant{percentile}_best",matrices=matrices, legends=labels, title="Ordered evaluation results", remove_diag=False, quantile=percentile)
            auc_best.extend(auc)
    return auc_best

percentile = 0.30
auc = print_all(percentile=percentile)
auc_best = print_best_final(percentile=percentile)

metric_prefix = "../evaluation/"


with open(f'{metric_prefix}metrics/auc_{percentile*100}.txt', 'w') as f:
    for a in auc:
        print(a, file=f)
print("end")

with open(f'{metric_prefix}metrics/auc_best_{percentile*100}.txt', 'w') as f:
    for a in auc_best:
        print(a, file=f)
print("end")
