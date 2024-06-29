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

def show_sorted_cross_play(name,matrices, legends, title="", remove_diag=False, quantile=0.15,draw=True):
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

    if draw:
        print(f"ukladam {name}")
        plt.savefig(f"{name}.png", dpi=300)
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
            sp = remove_daigonal(sp) #TODO potreba mit diagonalu take na vypocet pod krivkou? ne, chci to jak je to dobre mimo
            sp = scale_matrix(sp)
            matrices.append(sp)
            labels.append("SP")

            auc = show_sorted_cross_play(name=f"{layout}\\{s}_{layout}_quant{percentile}_all",matrices=matrices, legends=labels, title="Ordered evaluation results", remove_diag=False, quantile=percentile,draw=False)
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

            auc = show_sorted_cross_play(name=f"{layout}\\{s}_{layout}_quant{percentile}_best",matrices=matrices, legends=labels, title="Ordered evaluation results", remove_diag=False, quantile=percentile, draw=False)
            auc_best.extend(auc)
    return auc_best

stacking = ["chan", "tupl", "nost"]
exp_type = ["SP", "L0", "L1", "L2", "R0", "R1", "R2", "R0L0", "R1L1"]

def load_best_by_auc( percentile):
    ord_res = {}
    for s in stacking:
        ord_res[s] = {}
        for map in layouts_onions:
            ord_res[s][map] = {}
            for e in exp_type:
                ord_res[s][map][e] = np.zeros(12)
    for i in range(3, 12):
        filename = f"../evaluation/metrics/auc_{i}_{percentile*100}.txt"

        with open(file=filename, mode='r') as res_file:
            for line in res_file:
                if len(line) > 0:
                    splitted = line.split(",")
                    stack = splitted[0]
                    layout = splitted[1]
                    e_type = splitted[2]
                ord_res[stack][layout][e_type][i] = splitted[3]
    return ord_res


def load_best_by_auc_SP(percentile):
    ord_res = {}
    for s in stacking:
        ord_res[s] = {}
        for map in layouts_onions:
            ord_res[s][map] = {}
            for e in exp_type:
                ord_res[s][map][e] = np.zeros(30)
    for i in range(0, 30):
        filename = f"../evaluation/metrics/aucSP_{i}_{percentile * 100}.txt"

        with open(file=filename, mode='r') as res_file:
            for line in res_file:
                if len(line) > 0:
                    splitted = line.split(",")
                    stack = splitted[0]
                    layout = splitted[1]
                    e_type = splitted[2]
                ord_res[stack][layout][e_type][i] = splitted[3]
    return ord_res

def select_best_from_pop(matrix):
    row_1 = matrix[[0]]
    row_2 = matrix[[1]]

    best = get_sorted_pairwise_best(row_1,row_2)

    for i in range(2,len(matrix)):
        best = get_sorted_pairwise_best(best,matrix[[i]])

    return best

def print_best_POP(percentile=0.15):
    auc_best = []
    auc_table = load_best_by_auc(percentile)
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
                #m = select_best_from_pop(m[3:])
                best_index = np.argmax(auc_table[s][layout][e])
                m = m[best_index]
                m = scale_matrix(m) #pridelam body at mam hladkou primku
                matrices.append(m)
                labels.append(e)
            sp = remove_daigonal(sp)
            sp = scale_matrix(sp)
            matrices.append(sp)
            labels.append("SP")

            auc = show_sorted_cross_play(name=f"{layout}\\{s}_{layout}_quant{int(percentile*100)}_bestPOP",matrices=matrices, legends=labels, title="Ordered evaluation results", remove_diag=False, quantile=percentile, draw=True)
            auc_best.extend(auc)
    return auc_best

def print_best_POP_best_SP(percentile=0.15):
    auc_best = []
    auc_table = load_best_by_auc(percentile)
    auc_sp_table = load_best_by_auc_SP(percentile)
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
                best_index = np.argmax(auc_table[s][layout][e])
                m = m[[best_index]]
                m = scale_matrix(m) #pridelam body at mam hladkou primku
                matrices.append(m)
                labels.append(e)
            sp = remove_daigonal(sp)
            sp_best = np.argmax(auc_sp_table[s][layout][e])
            sp = sp[[sp_best]]
            sp = scale_matrix(sp)
            matrices.append(sp)
            labels.append("SP")

            auc = show_sorted_cross_play(name=f"{layout}\\{s}_{layout}_quant{int(percentile*100)}_bestPOPSP",matrices=matrices, legends=labels, title="Ordered evaluation results", remove_diag=False, quantile=percentile, draw=True)
            auc_best.extend(auc)
    return auc_best

def print_best_final_best_SP(percentile=0.15):
    auc_best = []
    auc_sp_table = load_best_by_auc_SP(percentile)
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
                m =  get_sorted_pairwise_best(m[[11]], m[[12]])
                m = scale_matrix(m) #pridelam body at mam hladkou primku
                matrices.append(m)
                labels.append(e)
            sp = remove_daigonal(sp)
            sp_best = np.argmax(auc_sp_table[s][layout][e])
            sp = sp[[sp_best]]
            sp = scale_matrix(sp)
            matrices.append(sp)
            labels.append("SP")

            auc = show_sorted_cross_play(name=f"{layout}\\{s}_{layout}_quant{int(percentile*100)}_bestfinalSP",matrices=matrices, legends=labels, title="Ordered evaluation results", remove_diag=False, quantile=percentile, draw=True)
            auc_best.extend(auc)
    return auc_best

def print_agent_on_index(a_i,percentile=0.15):
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
                m = np.sort(m[[a_i]])
                m = scale_matrix(m) #pridelam body at mam hladkou primku
                matrices.append(m)
                labels.append(e)
            sp = remove_daigonal(sp)
            sp = scale_matrix(sp)
            matrices.append(sp)
            labels.append("SP")

            auc = show_sorted_cross_play(name=f"{layout}\\{s}_{layout}_quant{percentile}_best",matrices=matrices, legends=labels, title="Ordered evaluation results", remove_diag=False, quantile=percentile,draw=False)
            auc_best.extend(auc)
    return auc_best


def print_SP_on_index(a_i,percentile=0.15):
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
            sp = remove_daigonal(sp)
            sp = sp[[a_i]]
            sp = scale_matrix(sp)
            matrices.append(sp)
            labels.append("SP")

            auc = show_sorted_cross_play(name=f"{layout}\\{s}_{layout}_quant{percentile}_SP{a_i}",matrices=matrices, legends=labels, title="Ordered evaluation results", remove_diag=False, quantile=percentile,draw=False)
            auc_best.extend(auc)
    return auc_best


metric_prefix = "../evaluation/"
percentile = 0.15

def execute_index_percentile():
    for a_i in range(3,12):
        auc_i = print_agent_on_index(a_i,percentile)
        with open(f'{metric_prefix}metrics/auc_{a_i}_{percentile*100}.txt', 'w') as f:
           for a in auc_i:
               print(a, file=f)
        print("end")

def execute_SP_percentile():
    for a_i in range(0, 30):
        auc_i = print_SP_on_index(a_i, percentile)
        with open(f'{metric_prefix}metrics/aucSP_{a_i}_{percentile * 100}.txt', 'w') as f:
            for a in auc_i:
                print(a, file=f)
        print("end")


def execute_best_and_all():
    auc = print_all(percentile=percentile)
    auc_best = print_best_final(percentile=percentile)
    with open(f'{metric_prefix}metrics/auc_{percentile*100}.txt', 'w') as f:
        for a in auc:
            print(a, file=f)
    print("end")

    with open(f'{metric_prefix}metrics/auc_best_{percentile*100}.txt', 'w') as f:
        for a in auc_best:
            print(a, file=f)
    print("end")

def execute_best_pop():
    auc_best = print_best_POP(percentile=percentile)

    with open(f'{metric_prefix}metrics/auc_best_POP_{percentile*100}.txt', 'w') as f:
        for a in auc_best:
            print(a, file=f)
    print("end")

def execute_best_popSP():
    auc_best = print_best_POP_best_SP(percentile=percentile)

    with open(f'{metric_prefix}metrics/auc_best_POPSSP_{percentile*100}.txt', 'w') as f:
        for a in auc_best:
            print(a, file=f)
    print("end")

def execute_best_finalSP():
    auc_best = print_best_final_best_SP(percentile=percentile)
    with open(f'{metric_prefix}metrics/auc_best_finalSP_{percentile*100}.txt', 'w') as f:
        for a in auc_best:
            print(a, file=f)
    print("end")

#execute_best_and_all()
#execute_best_pop()
execute_best_popSP()
execute_best_finalSP()
#execute_SP_percentile()
