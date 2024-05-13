import numpy as np
import os
import pandas as pd
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
#a = np.loadtxt("coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/evaluation/five_by_five/nost_coordination_ring_ref-30_ROP0.0_MSP_BRCoef0.0_BRClip0.0_LCoef0.0_LClip0.0_DSRFalse_PADFalse_NSP-1_X_nost_coordination_ring_ref-30_ROP0.0_MSP_BRCoef0.0_BRClip0.0_LCoef0.0_LClip0.0_DSRFalse_PADFalse_NSP-1_ENVROP0.0")
#a = np.loadtxt("coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/evaluation/coordination_ring/nost_coordination_ring_ref-30_ROP0.0_MSP_BRCoef0.0_BRClip0.0_LCoef0.0_LClip0.0_DSRFalse_PADFalse_NSP-1_X_nost_coordination_ring_ref-30_ROP0.0_MSP_BRCoef0.0_BRClip0.0_LCoef0.0_LClip0.0_DSRFalse_PADFalse_NSP-1_ENVROP0.0")
import scipy
import scipy.cluster.hierarchy as sch
PATH_PREFIX = "/storage/plzen1/home/ayshi/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/evaluation/"
def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    return corr_array[idx, :][:, idx]

def heat_map(file_path,layout_name, stack):
    """
    Visualises evaluated cross-play table as heat map
    """
    table = np.loadtxt(file_path)
    print("zacinam s heat map")
    #print(table)
    #table = np.around(table, decimals=2)
    fig, ax = plt.subplots()
    im = ax.imshow(table)

    ax.set_xticks(np.arange(len(table[0])))
    ax.set_yticks(np.arange(len(table)))

    plt.xlabel("player 1 agent")
    plt.ylabel("player 2 agent")

    cbar = plt.colorbar(im)
    cbar.set_label('Avg. cumulative reward')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fig.tight_layout()
    vis_path = "/storage/plzen1/home/ayshi/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/visualisation/"
    file_name = vis_path + layout_name + "/" + stack + "_"+ layout_name + "_ref-30_reordered" + ".png"
    print("jmeno filu je:" + file_name)
    plt.savefig(file_name)

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
#np.savetxt("./before",a)
for map in layouts_onions:
    for stack in ["nost","chan","tupl"]:	
        
        #prefix = "steps2754060_"
        prefix = ""
        path = f"{PATH_PREFIX}{map}/{prefix}{stack}_{map}_ref-30"
        if os.path.isfile(path):
            print(f"nalezeno {path}")
            try:
                a = np.loadtxt(path)
                if os.path.isfile(path + "_reordered"):
                    continue
                else:
                    reordered = cluster_corr(a)
                    np.savetxt(path + "_reordered", reordered)
                    heat_map(path + "_reordered", map, prefix + stack)
            except Exception as e:
                print("chyba")
                print(e)
        else:
            print(f"not found: {path}")
