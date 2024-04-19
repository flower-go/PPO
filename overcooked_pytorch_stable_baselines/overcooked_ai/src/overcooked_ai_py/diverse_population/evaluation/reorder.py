import numpy as np
import os
import pandas as pd
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
    for stack in ["nost","chan","tupl"]
        path = f"{PATH_PREFIX}{map}/{stack}_{map}_ref-30"
        if os.path.isfile(path):
            print(f"nalezeno {path}")
            a = np.loadtxt(path)
            reordered = cluster_corr(a)
            np.savetxt(path + "_reordered", reordered)
        else:
            print(f"not found: {path}")
