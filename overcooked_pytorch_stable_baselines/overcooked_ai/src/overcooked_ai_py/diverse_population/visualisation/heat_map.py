import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os


file_path = "./reordered_five"
file2 = "./before"
def heat_map(file_path,layout_name = "five_by_five", deterministic=True, eval_env=""):
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
    file_name = file_path + "_heat" + ".png"
    print("jmeno filu je:" + file_name)
    plt.savefig(file_name)

heat_map(file_path)
heat_map(file2)
