import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

def heat_map(table, group_name, layout_name, deterministic=True, eval_env=""):
    """
    Visualises evaluated cross-play table as heat map
    """
    print("zacinam s heat map")
    print(table)
    table = np.around(table, decimals=2)
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
    file_dir = f"{os.environ['PROJDIR']}/diverse_population/visualisation/{layout_name}/"
    file_name = file_dir + group_name + ('' if deterministic else '_STOCH')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = file_name + eval_env + ".png"
    print("jmeno filu je:" + file_name)
    plt.savefig(file_name)
    return file_name
