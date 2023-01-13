import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def heat_map(table, title, file_name, args, deterministic=True):
    table = np.around(table, decimals=2)
    # table[table <= 179] = 0


    fig, ax = plt.subplots()
    im = ax.imshow(table)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(table[0])))
    ax.set_yticks(np.arange(len(table)))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(table)):
        for j in range(len(table[0])):
            text = ax.text(j, i, table[i, j],
                           ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    # plt.savefig(f"diverse_population/visualisation/{args['layout_name']}/coordination_ring.png")
    file_name = f"diverse_population/visualisation/{args['layout_name']}/" + file_name + '' if deterministic else '_STOCH'
    file_name = file_name + ".png"
    plt.savefig(file_name)