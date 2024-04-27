#for each number, name here
    #grep it from ls wnadb
    #while you have it, go to files/images and copy heat map to corect place using name

import pandas as pd
import shutil
import os
import numpy as np
table_path="/storage/plzen1/home/ayshi/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/scripts/wand_table"
table = pd.read_csv(table_path, names=["name","fin","date","num","date2","id","server","nonsece","map","name2","some_bool","wandb"])
table["run_id"] = table["id"].map(lambda a: a.split(".")[0])

table = table.drop_duplicates("name")
dupl = table[table.duplicated('name', keep=False) == True]

wandb_path ="/storage/plzen1/home/ayshi/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/wandb"

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(wandb_path) if not isfile(join(wandb_path, f))]
print(onlyfiles)
files_dict = {}
for f in onlyfiles:
    id = f.split(".")[0].split("-")[-1]
    files_dict[id] = f

new_path = "/storage/plzen1/home/ayshi/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/visualisation/"
new_path_eval = "/storage/plzen1/home/ayshi/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/evaluation/"
#print(files_dict)
table = table.reset_index()
  # make sure indexes pair with number of rows
def move_images():
    for index, row in table.iterrows():
        id = row["run_id"]
        f  = files_dict[id]
        exp_name = row["name"]
        print(f"name is {row['name']}, id is {id} and file is {f}")
        image_file_dir = f"{wandb_path}/{f}/files/media/images"
        img_files=listdir(image_file_dir)
        print(f"image file is {img_files[0]}")
        new_img_path = f"{new_path}{row['map']}/{exp_name}"
        print(f"new img path is {new_img_path}")
        print(f"old path is {image_file_dir}")
        # 2nd option
        os.remove(new_img_path)
        shutil.copy(image_file_dir + "/" + img_files[0], new_img_path + ".png")
def extract_eval_table():
    for index, row in table.iterrows():
        id = row["run_id"]
        f  = files_dict[id]
        exp_name = row["name"]
        print(f"name is {row['name']}, id is {id} and file is {f}")
        log_file_path = f"{wandb_path}/{f}/files/output.log"
        new_img_path = f"{new_path_eval}{row['map']}/{exp_name}.npy"
        print(f"new img path is {new_img_path}")
        print(f"old path is {log_file_path}")
        append_t=False
        eval_table = ""
        with open(log_file_path) as file:
            for line in file:
                if append_t and "jmeno" in line:
                    print("koncime")
                    append_t = False
                if append_t:
                    eval_table += line
                if append_t == False and line.startswith("zacinam s heat map"):
                    append_t=True
        print(eval_table)
        a = np.array(eval_table)
        print("fdsaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        print(a)
        np.save(new_img_path,a)

extract_eval_table()
        

print("end")
