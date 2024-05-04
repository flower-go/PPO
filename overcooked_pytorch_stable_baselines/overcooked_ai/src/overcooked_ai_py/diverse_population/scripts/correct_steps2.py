import pandas as pd

table_path=("C:\\Users\\PetraVysušilová\\Documents\\research\\wand_table_all_steps2.csv")
table = pd.read_csv(table_path, names=["name","fin","date","num","date2","id","server","nonsece","map","name2","some_bool","wandb"])
table["run_id"] = table["id"].map(lambda a: a.split(".")[0])

table = table.drop_duplicates("name")
dupl = table[table.duplicated('name', keep=False) == True]

wandb_path =

table = table.reset_index()  # make sure indexes pair with number of rows
for index, row in table.iterrows():


print("end")