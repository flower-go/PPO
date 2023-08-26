import ast

file_name = "/Users/petravysusilova/Documents/TR/research/coding/PPO/vypis_vstup"
file1 = open(file_name, 'r')
count = 0
reading_log  = False
terrain = None

# Using for loop
for line in file1:
    if line[0:6] == "loguju":
        reading_log = True
        count = 1
    if reading_log:
        if count == 5:
            if terrain is None:
                print("thisss")
                print(line)
                terrain = ast.literal_eval(line)["terrain"]
            reading_log = False
            #print("pokracuju")
            continue
        #print(line)
        count += 1

# Closing files
file1.close()
print(terrain)