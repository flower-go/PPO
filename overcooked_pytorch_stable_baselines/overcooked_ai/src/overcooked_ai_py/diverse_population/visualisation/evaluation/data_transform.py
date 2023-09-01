import ast

file_name = "/Users/petravysusilova/Documents/TR/research/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/visualisation/evaluation/test_file"
file1 = open(file_name, 'r')
count = 0
reading_log  = False
terrain = None
reward = None
next_state = None

# Using for loop
for line in file1:
    if line.startswith("grid"):
        print(line)
        terrain = ast.literal_eval(line[5:])
    if line[0:6] == "loguju":
        reading_log = True
    if reading_log:
        if line.startswith("rew"):
            reward = float(line[7:])
            continue
        elif line.startswith("join"):
            joint_action = ast.literal_eval(line.split(":", 1)[1])
        elif line.startswith("next"):
            try:
                next_state = ast.literal_eval(line[11:])
            except:
                print("e")


# Closing files
file1.close()
print(terrain)