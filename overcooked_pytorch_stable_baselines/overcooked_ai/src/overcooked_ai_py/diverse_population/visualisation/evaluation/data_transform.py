import ast

file_name = "./diverse_population/visualisation/evaluation/test_file"

def load_data(filename = file_name):
    file1 = open(filename, 'r')
    terrain = None
    reward = None
    next_state = None
    result = []
    reading_log = False
    layout_name = None

    for line in file1:
        if line.startswith("layout_name"):
            layout_name = line.split(" ")[1]
        if line.startswith("grid"):
            print(line)
            terrain = ast.literal_eval(line[5:])
        if line[0:6] == "loguju":
            reading_log = True
        if reading_log:
            if line.startswith("r:"):
                reward = float(line.split(":", 1)[1])
                result.append({"action": joint_action, "reward": reward, "next_state": next_state })
                continue
            elif line.startswith("j:"):
                joint_action = ast.literal_eval(line.split(":", 1)[1])
            elif line.startswith("n:"):
                try:
                    next_state = ast.literal_eval(line.split(":", 1)[1])
                except:
                    print("e")
                    print(line)

    file1.close()
    return layout_name, terrain, result
