import ast
import os
#file_name = "./diverse_population/visualisation/evaluation/test_file"

def load_data(filename):
    print("cwd")
    print(os. getcwd())
    print("filename {0}", filename)
    file1 = open(filename, 'r')
    terrain = None
    reward = None
    next_state = None
    result = {}
    layout_name = None

    for line in file1:
        if line.startswith("layout_name"):
            layout_name = line[11:]
        if line.startswith("grid"):
            terrain = ast.literal_eval(line[5:])
        try:
            if line.startswith("t:"):
                thread = str(line.split(":", 1)[1])
            if line.startswith("r:"):
                reward = float(line.split(":", 1)[1])
                if result.get(thread) is None:
                    result[thread] = [{"action": joint_action, "reward": reward, "next_state": next_state }]
                else:
                    result[thread].append({"action": joint_action, "reward": reward, "next_state": next_state })
                continue
            elif line.startswith("j:"):
                joint_action = ast.literal_eval(line.split(":", 1)[1])
            elif line.startswith("n:"):
                next_state = ast.literal_eval(line.split(":", 1)[1])
        except:
            print("error")
            print(line)

    return layout_name, terrain, result
