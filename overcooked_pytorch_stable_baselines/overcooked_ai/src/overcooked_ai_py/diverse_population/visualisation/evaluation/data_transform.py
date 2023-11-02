import ast
import os
#file_name = "./diverse_population/visualisation/evaluation/test_file"
import ast
 
def load_data(filename):
    print("cwd")
    print(os. getcwd())
    print("filename {0}", filename)
    file1 = open(filename, 'r')
    result = []
    reward = 0
    grid = None
    name = None

    with open(filename, "r") as file:
        for i, line in enumerate(file):
            if i == 0:
                grid = ast.literal_eval(line[5:])
                continue
            if i == 2:
                name = line[5:]
                continue
            if len(line) > 1:
                try:
                    data = ast.literal_eval(line.strip())
                    if data["next_state"]["timestep"] == 0:
                        reward = 0
                    reward += data["reward"]
                    data["cumulative_reward"] = reward
                    result.append(data)  
                except Exception as e:
                    print("error on this line")
                    print(line)
                    print(e)

    return result, grid, name
