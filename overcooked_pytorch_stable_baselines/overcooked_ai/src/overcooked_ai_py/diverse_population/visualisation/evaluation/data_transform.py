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

    with open(filename, "r") as file:
        for line in file:
            if len(line) > 1:
                try:
                    data = ast.literal_eval(line.strip())
                    reward += data["reward:"]
                    data["cumulative_reward"] = reward
                    result.append(data)

                except:
                    print("error on this line")
                    print(line)
                    print(len(line))


    return result
