import ast
import os
#file_name = "./diverse_population/visualisation/evaluation/test_file"

def load_data(filename):
    print("cwd")
    print(os. getcwd())
    print("filename {0}", filename)
    file1 = open(filename, 'r')

    result = []

    with open(filename, "r") as file:
        for line in file:
            # Remove leading/trailing whitespace and load the JSON data
            data = json.loads(line.strip())

            # Append the parsed dictionary to the list
            result.append(data)

    return layout_name, terrain, result
