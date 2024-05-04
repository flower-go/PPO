import os
# Function to rename multiple files
def main():
        print(os.listdir("./"))
        path="./diverse_population/visualisation/maps/"
        for dir in os.listdir(path):
            print(dir)
            if os.path.isdir(path + dir):
                print("je to adresar")
                for dir2 in os.listdir(path + dir):
                    if "maps" in dir2:
                        my_source = path + dir + "/" + dir2
                        my_dest = my_source.replace("maps", "")
                        #os.rename(my_source, my_dest)
                        print(my_source)
                        print(my_dest)
# Driver Code
if __name__ == '__main__':
	# Calling main() function
	main()
