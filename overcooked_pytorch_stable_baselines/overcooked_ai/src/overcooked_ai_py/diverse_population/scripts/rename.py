import os
# Function to rename multiple files
CODE_PATH = "./diverse_population/"
heat_path = "evaluation/"
def prejmenovat():
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

def rename_R(r="R0"):
    path = CODE_PATH + heat_path
    for map in layouts_onions:
        for s in frame_stacking:
            file = f"{path}{map}/{s}_{map}_{r}_X_SP_EVAL2_ROP0.0_ENVROP0.0"
            if os.path.exists(file):
                print("existuje")
                new_name = f"{path}{map}/{s}_{map}_{r}_X_{s}_{map}_ref-30_ENVROP0.0"
                #print(new_name)
                os.rename(file,new_name)
            else:
                print(f"nenalezeno {file}")

layouts_onions = [
           "small_corridor",
           "five_by_five",
           "schelling",
           "centre_pots",
           "corridor",
           "pipeline",
           "scenario1_s",
           "large_room",
           "asymmetric_advantages", #tady
           "schelling_s",
            "coordination_ring", #tady
           "counter_circuit_o_1order", #tady
           "long_cook_time",
           "cramped_room", #tady
           "forced_coordination", #tady
           "m_shaped_s",
           "unident",
           "simple_o",
           "centre_objects",
           "scenario2_s",
           "scenario3",
           "scenario2",
           "scenario4",
           "bottleneck",
           "tutorial_0"]

frame_stacking = {"chan":"channels",
                  "tupl":"tuple",
                  "nost":"nostack", #effectively no stacking
                  }

exp_names = ["R0","R1", "R2", "L0", "L1", "L2", "R0L0", "R1L1"]
#prejmenovat()
for e in exp_names:
    rename_R(e)
