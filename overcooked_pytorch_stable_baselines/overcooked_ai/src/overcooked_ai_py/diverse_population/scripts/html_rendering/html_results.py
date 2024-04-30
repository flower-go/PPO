#import jinja2
#environment = jinja2.Environment()

#template = environment.from_string("Hello, {{ name }}!")
#a = template.render(name="World")
#print(a)

# write_messages.py
CODE_PATH = "C:/Users/PetraVysušilová/Documents/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/"
MDP_PATH = "C:/Users/PetraVysušilová/Documents/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/mdp/"
heat_path = "visualisation/"

from jinja2 import Environment, FileSystemLoader


max_score = 100
test_name = "Python Challenge"
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

environment = Environment(loader=FileSystemLoader("C:\\Users\\PetraVysušilová\\Documents\\coding\\PPO\\overcooked_pytorch_stable_baselines\\overcooked_ai\\src\\overcooked_ai_py\\diverse_population\\scripts\\html_rendering\\templates"))
template = environment.get_template("results.txt")
res_dict = {}

for i in layouts_onions:
    res_dict[i] = {}
    for s in frame_stacking:
        res_dict[i][frame_stacking[s]] = {}
    res_dict[i]["general"] = {}

def heat_maps():
    result = []
    path = CODE_PATH + heat_path
    for map in layouts_onions:
        for s in frame_stacking:
            file = f"{path}/{map}/{s}_{map}_ref-30_reordered.png"
            res_dict[map][frame_stacking[s]]["heat"] = file
            #result.append(file)

def heat_steps():
    path = CODE_PATH + heat_path
    for map in layouts_onions:
        for s in frame_stacking:
            file = f"{path}/{map}/steps2754060_{s}_{map}_ref-30_reordered.png"
            res_dict[map][frame_stacking[s]]["heat_s2"] = file
            file = f"{path}/{map}/steps1377030_{s}_{map}_ref-30_reordered.png"
            res_dict[map][frame_stacking[s]]["heat_s1"] = file
def map_image():
    path = CODE_PATH +  "visualisation/maps"
    for map in layouts_onions:
        for s in frame_stacking:
            file = f"{path}/{map}.png"
            res_dict[map]["general"]["layout"] = file


heat_maps()
heat_steps()
map_image()
with open("results.html", mode="w", encoding="utf-8") as results:
    results.write(template.render(maps=layouts_onions, res = res_dict))
    print(f"... wrote {results}")



