import sys
import os
import datetime

codedir = os.environ["CODEDIR"]
#codedir = /home/premek/DP/
projdir = os.environ["PROJDIR"]
#projdir = /home/premek/DP/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_pytorch
sys.path.append(codedir + "/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src")
sys.path.append(codedir + "/PPO/overcooked_pytorch_stable_baselines/stable-baselines3")
sys.path.append(codedir + "/PPO/overcooked_pytorch_stable_baselines")


from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import (
    SampleAgent,
    GreedyHumanModel,
    RandomAgent,
)
from overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld,
    OvercookedState,
    Recipe,
    SoupState,
)
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.visualization.visualization_utils import (
    show_image_in_ipython,
)
from overcooked_ai_py.utils import generate_temporary_file_path
from overcooked_ai_py.static import FONTS_DIR
from overcooked_ai_py.mdp.layout_generator import POT
import copy
import pygame
import os
import numpy as np
import json
os.environ['SDL_AUDIODRIVER'] = 'dsp'

#grid a state
#ONE_STEP_INPUT = [[['X', 'X', 'P', 'X', 'X'], ['T', ' ', ' ', ' ', 'T'], ['X', ' ', ' ', ' ', 'O'], ['X', 'X', 'D', 'S', 'X']],[]]

DEFAULT_VALUES = {
    "height": None,  # if None use grid_width - NOTE: can chop down hud if hud is wider than grid
    "width": None,  # if None use (hud_height+grid_height)
    "tile_size": 75,
    "window_fps": 30,
    "player_colors": ["blue", "green"],
    "is_rendering_hud": True,
    "hud_font_size": 10,
    "hud_system_font_name": None,  # if set to None use hud_font_path
    # needs to be overwritten with default - every pc has different pathes "hud_font_path": roboto_path,
    "hud_font_color": (255, 255, 255),  # white
    "hud_data_default_key_order": [
        "all_orders",
        "bonus_orders",
        "time_left",
        "score",
        "potential",
    ],
    "hud_interline_size": 10,
    "hud_margin_bottom": 10,
    "hud_margin_top": 10,
    "hud_margin_left": 10,
    "hud_distance_between_orders": 5,
    "hud_order_size": 15,
    "is_rendering_cooking_timer": True,
    "show_timer_when_cooked": True,
    "cooking_timer_font_size": 20,  # # if set to None use cooking_timer_font_path
    # needs to be overwritten with default - every pc has different pathes "cooking_timer_font_path": roboto_path,
    "cooking_timer_system_font_name": None,
    "cooking_timer_font_color": (255, 0, 0),  # red
    "grid": None,
    "background_color": (155, 101, 0),  # color of empty counter
}

Recipe.configure({})
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", default=None, type=str, help="path to file with behavior records in required format")
parser.add_argument("--output_file", default=None, type=str, help="path to the result destination")

args = parser.parse_args([] if "__file__" not in globals() else None)
def save_map_pic(filename, img_path):
    test_dict = file_to_dict(filename)
    print(filename)

    surface = StateVisualizer(**test_dict["config"]).render_state(
        **test_dict["kwargs"]
    )
    pygame.image.save(surface, img_path)

def file_to_dict(filename):
    #TODO tady napsat nacitani z file
    config = copy.deepcopy(DEFAULT_VALUES)
    #TODO zatim natvrdo
    grid = [['X', 'O', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'], ['X', ' ', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'], ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'], ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'P'], ['X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'X'], ['X', ' ', 'D', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'], ['X', 'S', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X']]
    all_orders = [{'ingredients': ['onion', 'onion', 'onion']}, {'ingredients': ['onion', 'onion']}, {'ingredients': ['onion']}]
    state = {
            "players": [{"position": (4, 4), "orientation": (0, 1), "held_object": None}, {"position": (11, 3),
                                                                                           "orientation": (1, 0), "held_object": None}],
            "objects": [{"name": 'onion', "position": (6, 1)}],
            "bonus_orders": [],
            "all_orders": [{'ingredients': ['onion', 'onion', 'onion']}, {'ingredients': ['onion', 'onion']}, {'ingredients': ['onion']}],
            "timestep": 292,
        }
    hud_data = {
    #    "all_orders": trajectory_random_pair["mdp_params"][0]["start_all_orders"]
        "all_orders": all_orders
    }
    kwargs = {"hud_data": hud_data, "grid": grid, "state": state}
    test_dict = {
        "config": config,
        "kwargs": kwargs,
        "comment": "Test simple recipes in hud. NOTE: failing to render stuff outside HUD also fails this test",
        "result_array_filename": "test_hud_2.npy",
    }
    copy.deepcopy(test_dict)
    test_dict["kwargs"]["state"] = OvercookedState.from_dict(
        test_dict["kwargs"]["state"]
    )

    return test_dict

if __name__ == "__main__":
    if args.output_file is None:
        args.output_file = "./diverse_population/visualisation/maps/eval_vis_" + str(datetime.datetime.now()) + ".png"
    save_map_pic(args.input_file, args.output_file)




