import sys
import os

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

def has_cooking_timer(state, grid):
    for obj in state.objects.values():
        if isinstance(obj, SoupState):
            (x_pos, y_pos) = obj.position
            if obj._cooking_tick > 0 and grid[y_pos][x_pos] == POT:
                print("found cooking object", obj)
                return True
    return False


Recipe.configure({})


def display_and_export_to_array(test_dict):
    test_dict = copy.deepcopy(test_dict)
    test_dict["kwargs"]["state"] = OvercookedState.from_dict(
        test_dict["kwargs"]["state"]
    )
    surface = StateVisualizer(**test_dict["config"]).render_state(
        **test_dict["kwargs"]
    )
    img_path = generate_temporary_file_path(
        "temporary_visualization", extension=".png"
    )
    pygame.image.save(surface, img_path)
    print("check if image is okay")
    show_image_in_ipython(img_path)
    return pygame.surfarray.array3d(surface)

def save_map_pic(test_dict, filename):
    test_dict = copy.deepcopy(test_dict)
    print(filename)
    test_dict["kwargs"]["state"] = OvercookedState.from_dict(
        test_dict["kwargs"]["state"]
    )
    surface = StateVisualizer(**test_dict["config"]).render_state(
        **test_dict["kwargs"]
    )
    img_path = "./diverse_population/visualisation/maps" + filename + ".png"
    pygame.image.save(surface, img_path)

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

def print_map(map_name):
    config = copy.deepcopy(DEFAULT_VALUES)
    mdp = OvercookedGridworld.from_layout_name(layout_name=map_name)
    agent_eval = AgentEvaluator(env_params={"horizon": 1}, mdp_fn=lambda _: mdp)
    trajectory_random_pair = agent_eval.evaluate_random_pair(num_games=1, display=False, native_eval=True)
    grid = trajectory_random_pair["mdp_params"][0]["terrain"]
    print("this is grid")
    print(grid)
    state = trajectory_random_pair["ep_states"][0][0]
    print("this is state")
    print(state)
    hud_data = {
        "all_orders": trajectory_random_pair["mdp_params"][0]["start_all_orders"]
    }
    kwargs = {"hud_data": hud_data, "grid": grid, "state": state.to_dict()}
    test_hud_2 = {
        "config": config,
        "kwargs": kwargs,
        "comment": "Test simple recipes in hud. NOTE: failing to render stuff outside HUD also fails this test",
        "result_array_filename": "test_hud_2.npy",
    }

    test_hud_2_array = save_map_pic(test_hud_2, map_name)

if __name__ == "__main__":
    directory = './data/layouts'
    layouts = []
    for filename in os.scandir(directory):
        if filename.is_file() and "multi" not in filename.name:
            print(filename.name)
            map_name = filename.name.split(".")[0]
            print(map_name)
            print_map(map_name)


