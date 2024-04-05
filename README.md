All code connected to population traiing is in the [directory diverse population](https://github.com/flower-go/PPO/tree/3f91fb5309cd3f9113c1a0b4bdfe99d80e97e1dc/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population)

# Cooperation with Unknown Agents in Multi-agent Overcooked Environment

This git repository (https://github.com/PremekBasta/PPO) serves the purpose of studying cooperation of artificial agents in multi agent cooking environment.
All of our experiments and findings are discussed in a detail in master thesis that is part of this project (Tex/MasterThesis).

The codebase for this project is inside repository overcooked_pytorch_stable_baselines, where two main components can be found.
First, stable-baselines3 is copy of Stable-Baselines3 RL library (https://github.com/DLR-RM/stable-baselines3) that we used and modified for the needs of our project. The most important modifications which are also described in chapter 6 a 7 of master thesis can be found in:
* stable_baselines3/common/torch_layers.py - All network structure modification
* stable_baselines3/common/on_policy_algorithm.py - Episodes experience collection including population partner sampling 
* stable_baselines3/ppo/ppo.py - Population KL divergence augmentation of objective function

And second, overcooked_ai is a copy of overcooked environment (https://github.com/HumanCompatibleAI/overcooked_ai) where all of our contribution lies inside diverse_population folder under src/overcooked_ai_py subfolder.


