# Codebase for our diverse population building method

In this folder we try to concentrate all the important parts regarding our diverse population approach separately from the main environment engine.
The main and most important file is "diverse_pool_build.py", which serves as an entry point for all our purposes, including training, evaluation and visualization.
The results from this entrypoint are then stored in corresponding subfolders (models, evaluation, visualisation).

To run the entrypoint given the installation source location eg. "/home/user/PPO"
one must:
* activate conda environment created during installation
* set environment variables
	export CODEDIR="/home/user"
	export PROJDIR="/home/user/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py"
	
Run the entrypoint to see the options
	python diverse_pool_build --help
	
	
To see our common usage look at the scripts folder where are stored scripts used to run batch experiments on MetaCentrum. 

Final visualised figures were generated using python jupyter notebooks located in scripts directory.


