# Codebase for our diverse population building method

In this separated folder we try to concentrate all important parts regarding our diverse population approach separately from the main environment engine.
Main and most important file is "diverse_pool_build.py" which serves as an entrypoint for all our purposes, which includes training, evaluation and visualisation.
The results from this entrypoint are then stored to corresponding subfolders (models, evaluation, visualisation).

To run the entrypoint given the installation source location eg. "/home/user/PPO"
one must:
* activate conda environment created during installation
* set environment variables
	export CODEDIR="/home/user"
	export PROJDIR="/home/user/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py"
	
Then entrypoint can be ran as follows:
	python diverse_pool_build.py
Run following to see the options
	python diverse_pool_build --help
	
	
To see our common usage look at the scripts folder where are stored scripts used to run batch experiments on MetaCentrum. 


