#!/bin/bash
home_dir="/storage/plzen1/home/ayshi"
module add conda-modules-py37
conda activate "$home_dir"/envs/overcooked_ai_terminal
echo -e "$PBS_JOBNAME\t$PBS_JOBID\t`hostname -f`\t$SCRATCHDIR" >> "$INFODIR"/jobs_info.txt
cd $home_dir
export CODEDIR=$(pwd)/coding
echo "codedir: " $CODEDIR
export PROJDIR="$home_dir"/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py
echo "projdir: " $PROJDIR
cd $PROJDIR
python diverse_population/visualisation/maps/evaluation/eval_visualisation.py