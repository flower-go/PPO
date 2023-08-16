#!/bin/bash

#home dir must be set correctly
if [ -z "$home_dir" ]; then home_dir="/storage/plzen1/home/ayshi"; fi
echo "home set to: $home_dir"

job_name="vis_eval_test"
echo $job_name

res_dir="$home_dir"/coding/results
echo "results can be found here: $res_dir"
date_name=$(date +%m%d-%H%M)

qsub -N "$date_name"_"$exp" -e "$res_dir" -o "$res_dir" -l select=1:ncpus=4:ngpus=1:mem=17gb:scratch_local=4gb -q gpu -l walltime=1:50:00 "$home_dir"/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/scripts/vis_eval.sh

