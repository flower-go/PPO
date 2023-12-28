#!/bin/bash
start='\"'
exp=$1
name=$start$exp
code_dir="/storage/plzen1/home/ayshi/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/scripts/qsubs"
echo "hledam:"
echo $name
command=$(cat $code_dir/* | grep $name)
echo $command
eval $command

