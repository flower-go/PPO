#!/bin/bash

#arguments parsers (dont pass equal sign as a value!!)
#source: https://unix.stackexchange.com/questions/129391/passing-named-arguments-to-shell-scripts
# potrebuju: exp, layout name, seed, frame_stacking_mode,vf_coef
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done


#home dir must be set correctly
if [ -z "$home_dir" ]; then home_dir="/storage/plzen1/home/ayshi"; fi
echo "home set to: $home_dir"
exp="maps"
res_dir="$home_dir"/coding/results
echo "results can be found here: $res_dir"
date_name=$(date +%m%d-%H%M)
echo "i will load file"
echo $input_file

echo "I will run $file"
qsub  -l select=1:ncpus=4:ngpus=1:mem=17gb:scratch_local=4gb -q gpu -l walltime=1:50:00 -v layout=$layout,input_file=$input_file,home_dir="$home_dir" "$home_dir"/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/scripts/print_maps.sh
