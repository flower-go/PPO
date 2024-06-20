#!/bin/bash

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done


module add conda-modules-py37
conda activate "$home_dir"/envs/overcooked_ai_terminal

cd $home_dir
echo "homedir:" $home_dir
export CODEDIR=$(pwd)/coding
echo "codedir: " $CODEDIR
export PROJDIR="$home_dir"/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py
echo "projdir: " $PROJDIR
INFODIR="$home_dir"/coding/results
export JOBID=$PBS_JOBID
echo -e "$PBS_JOBNAME\t$PBS_JOBID\t`hostname -f`\t$SCRATCHDIR" >> "$INFODIR"/jobs_info.txt
cd $PROJDIR
pwd

echo "parametry:"
echo $params


params=$(echo $params | sed 's/^/--/g' | sed 's/ / --/g')
python diverse_population/visualisation/evaluation/eval_visualisation.py $params > "$SCRATCHDIR"/out.txt 2> "$SCRATCHDIR"/err.txt

echo "python dobehl"
date_name=$(date +%m%d-%H%M)
echo "job id"
echo "$PBS_JOBID"
ls "$SCRATCHDIR"
cp "$SCRATCHDIR"/out.txt "$INFODIR"/"$date_name"."$PBS_JOBID"_out.txt
cp "$SCRATCHDIR"/err.txt "$INFODIR"/"$date_name"."$PBS_JOBID"_err.txt

echo "skopirovano"
echo "file se jmenuje:"
echo "$INFODIR"/"$date_name"."$PBS_JOBID"_err.txt
rm -rf "$SCRATCHDIR"/*
