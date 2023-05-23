#!/bin/bash
#TTOTO funguje
#create env
# je to adekvatni ke skriptu scripts/script.sh ale se spravnymi cestami, presouvam do scripts
module add conda-modules-py37
conda activate /storage/plzen1/home/ayshi/envs/overcooked_ai_terminal 

cd $home_dir
pwd
#mkdir premek2
#cd premek2
#git clone https://github.com/PremekBasta/PPO.git
#rm /storage/plzen1/home/ayshi/coding/premek2/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py/diverse_population/models/forced_coordination/POP_SMALL8/00.zip 
export CODEDIR=$(pwd)
echo "codedir/home is: ""$CODEDIR"
export PROJDIR=/storage/plzen1/home/ayshi/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py
echo $PROJDIR
cd $PROJDIR 
echo "jsem tu" 
pwd
python diverse_population/diverse_pool_build.py --layout_name=$layout_name --trained_models=$trained_models --mode=$mode --exp=$exp --eval_set_name=$eval_set_name --init_SP_agents=$init_SP_agents --kl_diff_loss_coef=$kl_diff_loss_coef --kl_diff_loss_clip=$kl_diff_loss_clip --kl_diff_bonus_reward_coef=$kl_diff_bonus_reward_coef --kl_diff_bonus_reward_clip=$kl_diff_bonus_reward_clip --seed=$seed --n_sample_partners=$n_sample_partners --frame_stacking=$frame_stacking --frame_stacking_mode=$frame_stacking_mode --vf_coef=$vf_coef > "$SCRATCHDIR"/out.txt 2> "$SCRATCHDIR"/err.txt

echo "python dobehl"
#pouze aktivuji svoje rpostredi
# a pridavam kopirovnai
INFODIR=/storage/plzen1/home/ayshi/coding/results
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

