#!/bin/bash
module add conda-modules-py37
conda activate "$home_dir"/envs/overcooked_ai_terminal

cd $home_dir
export CODEDIR=$(pwd)
echo "codedir: " $CODEDIR
export PROJDIR="$home_dir"/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py
echo "projdir: " $PROJDIR

cd $PROJDIR
echo $trained_models
echo $exp
echo $eval_set_name
echo $init_SP_agents
echo $delay_shared_rewards
echo $mode

echo $kl_diff_loss_coef
echo $kl_diff_loss_clip
echo $kl_diff_bonus_reward_coef
echo $kl_diff_bonus_reward_clip

echo $rnd_obj_prob_thresh_env
echo $seed
echo $n_sample_partners
echo $layout_name
echo $frame_stacking_mode
echo $frame_stacking

python diverse_population/diverse_pool_build.py --layout_name=$layout_name --trained_models=$trained_models --mode=$mode --exp=$exp --eval_set_name=$eval_set_name --init_SP_agents=$init_SP_agents --kl_diff_loss_coef=$kl_diff_loss_coef --kl_diff_loss_clip=$kl_diff_loss_clip --kl_diff_bonus_reward_coef=$kl_diff_bonus_reward_coef --kl_diff_bonus_reward_clip=$kl_diff_bonus_reward_clip --seed=$seed --n_sample_partners=$n_sample_partners --frame_stacking=$frame_stacking --frame_stacking_mode=$frame_stacking_mode --vf_coef=$vf_coef > "$SCRATCHDIR"/out.txt 2> "$SCRATCHDIR"/err.txt
echo "python dobehl"
INFODIR="$home_dir"/coding/results
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
