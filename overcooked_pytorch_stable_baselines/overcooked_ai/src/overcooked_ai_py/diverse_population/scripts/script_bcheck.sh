#!/bin/bash
module add conda-modules-py37
conda activate "$home_dir"/envs/overcooked_ai_terminal

cd $home_dir
export CODEDIR=$(pwd)/coding
echo "codedir: " $CODEDIR
export PROJDIR="$home_dir"/coding/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py
echo "projdir: " $PROJDIR
INFODIR="$home_dir"/coding/results
echo -e "$PBS_JOBNAME\t$PBS_JOBID\t`hostname -f`\t$SCRATCHDIR" >> "$INFODIR"/jobs_info.txt
cd $PROJDIR
pwd
echo $home_dir
echo "trained_models="$trained_models
echo "exp"=$exp
echo "eval_set_name"=$eval_set_name
echo "init_SP_agents"=$init_SP_agents
echo "delay_shared_rewards"=$delay_shared_rewards
echo "mode"=$mode
echo "execute_final_eval" = $execute_final_eval
echo "kl_diff_loss_coef"=$kl_diff_loss_coef
echo "kl_diff_loss_clip"=$kl_diff_loss_clip
echo "kl_diff_bonus_reward_coef"=$kl_diff_bonus_reward_coef
echo "kl_diff_bonus_reward_clip"=$kl_diff_bonus_reward_clip
echo "num_workers"=$num_workers
echo "rnd_obj_prob_thresh_env"=$rnd_obj_prob_thresh_env
echo "seed"=$seed
echo "n_sample_partners"=$n_sample_partners
echo "layout_name"=$layout_name
echo "frame_stacking_mode"=$frame_stacking_mode
echo "frame_stacking"=$frame_stacking
echo "execuet final eval"=$execute_final_eval
echo "checkp_step"=$checkp_steps
if [[ $execute_final_eval == "True" ]]; then
	execute="--execute_final_eval"
else
	execute=""
fi
echo $execute

python diverse_population/diverse_pool_build.py --log_dir=$PBS_JOBNAME --layout_name=$layout_name --trained_models=$trained_models --mode=$mode --exp=$exp --eval_set_name=$eval_set_name --init_SP_agents=$init_SP_agents --kl_diff_loss_coef=$kl_diff_loss_coef --kl_diff_loss_clip=$kl_diff_loss_clip $execute --kl_diff_bonus_reward_coef=$kl_diff_bonus_reward_coef --kl_diff_bonus_reward_clip=$kl_diff_bonus_reward_clip --seed=$seed --n_sample_partners=$n_sample_partners --checkp_step=$checkp_steps --frame_stacking=$frame_stacking --frame_stacking_mode=$frame_stacking_mode --num_workers=$num_workers --n_epochs=1 --vf_coef=$vf_coef --behavior_check > "$SCRATCHDIR"/out.txt 2> "$SCRATCHDIR"/err.txt
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

