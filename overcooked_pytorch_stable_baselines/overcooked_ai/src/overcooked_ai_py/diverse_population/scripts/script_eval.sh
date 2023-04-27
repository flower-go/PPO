#!/bin/bash
module add conda-modules-py37
conda activate /storage/plzen1/home/premek_basta/.conda/envs/overcooked_ai_terminal


cd /storage/plzen1/home/premek_basta/
export CODEDIR=$(pwd)
echo $CODEDIR
export PROJDIR=/storage/plzen1/home/premek_basta/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py
echo $PROJDIR

cd PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py
echo $trained_models
echo $exp
echo $eval_set_name
echo $base_eval_name
echo $init_SP_agents
echo $delay_shared_rewards
echo $mode
echo $eval_mode
echo $eval_agents

echo $kl_diff_loss_coef
echo $kl_diff_loss_clip
echo $kl_diff_bonus_reward_coef
echo $kl_diff_bonus_reward_clip

echo $rnd_obj_prob_thresh_env
echo $seed
echo $n_sample_partners
echo $final_eval_games_per_worker
echo $layout_name

echo $frame_stacking
echo $frame_stacking_mode
echo $vf_coef

python diverse_population/diverse_pool_build.py --layout_name=$layout_name --trained_models=$trained_models --mode=$mode --eval_mode=$eval_mode --exp=$exp --base_eval_name=$base_eval_name --eval_set_name=$eval_set_name --eval_agents=$eval_agents --init_SP_agents=$init_SP_agents --kl_diff_loss_coef=$kl_diff_loss_coef --kl_diff_loss_clip=$kl_diff_loss_clip --kl_diff_bonus_reward_coef=$kl_diff_bonus_reward_coef --kl_diff_bonus_reward_clip=$kl_diff_bonus_reward_clip --seed=$seed --n_sample_partners=$n_sample_partners --execute_final_eval --final_eval_games_per_worker=$final_eval_games_per_worker --rnd_obj_prob_thresh_env=$rnd_obj_prob_thresh_env --frame_stacking=$frame_stacking --frame_stacking_mode=$frame_stacking_mode --vf_coef=$vf_coef > "$SCRATCHDIR"/out.txt 2> "$SCRATCHDIR"/err.txt

cp "$SCRATCHDIR"/out.txt /storage/plzen1/home/premek_basta/results/_out.txt

cp "$SCRATCHDIR"/err.txt /storage/plzen1/home/premek_basta/results/_err.txt

