#!/bin/bash
#module add conda-modules-py37
#conda activate /storage/brno2/home/premek_basta/overcooked_ai_terminal

#cd /storage/brno2/home/premek_basta/
#export CODEDIR=$(pwd)
#echo $CODEDIR
#export PROJDIR=/storage/brno2/home/premek_basta/PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py
#echo $PROJDIR

#cd PPO/overcooked_pytorch_stable_baselines/overcooked_ai/src/overcooked_ai_py

exp=$1;
layout_name=$2;
seed=$3;
base_eval_name=$4;
eval_set_name=$5;
eval_mode=$6;
eval_agents=$7;
final_eval_games_per_worker=$8;
rnd_obj_prob_thresh_env=$9;
frame_stacking_mode=${10};
frame_stacking=${11};


echo $exp
echo $layout_name
echo $seed
echo $base_eval_name
echo $eval_set_name
echo $eval_mode
echo $eval_agents
echo $final_eval_games_per_worker
echo $rnd_obj_prob_thresh_env
echo $frame_stacking_mode
echo $frame_stacking

vf_coef=0.1;



job_name="${exp}_${layout_name}_${frame_stacking_mode}_${frame_stacking}_eval_"

echo "${job_name}"
#No diversification
qsub -N "${job_name}0" -l  select=1:ncpus=4:ngpus=1:mem=16gb:scratch_local=4gb -q gpu -l walltime=20:00:00 -v layout_name=$layout_name,trained_models=11,exp=$exp,init_SP_agents=3,delay_shared_rewards=False,mode="POP",kl_diff_bonus_reward_coef=0.0,kl_diff_bonus_reward_clip=0.0,kl_diff_loss_coef=0.0,kl_diff_loss_clip=0.0,seed=$seed,n_sample_partners=3,frame_stacking=$frame_stacking,frame_stacking_mode=$frame_stacking_mode,base_eval_name=$base_eval_name,eval_set_name=$eval_set_name,eval_mode=$eval_mode,eval_agents=$eval_agents,final_eval_games_per_worker=$final_eval_games_per_worker,rnd_obj_prob_thresh_env=$rnd_obj_prob_thresh_env,vf_coef=$vf_coef  scripts/script_eval.sh

#Rewards diversification bonus
qsub -N "${job_name}R0" -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_local=4gb -q gpu -l walltime=20:00:00 -v layout_name=$layout_name,trained_models=11,exp=$exp,init_SP_agents=3,delay_shared_rewards=False,mode="POP",kl_diff_bonus_reward_coef=0.08,kl_diff_bonus_reward_clip=0.025,kl_diff_loss_coef=0.0,kl_diff_loss_clip=0.0,seed=$seed,n_sample_partners=3,frame_stacking=$frame_stacking,frame_stacking_mode=$frame_stacking_mode,base_eval_name=$base_eval_name,eval_set_name=$eval_set_name,eval_mode=$eval_mode,eval_agents=$eval_agents,final_eval_games_per_worker=$final_eval_games_per_worker,rnd_obj_prob_thresh_env=$rnd_obj_prob_thresh_env,vf_coef=$vf_coef  scripts/script_eval.sh
qsub -N "${job_name}R1" -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_local=4gb -q gpu -l walltime=20:00:00 -v layout_name=$layout_name,trained_models=11,exp=$exp,init_SP_agents=3,delay_shared_rewards=False,mode="POP",kl_diff_bonus_reward_coef=0.15,kl_diff_bonus_reward_clip=0.05,kl_diff_loss_coef=0.0,kl_diff_loss_clip=0.0,seed=$seed,n_sample_partners=3,frame_stacking=$frame_stacking,frame_stacking_mode=$frame_stacking_mode,base_eval_name=$base_eval_name,eval_set_name=$eval_set_name,eval_mode=$eval_mode,eval_agents=$eval_agents,final_eval_games_per_worker=$final_eval_games_per_worker,rnd_obj_prob_thresh_env=$rnd_obj_prob_thresh_env,vf_coef=$vf_coef  scripts/script_eval.sh
qsub -N "${job_name}R2" -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_local=4gb -q gpu -l walltime=20:00:00 -v layout_name=$layout_name,trained_models=11,exp=$exp,init_SP_agents=3,delay_shared_rewards=False,mode="POP",kl_diff_bonus_reward_coef=0.1,kl_diff_bonus_reward_clip=0.075,kl_diff_loss_coef=0.0,kl_diff_loss_clip=0.0,seed=$seed,n_sample_partners=3,frame_stacking=$frame_stacking,frame_stacking_mode=$frame_stacking_mode,base_eval_name=$base_eval_name,eval_set_name=$eval_set_name,eval_mode=$eval_mode,eval_agents=$eval_agents,final_eval_games_per_worker=$final_eval_games_per_worker,rnd_obj_prob_thresh_env=$rnd_obj_prob_thresh_env,vf_coef=$vf_coef  scripts/script_eval.sh

#Loss diversification bonus
qsub -N "${job_name}L0" -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_local=4gb -q gpu -l walltime=20:00:00 -v layout_name=$layout_name,trained_models=11,exp=$exp,init_SP_agents=3,delay_shared_rewards=False,mode="POP",kl_diff_bonus_reward_coef=0.0,kl_diff_bonus_reward_clip=0.0,kl_diff_loss_coef=0.12,kl_diff_loss_clip=0.07,seed=$seed,n_sample_partners=3,frame_stacking=$frame_stacking,frame_stacking_mode=$frame_stacking_mode,base_eval_name=$base_eval_name,eval_set_name=$eval_set_name,eval_mode=$eval_mode,eval_agents=$eval_agents,final_eval_games_per_worker=$final_eval_games_per_worker,rnd_obj_prob_thresh_env=$rnd_obj_prob_thresh_env,vf_coef=$vf_coef  scripts/script_eval.sh
qsub -N "${job_name}L1" -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_local=4gb -q gpu -l walltime=20:00:00 -v layout_name=$layout_name,trained_models=11,exp=$exp,init_SP_agents=3,delay_shared_rewards=False,mode="POP",kl_diff_bonus_reward_coef=0.0,kl_diff_bonus_reward_clip=0.0,kl_diff_loss_coef=0.08,kl_diff_loss_clip=0.03,seed=$seed,n_sample_partners=3,frame_stacking=$frame_stacking,frame_stacking_mode=$frame_stacking_mode,base_eval_name=$base_eval_name,eval_set_name=$eval_set_name,eval_mode=$eval_mode,eval_agents=$eval_agents,final_eval_games_per_worker=$final_eval_games_per_worker,rnd_obj_prob_thresh_env=$rnd_obj_prob_thresh_env,vf_coef=$vf_coef  scripts/script_eval.sh
qsub -N "${job_name}L2" -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_local=4gb -q gpu -l walltime=20:00:00 -v layout_name=$layout_name,trained_models=11,exp=$exp,init_SP_agents=3,delay_shared_rewards=False,mode="POP",kl_diff_bonus_reward_coef=0.0,kl_diff_bonus_reward_clip=0.0,kl_diff_loss_coef=0.1,kl_diff_loss_clip=0.15,seed=$seed,n_sample_partners=3,frame_stacking=$frame_stacking,frame_stacking_mode=$frame_stacking_mode,base_eval_name=$base_eval_name,eval_set_name=$eval_set_name,eval_mode=$eval_mode,eval_agents=$eval_agents,final_eval_games_per_worker=$final_eval_games_per_worker,rnd_obj_prob_thresh_env=$rnd_obj_prob_thresh_env,vf_coef=$vf_coef  scripts/script_eval.sh


#Both Rewards and diversification bonus
qsub -N "${job_name}R0L0" -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_local=4gb -q gpu -l walltime=20:00:00 -v layout_name=$layout_name,trained_models=11,exp=$exp,init_SP_agents=3,delay_shared_rewards=False,mode="POP",kl_diff_bonus_reward_coef=0.08,kl_diff_bonus_reward_clip=0.02,kl_diff_loss_coef=0.08,kl_diff_loss_clip=0.02,seed=$seed,n_sample_partners=3,frame_stacking=$frame_stacking,frame_stacking_mode=$frame_stacking_mode,base_eval_name=$base_eval_name,eval_set_name=$eval_set_name,eval_mode=$eval_mode,eval_agents=$eval_agents,final_eval_games_per_worker=$final_eval_games_per_worker,rnd_obj_prob_thresh_env=$rnd_obj_prob_thresh_env,vf_coef=$vf_coef  scripts/script_eval.sh
qsub -N "${job_name}R1L1" -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_local=4gb -q gpu -l walltime=20:00:00 -v layout_name=$layout_name,trained_models=11,exp=$exp,init_SP_agents=3,delay_shared_rewards=False,mode="POP",kl_diff_bonus_reward_coef=0.1,kl_diff_bonus_reward_clip=0.04,kl_diff_loss_coef=0.1,kl_diff_loss_clip=0.03,seed=$seed,n_sample_partners=3,frame_stacking=$frame_stacking,frame_stacking_mode=$frame_stacking_mode,base_eval_name=$base_eval_name,eval_set_name=$eval_set_name,eval_mode=$eval_mode,eval_agents=$eval_agents,final_eval_games_per_worker=$final_eval_games_per_worker,rnd_obj_prob_thresh_env=$rnd_obj_prob_thresh_env,vf_coef=$vf_coef  scripts/script_eval.sh

