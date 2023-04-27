#!/bin/bash


exp=$1;
layout_name=$2;
seed=$3;
frame_stacking_mode=$4
vf_coef=$5

echo $exp
echo $layout_name
echo $seed
echo $frame_stacking_mode

job_name="${exp}_${layout_name}_${frame_stacking_mode}_train"
echo $job_name

#No diversification
qsub -N "${job_name}0" -l select=1:ncpus=3:ngpus=1:mem=15gb:scratch_local=4gb -q gpu -l walltime=23:50:00 -v layout_name=$layout_name,trained_models=11,exp=$exp,init_SP_agents=3,delay_shared_rewards=False,mode="POP",kl_diff_bonus_reward_coef=0.0,kl_diff_bonus_reward_clip=0.0,kl_diff_loss_coef=0.0,kl_diff_loss_clip=0.0,seed=$seed,n_sample_partners=3,frame_stacking=4,frame_stacking_mode=$frame_stacking_mode,vf_coef=$vf_coef  scripts/script.sh

#Rewards diversification bonus
qsub -N "${job_name}R0" -l select=1:ncpus=3:ngpus=1:mem=15gb:scratch_local=4gb -q gpu -l walltime=23:50:00 -v layout_name=$layout_name,trained_models=11,exp=$exp,init_SP_agents=3,delay_shared_rewards=False,mode="POP",kl_diff_bonus_reward_coef=0.08,kl_diff_bonus_reward_clip=0.025,kl_diff_loss_coef=0.0,kl_diff_loss_clip=0.0,seed=$seed,n_sample_partners=3,frame_stacking=4,frame_stacking_mode=$frame_stacking_mode,vf_coef=$vf_coef  scripts/script.sh
qsub -N "${job_name}R1" -l select=1:ncpus=3:ngpus=1:mem=15gb:scratch_local=4gb -q gpu -l walltime=23:50:00 -v layout_name=$layout_name,trained_models=11,exp=$exp,init_SP_agents=3,delay_shared_rewards=False,mode="POP",kl_diff_bonus_reward_coef=0.15,kl_diff_bonus_reward_clip=0.05,kl_diff_loss_coef=0.0,kl_diff_loss_clip=0.0,seed=$seed,n_sample_partners=3,frame_stacking=4,frame_stacking_mode=$frame_stacking_mode,vf_coef=$vf_coef  scripts/script.sh
qsub -N "${job_name}R2" -l select=1:ncpus=3:ngpus=1:mem=15gb:scratch_local=4gb -q gpu -l walltime=23:50:00 -v layout_name=$layout_name,trained_models=11,exp=$exp,init_SP_agents=3,delay_shared_rewards=False,mode="POP",kl_diff_bonus_reward_coef=0.1,kl_diff_bonus_reward_clip=0.075,kl_diff_loss_coef=0.0,kl_diff_loss_clip=0.0,seed=$seed,n_sample_partners=3,frame_stacking=4,frame_stacking_mode=$frame_stacking_mode,vf_coef=$vf_coef  scripts/script.sh


#Loss diversification bonus
qsub -N "${job_name}L0" -l select=1:ncpus=3:ngpus=1:mem=15gb:scratch_local=4gb -q gpu -l walltime=23:50:00 -v layout_name=$layout_name,trained_models=11,exp=$exp,init_SP_agents=3,delay_shared_rewards=False,mode="POP",kl_diff_bonus_reward_coef=0.0,kl_diff_bonus_reward_clip=0.0,kl_diff_loss_coef=0.12,kl_diff_loss_clip=0.07,seed=$seed,n_sample_partners=3,frame_stacking=4,frame_stacking_mode=$frame_stacking_mode,vf_coef=$vf_coef  scripts/script.sh
qsub -N "${job_name}L1" -l select=1:ncpus=3:ngpus=1:mem=15gb:scratch_local=4gb -q gpu -l walltime=23:50:00 -v layout_name=$layout_name,trained_models=11,exp=$exp,init_SP_agents=3,delay_shared_rewards=False,mode="POP",kl_diff_bonus_reward_coef=0.0,kl_diff_bonus_reward_clip=0.0,kl_diff_loss_coef=0.08,kl_diff_loss_clip=0.03,seed=$seed,n_sample_partners=3,frame_stacking=4,frame_stacking_mode=$frame_stacking_mode,vf_coef=$vf_coef  scripts/script.sh
qsub -N "${job_name}L2" -l select=1:ncpus=3:ngpus=1:mem=15gb:scratch_local=4gb -q gpu -l walltime=23:50:00 -v layout_name=$layout_name,trained_models=11,exp=$exp,init_SP_agents=3,delay_shared_rewards=False,mode="POP",kl_diff_bonus_reward_coef=0.0,kl_diff_bonus_reward_clip=0.0,kl_diff_loss_coef=0.1,kl_diff_loss_clip=0.15,seed=$seed,n_sample_partners=3,frame_stacking=4,frame_stacking_mode=$frame_stacking_mode,vf_coef=$vf_coef  scripts/script.sh


#Both Rewards and diversification bonus
qsub -N "${job_name}R0L0" -l select=1:ncpus=3:ngpus=1:mem=15gb:scratch_local=4gb -q gpu -l walltime=23:50:00 -v layout_name=$layout_name,trained_models=11,exp=$exp,init_SP_agents=3,delay_shared_rewards=False,mode="POP",kl_diff_bonus_reward_coef=0.08,kl_diff_bonus_reward_clip=0.02,kl_diff_loss_coef=0.08,kl_diff_loss_clip=0.02,seed=$seed,n_sample_partners=3,frame_stacking=4,frame_stacking_mode=$frame_stacking_mode,vf_coef=$vf_coef  scripts/script.sh
qsub -N "${job_name}R1L1" -l select=1:ncpus=3:ngpus=1:mem=15gb:scratch_local=4gb -q gpu -l walltime=23:50:00 -v layout_name=$layout_name,trained_models=11,exp=$exp,init_SP_agents=3,delay_shared_rewards=False,mode="POP",kl_diff_bonus_reward_coef=0.1,kl_diff_bonus_reward_clip=0.04,kl_diff_loss_coef=0.1,kl_diff_loss_clip=0.03,seed=$seed,n_sample_partners=3,frame_stacking=4,frame_stacking_mode=$frame_stacking_mode,vf_coef=$vf_coef  scripts/script.sh

