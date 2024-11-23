import numpy as np
import torch
import argparse
import os

import rl_utils
import RLIO_TD3_BC

import wandb



# import rollout_storage as rs

def main():
	"""
	RLIO all-LIO : RL LIO, the all things need to tune the LIO
	"""
	state_dim = 64 # Dim Pointnet output
	action_dim = 4 # num params to tune

	# Action related
	max_action = 1.0

	# Training related
	max_timesteps = 10000
	num_trajs = 8 #4
	num_epochs = 4
	mini_batch_size = 64 # 512 # 256
	# mini_batch_size = 16 # 128 # 256


	# TD3
	expl_noise = 0.1         
	discount = 0.99
	tau = 0.005               
	policy_noise = 0.2
	noise_clip = 0.5        
	policy_freq = 2 
	# TD3 + BC
	alpha = 2.5
	# normalize = True

	# Batch size related
	# num_points_per_scan=1024
	num_points_per_scan=512  # 1024
	max_batch_size = 6000 #3000 for 4 trajs

	learning_rate = 1e-3 # 3e-4


	kwargs = {
		"state_dim": state_dim,  # Will be dims of PointNet output
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": discount,
		"tau": tau,
		# TD3
		"policy_noise": policy_noise * max_action,
		"noise_clip": noise_clip * max_action,
		"policy_freq": policy_freq,
		# TD3 + BC
		"alpha": alpha,

		"learning_rate":learning_rate
	}

	# Initialize policy
	policy = RLIO_TD3_BC.RLIO_TD3_BC(**kwargs)
	policy.init_storage_and_converter(max_batch_size, mini_batch_size, num_epochs, num_trajs, num_points_per_scan)

	for it in range(int(max_timesteps)):
		print(f"Step: {it}")

		policy.rollout_storage.reset_batches()
		policy.data_converter.preprocess_trajectory()

		mean_reward, mean_actor_loss, mean_critic_loss, mean_target_Q, mean_Q_error = policy.train()
		if WANDB:
			log_wandb(locals())


def log_wandb(locs):
	wandb_dict = {}
	wandb_dict['Loss/mean_reward'] = locs["mean_reward"]
	wandb_dict['Loss/mean_actor_loss'] = locs["mean_actor_loss"]
	wandb_dict['Loss/mean_critic_loss'] = locs["mean_critic_loss"]
	wandb_dict['Loss/mean_target_Q'] = locs["mean_target_Q"]
	wandb_dict['Loss/mean_Q_error'] = locs["mean_Q_error"]



	wandb.log(wandb_dict, step=locs['it'])

WANDB = True

if __name__ == "__main__":
	if WANDB:
		wandb.init(project="rlio")

	main()
