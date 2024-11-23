import numpy as np
import torch
import argparse
import os

import rl_utils
import RLIO_TD3_BC

import wandb
import time


wandb.login()

# import rollout_storage as rs

def main():
	"""
	RLIO all-LIO : RL LIO, the all things need to tune the LIO
	"""

	if WANDB:
		wandb.init(project="rlio")

		if WANDB_SWEEP:
			# run_name = f"lr_{wandb.config.learning_rate}_alpha_{wandb.config.alpha}_noise_{wandb.config.policy_noise}"
			# run_name = f"lr_{wandb.config.learning_rate:.5f}"
			run_name = f"lr_{wandb.config.learning_rate:.5f}_state_dim_{wandb.config.state_dim}_tau_{wandb.config.tau}_alpha_{wandb.config.alpha}"
		else:
			run_name = "default_run"

		wandb.run.name = run_name
		wandb.run.save()

	save_model = True
	save_interval = 200
	timestamp = time.strftime("%Y%m%d_%H%M%S")
	save_path = f"/home/lim/rlio_ws/src/rlio/log/{timestamp}"
	os.makedirs(save_path, exist_ok=True)


	state_dim = 64 # Dim Pointnet output
	action_dim = 4 # num params to tune

	# Training related
	max_timesteps = 1000
	num_trajs = 4
	num_epochs = 4 # 4
	mini_batch_size = 256 # 64 # 512

	# TD3
	discount = 0.99
	tau = 0.005               
	policy_noise = 0.2
	noise_clip = 0.5        
	policy_freq = 2 
	# TD3 + BC
	alpha = 2.5

	# Batch size related
	num_points_per_scan= 512 # 512  # 1024
	max_batch_size = 6000 #3000 for 4 trajs

	learning_rate = 3e-4 # 3e-4

	if WANDB_SWEEP:
		learning_rate = wandb.config.learning_rate
		state_dim = wandb.config.state_dim
		tau = wandb.config.tau
		alpha = wandb.config.alpha	

	add_critic_pointnet = False


	kwargs = {
		"state_dim": state_dim,  # Will be dims of PointNet output
		"action_dim": action_dim,
		# "max_action": max_action,
		"discount": discount,
		"tau": tau,
		# TD3
		"policy_noise": policy_noise,
		"noise_clip": noise_clip,
		"policy_freq": policy_freq,
		# TD3 + BC
		"alpha": alpha,

		"learning_rate":learning_rate
	}

	# Initialize policy
	policy = RLIO_TD3_BC.RLIO_TD3_BC(**kwargs)
	policy.init_storage_and_converter(max_batch_size, mini_batch_size, num_epochs, num_trajs, num_points_per_scan)

	for it in range(int(max_timesteps)):
		# print()

		start = time.time()

		policy.rollout_storage.reset_batches()
		policy.data_converter.preprocess_trajectory()

		preprocess_stop = time.time()

		mean_reward, mean_actor_loss, mean_critic_loss, mean_target_Q, mean_Q_error = policy.train(add_critic_pointnet)

		if WANDB_SWEEP:
			score =  objective(mean_target_Q)
			log_wandb(locals(), score)

		stop = time.time()
		preprocess_time = preprocess_stop - start
		train_time = stop - preprocess_stop

		total_time = stop - start
		print(f"it : {it} total_time : {total_time}s, pre_time : {preprocess_time}s, train_time : {train_time}s")
		if WANDB:
			log_wandb(locals())

		if save_model and it % save_interval == 0:
			path = save_path + f"/{it}"
			os.makedirs(path, exist_ok=True)

			save(policy.actor, path + "/actor.pt")
			save(policy.pointnet, path + "/pointnet.pt")

		start = stop

def save(model, filename):
	torch.save(model.state_dict(), filename)


def log_wandb(locs, score=None):
	wandb_dict = {}
	wandb_dict['Loss/mean_reward'] = locs["mean_reward"]
	wandb_dict['Loss/mean_actor_loss'] = locs["mean_actor_loss"]
	wandb_dict['Loss/mean_critic_loss'] = locs["mean_critic_loss"]
	wandb_dict['Loss/mean_target_Q'] = locs["mean_target_Q"]
	wandb_dict['Loss/mean_Q_error'] = locs["mean_Q_error"]

	wandb_dict['Performance/preprocess_time'] = locs["preprocess_time"]
	wandb_dict['Performance/train_time'] = locs["train_time"]

	if score is not None:
		wandb_dict['score'] = score

	wandb.log(wandb_dict, step=locs['it'])

	

WANDB = True
WANDB_SWEEP = True


def objective(mean_target_Q):
    # score = config.x**3 + config.y
	# score = config.mean_target_Q
	return mean_target_Q


if __name__ == "__main__":

	if WANDB_SWEEP:
		
		sweep_configuration = {
			"method": "random",
			"metric": {"goal": "maximize", "name": "score"},
			"parameters": {
				"learning_rate": {"max:": 1e-3, "min": 1e-5},
				"state_dim": {"value": [32, 64, 128]},
				"tau": {"max":0.1, "min":0.0001},
				"alpha": {"max": 10, "min": 0.25},
			}
		}


		# state_dim = 64 # Dim Pointnet output
		# action_dim = 4 # num params to tune

		# # Training related
		# max_timesteps = 10#00
		# num_trajs = 4
		# num_epochs = 4 # 4
		# mini_batch_size = 256 # 64 # 512

		# # TD3
		# discount = 0.99
		# tau = 0.005               
		# policy_noise = 0.2
		# noise_clip = 0.5        
		# policy_freq = 2 
		# # TD3 + BC
		# alpha = 2.5

		# # Batch size related
		# num_points_per_scan= 5 #12 # 512  # 1024
		# max_batch_size = 6000 #3000 for 4 trajs

		# learning_rate = 3e-4 # 3e-4

		sweep_id = wandb.sweep(sweep=sweep_configuration, project="rlio_sweep")
		# score = objective(wandb.config)

		wandb.agent(sweep_id, function=main, count=10)

	else:
		main()
