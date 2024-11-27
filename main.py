import numpy as np
import torch
import argparse
import os

import RLIO_TD3_BC
import RLIO_DDQN_BC

import wandb
import time

WANDB = False
WANDB_SWEEP = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
	"""
	RLIO all-LIO : RL LIO, the all things need to tune the LIO
	"""

	if WANDB:
		wandb.init(project="rlio")

		if WANDB_SWEEP:
			# run_name = f"lr_{wandb.config.learning_rate}_alpha_{wandb.config.alpha}_noise_{wandb.config.policy_noise}"
			# run_name = f"lr_{wandb.config.learning_rate:.5f}"
			run_name = f"lr_{wandb.config.learning_rate:.5f}"
		else:
			run_name = "default_run"

		wandb.run.name = run_name
		wandb.run.save()

	save_model = False
	save_interval = 200
	timestamp = time.strftime("%Y%m%d_%H%M%S")
	save_path = f"/home/lim/rlio_ws/src/rlio/log/{timestamp}"
	os.makedirs(save_path, exist_ok=True)

	action_dim = 4 # num params to tune
	action_discrete_ranges = {
		0: torch.tensor([0.2, 0.4, 0.5, 0.6, 0.8, 1.0], device=device),
		1: torch.tensor([2, 3, 4, 5], device=device),
		2: torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0], device=device),
		3: torch.tensor([0.005, 0.001, 0.05, 0.01, 0.1], device=device)
	}

	# Training related
	max_timesteps = 1000
	num_epochs = 2 # 4

	# Batch size related
	num_points_per_scan = 1024 # 1024 # 256 # 1024 # 512  # 1024
	# (2, 16, 128)
	num_ids = 2 				# exp01, exp02, ...
	num_trajs = 8				# = num actions
	num_steps = 32		 		# for each traj  -> full batch size = num_ids * num_trajs * num_steps
	mini_batch_size = 256 		# 256
	
	# DDQN
	learning_rate = 3e-4 # 3e-4
	update_target_freq = 2
	discount = 0.99

	# BC
	alpha = 2.5

	use_val = True
	val_freq = 5 # 5
	error_sigma = 0.1

	if WANDB_SWEEP:
		learning_rate = wandb.config.learning_rate
		alpha = wandb.config.alpha
		discount = wandb.config.discount
		update_target_freq = wandb.config.update_target_freq
		error_sigma = wandb.config.error_sigma

	kwargs = {
		"action_dim": action_dim,
		"discount": discount,
		"update_target_freq": update_target_freq,
		"alpha": alpha,

		"learning_rate":learning_rate,
		"mini_batch_size":mini_batch_size,

		"action_discrete_ranges": action_discrete_ranges,
	}

	# Initialize policy
	policy = RLIO_DDQN_BC.RLIO_DDQN_BC(**kwargs)

	# policy.init_storage_and_converter(max_batch_size, mini_batch_size, num_epochs, num_trajs, num_points_per_scan)
	policy.init_storage_and_converter(
										mini_batch_size=mini_batch_size,
										num_epochs=num_epochs,
										num_ids=num_ids,
										num_trajs=num_trajs,
										num_steps=num_steps,
										num_points_per_scan=num_points_per_scan,
										error_sigma=error_sigma)	

	mean_reward_val = 0.0

	for it in range(int(max_timesteps)):

		start = time.time()

		policy.rollout_storage.reset_batches()
		policy.data_converter.preprocess_trajectory()

		preprocess_stop = time.time()

		policy.train(it=it)
		# mean_reward, mean_actor_loss, mean_critic_loss, mean_target_Q, mean_Q_error = policy.train(add_critic_pointnet)

		stop = time.time()
		preprocess_time = preprocess_stop - start
		train_time = stop - preprocess_stop

		total_time = stop - start
		print(f"it : {it} total_time : {total_time}s, pre_time : {preprocess_time}s, train_time : {train_time}s")

		if use_val and it % val_freq == 0:
			mean_reward_val = policy.validation()

			print("mean_rew_val : ", mean_reward_val)

		if WANDB:
			log_wandb(locals())

		# if WANDB_SWEEP:
		# 	score =  mean_reward_val - mean_reward

		# 	log_wandb(locals(), score)

		if save_model and it % save_interval == 0:
			path = save_path + f"/{it}"
			os.makedirs(path, exist_ok=True)

			save(policy.actor, path + "/actor.pt")
			save(policy.pointnet, path + "/pointnet.pt")

		start = stop

def save(model, filename):
	torch.save(model.state_dict(), filename)


def log_wandb(locs, score=None):
	# Note : mean_reward = just from the training set (set character)
	#        mean_reward_val = from the policy
	#        thus, the difference btw the mean_reward and val_reward = score ( = val - data, larger is better)
	# score =  mean_reward_val - mean_reward

	wandb_dict = {}
	wandb_dict['Loss/mean_reward'] = locs.get("mean_reward", 0.0)
	wandb_dict['Loss/mean_reward_val'] = locs.get("mean_reward_val", 0.0)

	wandb_dict['Loss/mean_actor_loss'] = locs.get("mean_actor_loss", 0.0)
	wandb_dict['Loss/mean_critic_loss'] = locs.get("mean_critic_loss", 0.0)	
	wandb_dict['Loss/mean_target_Q'] = locs.get("mean_target_Q", 0.0)
	wandb_dict['Loss/mean_Q_error'] = locs.get("mean_Q_error", 0.0)

	wandb_dict['Performance/preprocess_time'] = locs.get("preprocess_time", 0.0)
	wandb_dict['Performance/train_time'] = locs.get("train_time", 0.0)

	if score is not None:
		wandb_dict['score'] = score

	wandb.log(wandb_dict, step=locs['it'])
	

if __name__ == "__main__":

	if WANDB:
		wandb.login()

	if WANDB_SWEEP:
		
		sweep_configuration = {
			"method": "random",	 # "random"  # "bayes"
			"metric": {"goal": "maximize", "name": "score"},
			"parameters": {
				"learning_rate": {"max": 1e-3, "min": 1e-5},
				"alpha": {"max": 5.0, "min": 1.0},
				"discount": {"values": [0.99, 0.98, 0.97]},
				"update_target_freq": {"values": [2, 3, 5]},
				# "batch_cfg": {
				# 	"values": [
				# 		{"param1": 4, "param2": 4, "param3": 8},
				# 		{"param1": 1, "param2": 16, "param3": 8},
				# 		]
				# 	},
				"error_sigma" : {"max":1.0, "min":0.1},
			}
		}

		sweep_id = wandb.sweep(sweep=sweep_configuration, project="rlio_sweep")

		wandb.agent(sweep_id, function=main, count=1000)

	else:
		main()
