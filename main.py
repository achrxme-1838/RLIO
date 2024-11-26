import numpy as np
import torch
import argparse
import os

import rl_utils
import RLIO_TD3_BC

import wandb
import time

WANDB = False
WANDB_SWEEP = False


def main():
	"""
	RLIO all-LIO : RL LIO, the all things need to tune the LIO
	"""

	if WANDB:
		wandb.init(project="rlio")

		if WANDB_SWEEP:
			# run_name = f"lr_{wandb.config.learning_rate}_alpha_{wandb.config.alpha}_noise_{wandb.config.policy_noise}"
			# run_name = f"lr_{wandb.config.learning_rate:.5f}"
			run_name = f"lr_{wandb.config.learning_rate:.5f}_state_dim_{wandb.config.state_dim}"
		else:
			run_name = "default_run"

		wandb.run.name = run_name
		wandb.run.save()

	save_model = False
	save_interval = 200
	timestamp = time.strftime("%Y%m%d_%H%M%S")
	save_path = f"/home/lim/rlio_ws/src/rlio/log/{timestamp}"
	os.makedirs(save_path, exist_ok=True)


	state_dim = 64 # Dim Pointnet output
	action_dim = 4 # num params to tune

	# Training related
	max_timesteps = 1000
	num_epochs = 1 # 4
	mini_batch_size = 256 # 1024 # 256 # 1024 #64 #256 # 64 # 512

	# TD3
	discount = 0.99
	tau = 0.005               
	policy_noise = 0.2
	noise_clip = 0.5        
	policy_freq = 2 
	# TD3 + BC
	alpha = 2.5

	# Batch size related
	num_points_per_scan = 1024 # 256 # 1024 # 512  # 1024

	# (2, 16, 128)
	num_ids = 2 		# exp01, exp02, ...
	num_trajs = 16 #16 		# = num actions
	num_steps = 128 # 128 #128		# 64 		# for each traj  -> full batch size = num_ids * num_trajs * num_steps

	learning_rate = 3e-4 # 3e-4

	use_val = True
	val_freq = 10 # 5
	reward_scale = 1.0

	if WANDB_SWEEP:
		learning_rate = wandb.config.learning_rate
		state_dim = wandb.config.state_dim
		# tau = wandb.config.tau
		# alpha = wandb.config.alpha
		# discount = wandb.config.discount

		# policy_noise = wandb.config.policy_noise
		# policy_freq	= wandb.config.policy_freq

		# num_ids = wandb.config.batch_cfg["param1"]		
		# num_trajs =	wandb.config.batch_cfg["param2"]			
		# num_steps = wandb.config.batch_cfg["param3"]

		reward_scale = wandb.config.reward_scale

	add_critic_pointnet = False


	kwargs = {
		"state_dim": state_dim,  # Will be dims of PointNet output
		"action_dim": action_dim,
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
	# policy.init_storage_and_converter(max_batch_size, mini_batch_size, num_epochs, num_trajs, num_points_per_scan)
	policy.init_storage_and_converter(
										mini_batch_size=mini_batch_size,
										num_epochs=num_epochs,
										num_ids=num_ids,
										num_trajs=num_trajs,
										num_steps=num_steps,
										num_points_per_scan=num_points_per_scan,
										reward_scale=reward_scale)	

	mean_reward_val = 0.0

	for it in range(int(max_timesteps)):

		start = time.time()

		policy.rollout_storage.reset_batches()
		policy.data_converter.preprocess_trajectory()

		preprocess_stop = time.time()

		mean_reward, mean_actor_loss, mean_critic_loss, mean_target_Q, mean_Q_error = policy.train(add_critic_pointnet)

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

		if WANDB_SWEEP:
			score =  (mean_reward_val - mean_reward)/reward_scale

			log_wandb(locals(), score)

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
	

def objective(mean_target_Q):
    # score = config.x**3 + config.y
	# score = config.mean_target_Q
	return mean_target_Q


if __name__ == "__main__":

	if WANDB:
		wandb.login()

	if WANDB_SWEEP:
		
		sweep_configuration = {
			"method": "random",	 # "random"  # "bayes"
			"metric": {"goal": "maximize", "name": "score"},
			"parameters": {
				"learning_rate": {"max": 1e-3, "min": 1e-5},
				"state_dim": {"values": [32, 64, 128]},
				# "tau": {"max":0.1, "min":0.0001},
				# "alpha": {"max": 10.0, "min": 0.25},
				# "discount": {"values": [0.99, 0.975, 0.95]},

				# "policy_noise": {"values": [0.1, 0.2, 0.3]},
				# "policy_freq": {"values": [1, 2, 4]},

				# "batch_cfg": {
				# 	"values": [
				# 		{"param1": 4, "param2": 4, "param3": 8},
				# 		{"param1": 1, "param2": 16, "param3": 8},
				# 		]
				# 	},
				"reward_scale" : {"max":100.0, "min":1.0},
			}
		}

		sweep_id = wandb.sweep(sweep=sweep_configuration, project="rlio_sweep")

		wandb.agent(sweep_id, function=main, count=1000)

	else:
		main()
