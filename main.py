import numpy as np
import torch
import argparse
import os

import RLIO_DDQN_BC

import wandb
import time
import copy

WANDB = False
WANDB_SWEEP = False

EXPORT_POLICY = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def load(policy: RLIO_DDQN_BC.RLIO_DDQN_BC, path):
# 	loaded_dir = torch.load(path)
# 	policy.Q_network.load_state_dict(loaded_dir)


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

	save_model = True
	save_interval = 200
	timestamp = time.strftime("%Y%m%d_%H%M%S")
	save_path = f"/home/lim/rlio_ws/src/rlio/log/{timestamp}"
	os.makedirs(save_path, exist_ok=True)
	
	defalut_param = [0.5, 3, 0.6, 0.001]

	action_dim = 4 # num params to tune
	action_discrete_ranges = {
		0: torch.tensor([0.2, 0.4, 0.5, 0.6, 0.8, 1.0], device=device),
		1: torch.tensor([2, 3, 4, 5], device=device),
		2: torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0], device=device),
		3: torch.tensor([0.005, 0.001, 0.05, 0.01, 0.1], device=device)
	}

	# Training related
	max_timesteps = 5000 # 1000
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
	update_target_freq = 5
	discount = 0.99

	# BC
	alpha = 2.5

	use_val = True
	val_freq = 5 # 5
	error_sigma = 0.1
	error_scale = 10

	if WANDB_SWEEP:
		learning_rate = wandb.config.learning_rate
		alpha = wandb.config.alpha
		# discount = wandb.config.discount
		# update_target_freq = wandb.config.update_target_freqs
		error_sigma = wandb.config.error_sigma
		error_scale = wandb.config.error_scale

	kwargs = {
		"action_dim": action_dim,
		"discount": discount,
		"update_target_freq": update_target_freq,
		"alpha": alpha,

		"learning_rate":learning_rate,
		"mini_batch_size":mini_batch_size,

		"action_discrete_ranges": action_discrete_ranges,
		"default_param": defalut_param,

		"error_scale": error_scale,
		"error_sigma": error_sigma,
	}

	# Initialize policy
	policy = RLIO_DDQN_BC.RLIO_DDQN_BC(**kwargs)


	if EXPORT_POLICY:
		save_root_path = '/home/lim/rlio_ws/exported_models'
		load_root_path = '/home/lim/rlio_ws/src/rlio/log'

		dir_name = '20241129_103031'
		check_point = 4800

		# dir_name = '20241203_214100'
		# check_point = 0

		# Load
		load_dir = os.path.join(load_root_path, dir_name, str(check_point))

		policy.Q_network.eval()
		loaded_dict = torch.load(os.path.join(load_dir, 'Q_net.pt'))
		policy.Q_network.load_state_dict(loaded_dict)

		# Export
		save_dir = os.path.join(save_root_path, dir_name, str(check_point))
		os.makedirs(save_dir, exist_ok=True)
		save_path_pt = os.path.join(save_dir, 'Q_net.pt')
		model = copy.deepcopy(policy.Q_network).to('cpu')

		traced_script_module = torch.jit.script(model)
		traced_script_module.save(save_path_pt)

		policy.Q_network.train()

		print("Model exported to : ", save_path_pt)



	# policy.init_storage_and_converter(max_batch_size, mini_batch_size, num_epochs, num_trajs, num_points_per_scan)
	policy.init_storage_and_converter(
										mini_batch_size=mini_batch_size,
										num_epochs=num_epochs,
										num_ids=num_ids,
										num_trajs=num_trajs,
										num_steps=num_steps,
										num_points_per_scan=num_points_per_scan,
										# error_sigma=error_sigma,
										)	

	mean_reward_val = 0.0
	mean_default_reward_val = 0.0
	rew_gain = 0.0

	# mean_rew_gain = 0.0
	# mean_rew_gain_update_num = 0
	recent_rew_gains = []

	for it in range(int(max_timesteps)):

		start = time.time()

		policy.rollout_storage.reset_batches()
		policy.data_converter.preprocess_trajectory()

		preprocess_stop = time.time()

		# mean_reward, mean_default_reward_val, total_loss, DDQN_loss, BC_loss, mean_target_Q = policy.train(it=it)
		mean_reward, total_loss, DDQN_loss, BC_loss, mean_target_Q = policy.train(it=it)


		val_vs_train = mean_reward_val - mean_reward
		# rew_gain = mean_reward - mean_default_reward_val
		# mean_reward, mean_actor_loss, mean_critic_loss, mean_target_Q, mean_Q_error = policy.train(add_critic_pointnet)

		stop = time.time()
		preprocess_time = preprocess_stop - start
		train_time = stop - preprocess_stop

		total_time = stop - start
		print(f"it : {it} total_time : {total_time}s, pre_time : {preprocess_time}s, train_time : {train_time}s")

		if use_val and it % val_freq == 0:
			mean_reward_val, mean_default_reward_val = policy.validation()
			rew_gain = mean_reward_val - mean_default_reward_val

			recent_rew_gains.append(rew_gain)
			if len(recent_rew_gains) > 10:
				recent_rew_gains.pop(0)

			mean_rew_gain = sum(recent_rew_gains) / len(recent_rew_gains)
			# mean_rew_gain_update_num += 1

			print("mean_rew_val : ", mean_reward_val, " / mean_default_rew : ", mean_default_reward_val)

		if WANDB:
			log_wandb(locals())

		if WANDB_SWEEP:
			# score =  mean_reward_val - mean_reward
			# score = mean_reward_val - mean_default_reward_val
			score = mean_rew_gain
			log_wandb(locals(), score)

		if save_model and it % save_interval == 0:
			path = save_path + f"/{it}"
			os.makedirs(path, exist_ok=True)

			save(policy.Q_network, path + "/Q_net.pt")

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


	wandb_dict['Loss/total_loss'] = locs.get("total_loss", 0.0)
	wandb_dict['Loss/DDQN_loss'] = locs.get("DDQN_loss", 0.0)
	wandb_dict['Loss/BC_loss'] = locs.get("BC_loss", 0.0)
	wandb_dict["Loss/mean_target_Q"] = locs.get("mean_target_Q", 0.0)

	wandb_dict["Loss/val_vs_train"] = locs.get("val_vs_train", 0.0)
	wandb_dict["Loss/rew_gain"] = locs.get("rew_gain", 0.0)

	# wandb_dict['Loss/mean_actor_loss'] = locs.get("mean_actor_loss", 0.0)
	# wandb_dict['Loss/mean_critic_loss'] = locs.get("mean_critic_loss", 0.0)	
	# wandb_dict['Loss/mean_target_Q'] = locs.get("mean_target_Q", 0.0)
	# wandb_dict['Loss/mean_Q_error'] = locs.get("mean_Q_error", 0.0)

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
				# "learning_rate": {"max": 1e-3, "min": 1e-5},
				"learning_rate": {"max": 1e-3, "min": 3e-5},

				"alpha": {"max": 5.0, "min": 2.0},
				# "discount": {"values": [0.99, 0.98, 0.97]},
				# "update_target_freq": {"values": [5]},
				# "batch_cfg": {
				# 	"values": [
				# 		{"param1": 4, "param2": 4, "param3": 8},
				# 		{"param1": 1, "param2": 16, "param3": 8},
				# 		]
				# 	},
				"error_sigma" : {"max":1.0, "min":0.5},
				"error_scale" : {"max":100.0, "min":10.0},
			}
		}

		sweep_id = wandb.sweep(sweep=sweep_configuration, project="rlio_sweep")

		wandb.agent(sweep_id, function=main, count=1000)

	else:
		main()
