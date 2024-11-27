import numpy as np
import torch
import open3d as o3d
import os

import provider

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, num_trajs):
		max_scans_per_traj = 1000
		num_points = 10
		# self.max_size = max_size
		# self.ptr = 0
		# self.size = 0

		# self.state = np.zeros((max_size, state_dim))

		# self.action = np.zeros((max_size, action_dim))
		# self.next_state = np.zeros((max_size, state_dim))
		# self.reward = np.zeros((max_size, 1))
		# self.not_done = np.zeros((max_size, 1))

		# [#Traj, #scans/traj, #points, 3]
		self.point_input = torch.zeros((num_trajs, max_scans_per_traj, num_points, 3))


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def convert_D4RL(self, dataset):
		self.state = dataset['observations']
		self.action = dataset['actions']
		self.next_state = dataset['next_observations']
		self.reward = dataset['rewards'].reshape(-1,1)
		self.not_done = 1. - dataset['terminals'].reshape(-1,1)
		self.size = self.state.shape[0]


	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std
	
	def add_new_trajs_to_buffer(self, num_trajs):
		# TODO: Bufffer (sequential) -> need to be random sampled (shuffle)
		# traj_path ="/home/lim/HILTI22/exp14_basement_2/RLIO_1122test/Hesai/pvlio"

		# traj_path = "/home/lim/HILTI22/RLIO1122test_all_pvlio/Hesai/ours/0.2_2_0.2_0.1/frames"
		traj_path = "/home/lim/HILTI22/RLIO1122test_all_pvlio/Hesai/ours/"


		for i in range(num_trajs):
			param_name = "0.2_2_0.2_0.1" # Need to be replace with random sampling"

			state_idx = 0

			for ii in range(1000):
				print(ii)
				pcd_name = "F_" + str(ii) +"_ours_" + param_name + ".pcd" 
				pcd_path = traj_path +param_name + "/frames/" + pcd_name

				if os.path.exists(pcd_path):
					pcd = o3d.io.read_point_cloud(pcd_path)
					# (Nx3 array: [x, y, z])
					points = np.asarray(pcd.points)
					# print(points)

					# provider.random_point_dropout(points)  # Make it as regular shepe [Batch, N, 3]

					# points = points.data.numpy()
					# points = provider.random_point_dropout(points)
					# points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
					# points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
					# points = torch.Tensor(points)
					# points = points.transpose(2, 1)
					
					# points_tensor = torch.tensor(points, dtype=torch.float32)
				else:
					# print(pcd_path)




					state_idx += 1


class TrajectoryConverter(object):
	"""
	pcd -> [B(=T), N, 3] (+pointnet proprecessing)
	

	"""
	def __init__(self, num_traj) -> None:
		self.num_traj = num_traj
				
					
	def trajectory_converter(self):

		
		pass