import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rlio_rollout_stoage

import rlio_data_converter
import rlio_rollout_stoage

import pointnet1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
	"""
	action range:
		0: 0.25~1.0
		1: 3~5
		2: 0.25~1.0
		3: 0.005~0.1
	"""

	def __init__(self, state_dim, action_dim):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)

		# Define action range min and max for each action dimension
		self.action_min = torch.tensor([0.25, 3.0, 0.25, 0.005], device=device)
		self.action_max = torch.tensor([1.0, 5.0, 1.0, 0.1], device=device)
		
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		a = self.l3(a)
		action = 0.5 * (a + 1) * (self.action_max - self.action_min) + self.action_min
		return action


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class RLIO_TD3_BC(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		# max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		alpha=2.5,
		learning_rate=3e-4,
	):

		self.actor = Actor(state_dim, action_dim).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

		# self.pointnet = pointnet2.PointNet2Encoder(num_class=state_dim, normal_channel=True).to(device)
		self.pointnet = pointnet1.PointNet_RLIO(k=state_dim, normal_channel=False).to(device)
		self.critic_pointnet = pointnet1.PointNet_RLIO(k=state_dim, normal_channel=False).to(device)



		# self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.alpha = alpha

		self.total_it = 0

		self.rollout_storage = None
		self.data_converter = None


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()
	
	def init_storage_and_converter(self,max_batch_size, mini_batch_size, num_epochs, num_trajs, num_points_per_scan):

		self.rollout_storage = rlio_rollout_stoage.RLIORolloutStorage(max_batch_size=max_batch_size, num_points=num_points_per_scan,
																	mini_batch_size=mini_batch_size, num_epochs=num_epochs)
		self.data_converter = rlio_data_converter.RLIODataConverter(self.rollout_storage, num_trajs=num_trajs, num_points_per_scan=num_points_per_scan)

	def reset_batches(self):
		self.rollout_storage.reset_batches()

	def process_trajectory(self):
		self.data_converter.preprocess_trajectory()


	def train(self, add_critic_pointnet):
		self.total_it += 1

		mean_reward = 0
		mean_actor_loss = 0
		mean_critic_loss = 0

		mean_target_Q = 0
		mean_Q_error = 0

		num_update = 0

		generator= self.rollout_storage.mini_batch_generator()

		for points, next_points, reward, actions, dones in generator:

			state, _ = self.pointnet(points) # global feature & feature transform matrix(auxiliary)
			next_state, _ = self.pointnet(next_points) # global feature & feature transform matrix(auxiliary)

			not_done = ~dones

			with torch.no_grad():
				# Select action according to policy and add clipped noise
				noise = (
					torch.randn_like(actions) * self.policy_noise
				).clamp(-self.noise_clip, self.noise_clip)
				
				next_action = self.actor_target(next_state) + noise

				# Compute the target Q value
				target_Q1, target_Q2 = self.critic_target(next_state, next_action)
				target_Q = torch.min(target_Q1, target_Q2)
				target_Q = reward + not_done * self.discount * target_Q

				mean_target_Q += target_Q.mean().item()


			# Get current Q estimates

			if add_critic_pointnet:
				critic_state, _ = self.critic_pointnet(points.detach())
				current_Q1, current_Q2 = self.critic(critic_state, actions)
			else:
				current_Q1, current_Q2 = self.critic(state.detach(), actions)  # detach() state to avoid updating pointnet twice

			mean_Q_error += (current_Q1 - target_Q).abs().mean().item()
			mean_Q_error += (current_Q2 - target_Q).abs().mean().item()

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Delayed policy updates
			if self.total_it % self.policy_freq == 0:

				# Compute actor loss
				pi = self.actor(state)

				# print(pi[0])

				Q = self.critic.Q1(state, pi)
				lmbda = self.alpha/Q.abs().mean().detach()

				actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, actions)  # Maximize current Q value
				
				# Optimize the actor 
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()

				# Update the frozen target models
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			else:
				actor_loss = torch.tensor(0.0)

			mean_reward += reward.mean().item()
			mean_actor_loss += actor_loss.mean().item()
			mean_critic_loss += critic_loss.mean().item()
			num_update += 1


		mean_actor_loss /= num_update
		mean_critic_loss /= num_update
		mean_reward /= num_update

		mean_target_Q /= num_update
		mean_Q_error /= (2*num_update)

		return mean_reward, mean_actor_loss, mean_critic_loss, mean_target_Q, mean_Q_error



	def log(self, reward, actor_loss, critic_loss):
		print(reward, actor_loss, critic_loss)

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)