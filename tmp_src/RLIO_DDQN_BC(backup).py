import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import rlio_rollout_stoage
import rlio_data_converter
from torch.autograd import Variable
import pointnet1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RLIO_DDQN_BC(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 discount=0.99,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2,
                 alpha=2.5,
                 learning_rate=3e-4,
                 mini_batch_size=64):
        self.data_converter = None
        self.rollout_storage = None

        self.mini_batch_size = mini_batch_size
        self.action_number = action_dim
        self.discount = discount

        self.update_target_frequency = 10

        # Discrete action ranges for each action dimension (with possible values for each dimension)
        # self.action_discrete_ranges = {
        #     0: torch.tensor([0.2, 0.4, 0.5, 0.6, 0.8, 1.0], device=device),
        #     1: torch.tensor([2, 3, 4, 5], device=device),
        #     2: torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0], device=device),
        #     3: torch.tensor([0.005, 0.001, 0.05, 0.01, 0.1], device=device)
        # }
        self.action_discrete_ranges = {
            0: torch.tensor([0.2, 0.4, 0.5, 0.6, 0.8, 1.0], device=device),
            1: torch.tensor([2, 3, 3, 4, 4, 5], device=device),
            2: torch.tensor([0.2, 0.2, 0.4, 0.6, 0.8, 1.0], device=device),
            3: torch.tensor([0.005, 0.005, 0.001, 0.05, 0.01, 0.1], device=device)
        }
        # Initialize Q-network and target network (PointNet-based)
        self.Q_network = pointnet1.PointNet_RLIO(normal_channel=False, action_discrete_ranges=self.action_discrete_ranges).to(device)
        self.target_network = pointnet1.PointNet_RLIO(normal_channel=False, action_discrete_ranges=self.action_discrete_ranges).to(device)
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=learning_rate)

    def init_storage_and_converter(self, mini_batch_size, num_epochs, num_ids, num_trajs, num_steps, num_points_per_scan, reward_scale=1.0):
        self.rollout_storage = rlio_rollout_stoage.RLIORolloutStorage(
            mini_batch_size=mini_batch_size,
            num_epochs=num_epochs,
            num_ids=num_ids,
            num_trajs=num_trajs,
            num_steps=num_steps,
            num_points_per_scan=num_points_per_scan
        )
        self.data_converter = rlio_data_converter.RLIODataConverter(
            rollout_storage=self.rollout_storage,
            mini_batch_size=mini_batch_size,
            num_epochs=num_epochs,
            num_ids=num_ids,
            num_trajs=num_trajs,
            num_steps=num_steps,
            num_points_per_scan=num_points_per_scan,
            reward_scale=reward_scale
        )

        self.data_converter.load_all_pcds()
        self.data_converter.set_pcd_name_array_for_each_ids()
    
    def update_target_network(self):
        # Copy the current Q-network to the target network
        self.target_network.load_state_dict(self.Q_network.state_dict())

    def update_Q_network(self, state, action, reward, state_new, done):
        # Convert done to a float tensor
        done = done.float().unsqueeze(1)

        with torch.no_grad():
            # Get action log probabilities from Q-network
            action_log_probs, _ = self.Q_network(state_new)
            action_probs = torch.exp(action_log_probs)

            action_indices = torch.argmax(action_probs, dim=-1).unsqueeze(-1)

            # Convert action indices to one-hot encoding
            action_new_onehot = torch.zeros(self.mini_batch_size, self.action_number, len(self.action_discrete_ranges[0]), device=device)
            action_new_onehot.scatter_(2, action_indices, 1.0)

            # Compute target Q-value: y = reward + discount * Q_target(s', a')
            target_Q, _ = self.target_network(state_new)

            y = reward + (self.discount * (torch.sum(target_Q * action_new_onehot, dim=1)) * done)

        action_onehot = torch.zeros(self.mini_batch_size, self.action_number, len(self.action_discrete_ranges[0]), device=state.device)
        for dim in range(self.action_number):
            action_value = action[:, dim]
            action_range = self.action_discrete_ranges[dim]

            action_indices = torch.nonzero(action_range == action_value[:, None], as_tuple=False)[:, 1]
            print(action_indices) # TODO: when duplicate values are present, the first index is returned?
            action_onehot[:, dim, action_indices] = 1.0
            # action_onehot[torch.arange(action_value.size(0)), dim, action_indices] = 1.0

        print(action_onehot)
        print(self.Q_network(state))

        # Compute the Q-value for the given state-action pair
        Q = (self.Q_network(state) * action_onehot).sum(dim=1)

        # Compute the loss (Mean Squared Error)
        loss = F.mse_loss(Q, y)

        # Perform backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Return the loss value as a Python scalar
        return loss.item()
    

    def train(self, it):

        generator= self.rollout_storage.mini_batch_generator()

        for points, next_points, reward, actions, dones in generator:
            if it % self.update_target_frequency == 0:
                self.update_target_network()

            loss = self.update_Q_network(points, actions, reward, next_points, dones)

        
        return loss