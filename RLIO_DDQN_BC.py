import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import rlio_rollout_stoage
import rlio_data_converter
from torch.autograd import Variable
import pointnet1
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RLIO_DDQN_BC(object):
    def __init__(self,
                 action_dim,
                 discount=0.99,
                 update_target_freq=2,
                 alpha=2.5,
                 learning_rate=3e-4,
                 mini_batch_size=64,
                 action_discrete_ranges=None
                 ):
        self.data_converter = None
        self.rollout_storage = None

        self.mini_batch_size = mini_batch_size
        self.action_number = action_dim
        self.discount = discount

        self.update_target_frequency = update_target_freq
        self.alpha = alpha

        # Discrete action ranges for each action dimension (with possible values for each dimension)
        self.action_discrete_ranges = action_discrete_ranges
        # self.action_discrete_ranges = {
        #     0: torch.tensor([0.2, 0.4, 0.5, 0.6, 0.8, 1.0], device=device),
        #     1: torch.tensor([2, 3, 4, 5], device=device),
        #     2: torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0], device=device),
        #     3: torch.tensor([0.005, 0.001, 0.05, 0.01, 0.1], device=device)
        # }

        # Initialize Q-network and target network (PointNet-based)
        self.Q_network = pointnet1.PointNet_RLIO(normal_channel=False, action_discrete_ranges=self.action_discrete_ranges).to(device)
        self.target_network = pointnet1.PointNet_RLIO(normal_channel=False, action_discrete_ranges=self.action_discrete_ranges).to(device)
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=learning_rate)

    def init_storage_and_converter(self, mini_batch_size, num_epochs, num_ids, num_trajs, num_steps, num_points_per_scan, error_sigma=1.0):
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
            error_sigma=error_sigma
        )

        self.data_converter.load_all_pcds()
        self.data_converter.set_pcd_name_array_for_each_ids()
    
    def update_target_network(self):
        # Copy the current Q-network to the target network
        self.target_network.load_state_dict(self.Q_network.state_dict())

    def update_Q_network(self, state, action, reward, state_new, done):
        not_done = ~done.squeeze()
        reward = reward.squeeze()

        # Set networks to evaluation mode
        self.Q_network.eval()
        self.target_network.eval()

        # Calculate argmax_a' Q_current(s', a') using the current Q-network
        with torch.no_grad():

            flatten_Q_values, _ = self.Q_network(state_new)      # flatten_Q_values: [batch_size, action_class(different for each)*action_dim]
            action_news = []
            soft_log_max_actions =[]   # for the classification (BC)

            for i in range(self.action_number):
                # TODO: action_new: from the current Q-network
                
                soft_max_Q_value = F.softmax(flatten_Q_values[:, i*self.action_discrete_ranges[i].shape[0]:(i+1)*self.action_discrete_ranges[i].shape[0]], dim=1)
                
                # soft_log_max_Q_value: for the classification (BC)
                soft_log_max_Q_value = F.log_softmax(flatten_Q_values[:, i*self.action_discrete_ranges[i].shape[0]:(i+1)*self.action_discrete_ranges[i].shape[0]], dim=1)
                soft_log_max_actions.append(soft_log_max_Q_value)

                max_idx = soft_max_Q_value.argmax(dim=1)
                action_new_onehot = torch.zeros(self.mini_batch_size, self.action_discrete_ranges[i].shape[0], device=device)
                action_new_onehot.scatter_(1, max_idx.view(-1, 1), 1.0)
                action_new_onehot = action_new_onehot.detach()
                action_news.append(action_new_onehot)

        # Compute the target value: y = r + discount_factor * Q_target(s', a') * (1 - done)
        # with torch.no_grad():
            flatten_target_Q_values, _ = self.target_network(state_new)
            target_Q_sum = torch.zeros(self.mini_batch_size, device=device)
            for i in range(self.action_number):
                target_Q = (flatten_target_Q_values[:, i*self.action_discrete_ranges[i].shape[0]:(i+1)*self.action_discrete_ranges[i].shape[0]] \
                            * action_news[i]).sum(dim=1)  #[batch_size]
                
                target_Q_sum += target_Q
                
            mean_target_Q = target_Q_sum / self.action_number
            y = reward + (self.discount * mean_target_Q * not_done)

        self.Q_network.train()

        soft_log_max_actions =[]
        # Resample the action for the classification (BC)
        flatten_Q_values_for_BC, _ = self.Q_network(state_new) 
        for i in range(self.action_number):
            # soft_log_max_Q_value: for the classification (BC)
            soft_log_max_Q_value = F.log_softmax(flatten_Q_values_for_BC[:, i*self.action_discrete_ranges[i].shape[0]:(i+1)*self.action_discrete_ranges[i].shape[0]], dim=1)
            soft_log_max_actions.append(soft_log_max_Q_value)


        # Train the Q-network: Q(s, a) -> y
        flatten_Q_values, _ = self.Q_network(state)
        lmbda = self.alpha/flatten_Q_values.abs().mean().detach()

        Q_sum = torch.zeros(self.mini_batch_size, device=device)
        BC_loss_sum = 0

        for i in range(self.action_number):
            action_range_i = self.action_discrete_ranges[i]
            closest_indices = torch.argmin(torch.abs(action_range_i.unsqueeze(0) - action[:, i].unsqueeze(1)), dim=1)
            one_hot_action = torch.zeros(self.mini_batch_size, self.action_discrete_ranges[i].shape[0], device=device)
            one_hot_action[torch.arange(self.mini_batch_size), closest_indices] = 1.0

            BC_loss = F.nll_loss(soft_log_max_actions[i], closest_indices)

            BC_loss_sum += BC_loss
            
            Q = (flatten_Q_values[:, i*self.action_discrete_ranges[i].shape[0]:(i+1)*self.action_discrete_ranges[i].shape[0]] \
                * one_hot_action).sum(dim=1)
            Q_sum += Q

        mean_Q = Q_sum / self.action_number
        
        DDQN_loss = F.mse_loss(input=mean_Q, target=y.detach())
        BC_loss = BC_loss_sum / self.action_number

        loss =  lmbda *DDQN_loss + BC_loss
        # loss =  DDQN_loss + BC_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        return loss.item()
    

    def train(self, it):

        generator= self.rollout_storage.mini_batch_generator()

        for points, next_points, reward, actions, dones in generator:
            if it % self.update_target_frequency == 0:
                self.update_target_network()

            loss = self.update_Q_network(points, actions, reward, next_points, dones)
        
        return loss
    
    def validation(self):

        mean_reward = 0
        num_reward_added = 0
        
        sampled_traj_name_pairs = self.data_converter.random_select_trajectory()
        sample_traj_num = min(10, len(sampled_traj_name_pairs))
        sampled_traj_name_pairs = random.sample(sampled_traj_name_pairs, sample_traj_num)

        self.Q_network.eval()

        for exp_dir, sub_dir in sampled_traj_name_pairs:
            # Get state
            valid_steps = self.data_converter.get_valid_idices(exp_dir, sub_dir) 
            last_valid_step = valid_steps[-1]
            sample_step_num = min(10, len(valid_steps))
            selected_steps = random.sample(list(valid_steps), sample_step_num)

            points_in_this_traj, _, _ = self.data_converter.load_points_from_all_pcd(exp_dir, selected_steps, valid_steps, last_valid_step) 
            points_in_this_traj_np = points_in_this_traj.cpu().numpy()
            processed_points = self.data_converter.pointnet_preprocess(points_in_this_traj_np)

            processed_points_tensor = processed_points.to(device).clone().detach()
            state = processed_points_tensor
            ########

            flatten_Q_values, _ = self.Q_network(state)  # [batch_size, action_class(different for each)*action_dim]
            selected_actions = []  # [batch_size, action_dim]

            for i in range(self.action_number):
                soft_max_Q_value = F.softmax(flatten_Q_values[:, i*self.action_discrete_ranges[i].shape[0]:(i+1)*self.action_discrete_ranges[i].shape[0]], dim=1)
                max_idx = soft_max_Q_value.argmax(dim=1)
                action_new_onehot = torch.zeros(sample_step_num, self.action_discrete_ranges[i].shape[0], device=device)
                action_new_onehot.scatter_(1, max_idx.view(-1, 1), 1.0)
                action_new_onehot = action_new_onehot.detach()

                selected_action_i = self.action_discrete_ranges[i][max_idx]
                selected_actions.append(selected_action_i)

            selected_actions_tensor = torch.stack(selected_actions, dim=1)

            for i in range(sample_step_num):
                # new_sub_dir = f"{selected_actions_tensor[0]:.1f}_{selected_actions_tensor[1]:.0f}_{selected_actions_tensor[2]:.1f}_{selected_actions_tensor[3]:g}"
                new_sub_dir = f"{selected_actions_tensor[i, 0].item():.1f}_{selected_actions_tensor[i, 1].item():.0f}_{selected_actions_tensor[i, 2].item():.1f}_{selected_actions_tensor[i, 3].item():g}"
                new_valid_steps = self.data_converter.get_valid_idices(exp_dir, new_sub_dir)
                new_last_valid_step = new_valid_steps[-1]
                step = selected_steps[i]
                reward = self.data_converter._get_reward_for_valid(exp_dir, new_sub_dir, step, new_valid_steps, new_last_valid_step)

                mean_reward += reward
                num_reward_added += 1
            # print(selected_actions_tensor)

        mean_reward /= num_reward_added

        self.Q_network.train()

        return mean_reward

