import torch

class RLIORolloutStorage:
    def __init__(self, max_batch_size=1000, num_points=1024, mini_batch_size=16, num_epochs=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mini_batch_size = mini_batch_size
        self.num_epochs = num_epochs

        self.max_batch_size = max_batch_size
        self.num_points = num_points

        self.current_batch_idx = 0

        self.points_batch = torch.zeros(self.max_batch_size, 3, self.num_points, dtype=torch.float32, device=self.device, requires_grad=False)
        self.next_points_batch = torch.zeros(self.max_batch_size, 3, self.num_points, dtype=torch.float32, device=self.device, requires_grad=False)
        self.rewards_batch = torch.zeros(self.max_batch_size, dtype=torch.float32, device=self.device, requires_grad=False)
        self.params_batch = torch.zeros(self.max_batch_size, 4, dtype=torch.float32, device=self.device, requires_grad=False)
        self.dones_batch = torch.zeros(self.max_batch_size, dtype=torch.bool, device=self.device, requires_grad=False)


    def reset_batches(self):
        self.current_batch_idx = 0

        self.points_batch = torch.zeros(self.max_batch_size, 3, self.num_points, device=self.device)
        self.next_points_batch = torch.zeros(self.max_batch_size, 3, self.num_points, device=self.device)

        self.rewards_batch = torch.zeros(self.max_batch_size, device=self.device)

    def add_new_trajs_to_buffer(self, points, next_points, rewards, params, dones):
        """
        points, next_points: (#frames, 3, 1024)
        rewards: (#frames, 6)
        """
        num_frames, _, num_points = points.shape # (#frames, 3, 1024)
        print(num_points, num_frames)
        assert num_points == self.num_points     # 1024

        if self.current_batch_idx + num_frames > self.max_batch_size:
            raise AssertionError("Rollout buffer overflow")

        self.points_batch[self.current_batch_idx:self.current_batch_idx+num_frames] = points
        self.next_points_batch[self.current_batch_idx:self.current_batch_idx+num_frames] = next_points
        self.rewards_batch[self.current_batch_idx:self.current_batch_idx+num_frames] = torch.tensor(rewards)
        self.params_batch[self.current_batch_idx:self.current_batch_idx+num_frames] = torch.tensor(params)
        self.dones_batch[self.current_batch_idx:self.current_batch_idx+num_frames] = torch.tensor(dones)

        self.current_batch_idx += num_frames

    def mini_batch_generator(self):
        """
        Generate mini-batches for training.
        """
        self.num_mini_batches = self.current_batch_idx // self.mini_batch_size

        print("====================================")
        print(self.current_batch_idx)
        print(self.num_mini_batches)
        print(self.mini_batch_size)

        valid_point_batch = self.points_batch[:self.current_batch_idx].detach()
        valid_next_point_batch = self.next_points_batch[:self.current_batch_idx].detach()

        valid_rewards_batch = self.rewards_batch[:self.current_batch_idx].detach()
        valid_params_batch = self.params_batch[:self.current_batch_idx].detach()
        valid_dones_batch = self.dones_batch[:self.current_batch_idx].detach()
        
        for _ in range(self.num_epochs):
            # To deal with the case where the number of samples is not a multiple of the mini-batch size
            indices = torch.randperm(self.current_batch_idx, requires_grad=False, device=self.device)

            for i in range(self.num_mini_batches):

                start = i*self.mini_batch_size
                end = (i+1)*self.mini_batch_size
                batch_idx = indices[start:end]

                state_batch = valid_point_batch[batch_idx]
                next_state_batch = valid_next_point_batch[batch_idx]
                reward_batch = valid_rewards_batch[batch_idx].unsqueeze(1)
                actions_batch = valid_params_batch[batch_idx]
                dones_batch = valid_dones_batch[batch_idx].unsqueeze(1)

                yield state_batch, next_state_batch, reward_batch, actions_batch, dones_batch