import torch
import numpy as np
import os
import random
import zipfile
from io import BytesIO
import open3d as o3d

import provider
import rlio_rollout_stoage

import time

class RLIODataConverter:
    def __init__(self, rollout_storage: rlio_rollout_stoage.RLIORolloutStorage, 
                mini_batch_size, num_epochs, num_ids, num_trajs, num_steps, num_points_per_scan):
        self.base_path = "/home/lim/HILTI22" 
        self.num_trajs = num_trajs
        self.num_points_per_scan = num_points_per_scan

        self.num_ids = num_ids
        self.num_trajs = num_trajs
        self.num_steps = num_steps

        self.rollout_storage = rollout_storage
        self.error_array_name = 'error_array.npy'
        self.time_array_name = 'time.npy'

    def set_pcd_name_array_for_each_ids(self):
        self.pcd_name_dict = {}
        exp_dirs = [d for d in os.listdir(self.base_path) if d.startswith('exp') and os.path.isdir(os.path.join(self.base_path, d))]

        for exp_dir in exp_dirs:
            pcd_path = os.path.join(self.base_path, exp_dir, 'RLIO_1122test', 'LiDARs', 'Hesai')
            pcd_names = [f for f in os.listdir(pcd_path) if f.endswith('.pcd')]

            self.pcd_name_dict[exp_dir] = pcd_names


    def random_select_trajectory(self):
        exp_dirs = [d for d in os.listdir(self.base_path) if d.startswith('exp') and os.path.isdir(os.path.join(self.base_path, d))]
        seleced_exp_dirs = random.sample(exp_dirs, self.num_ids)

        valid_pairs = []

        for exp_dir in seleced_exp_dirs:
            ours_path = os.path.join(self.base_path, exp_dir, 'RLIO_1122test', 'Hesai', 'ours')

        if os.path.exists(ours_path):
            sub_dirs = [d for d in os.listdir(ours_path) if os.path.isdir(os.path.join(ours_path, d))]
            
            selected_sub_dirs = []
            attempts = 0
            max_attempts = 10 
            while len(selected_sub_dirs) < self.num_trajs and attempts < max_attempts:
                remaining_sub_dirs = list(set(sub_dirs) - set(selected_sub_dirs))
                if not remaining_sub_dirs:
                    break
                to_sample = min(len(remaining_sub_dirs), self.num_trajs - len(selected_sub_dirs))
                sampled_dirs = random.sample(remaining_sub_dirs, to_sample)

                for sub_dir in sampled_dirs:
                    status_file = os.path.join(ours_path, sub_dir, 'status.txt')
                    if os.path.exists(status_file):
                        with open(status_file, 'r') as f:
                            content = f.read().strip()
                        if content == "Finished":
                            selected_sub_dirs.append(sub_dir)

                attempts += 1

            for sub_dir in selected_sub_dirs:
                valid_pairs.append((exp_dir, sub_dir))

        return valid_pairs
    
    def restore_path(self, exp_dir, sub_dir):
        # restored_path = os.path.join( self.base_path, exp_dir, 'RLIO1122test_all_pvlio', 'Hesai', 'ours', sub_dir)
        restored_path = os.path.join( self.base_path, exp_dir, 'RLIO_1122test', 'Hesai', 'ours', sub_dir)

        return restored_path

    def preprocess_errors(self, exp_dir, sub_dir):
        restored_path = self.restore_path(exp_dir, sub_dir)

        poses_txt = os.path.join(restored_path, 'poses.txt')
        poses = np.loadtxt(poses_txt)  # (time, x, y, z, ...)
        poses_times = poses[:, 0]

        errors_zip = os.path.join(restored_path, 'errors', 'rpe_trans.zip')

        if os.path.exists(errors_zip):
            with zipfile.ZipFile(errors_zip, 'r') as zip_ref:

                if self.error_array_name in zip_ref.namelist():
                    with zip_ref.open(self.error_array_name) as file:
                        error_array = np.load(BytesIO(file.read()))

                if self.time_array_name in zip_ref.namelist():
                    with zip_ref.open(self.time_array_name) as file:
                        time_array = np.load(BytesIO(file.read()))
                    if len(time_array) == len(error_array) + 1: # Note: it is yield.. need to check
                        time_array = time_array[:-1]
                    # np.set_printoptions(suppress=True, precision=20, formatter={'float': '{:.10f}'.format})

            selected_steps = np.array([np.argmin(np.abs(poses_times - time)) for time in time_array])
            errors = error_array

        else:
            print(f"No rpe_trans error zip file: {errors_zip}")

        # print(selected_steps)

        return selected_steps, errors
    
    def _sample_for_fixed_num_points(self, path_to_pcd):
        pcd = o3d.io.read_point_cloud(path_to_pcd)
        points = np.asarray(pcd.points)

        if points.shape[0] > self.num_points_per_scan:
            indices = np.random.choice(points.shape[0], self.num_points_per_scan, replace=False)
            sampled_points = points[indices]
        else:
            indices = np.random.choice(points.shape[0], self.num_points_per_scan, replace=True)
            sampled_points = points[indices]

        return sampled_points

    def fixed_num_sample_points(self, exp_dir, selected_steps, valid_steps, last_valid_step):
        """
        To make the number of points fixed, it randomly samples points from each frame.
        output: 
            points (#steps, 1024(#points/scan), 3)
            next_points (#steps, 1024(#points/scan), 3)
            done (#steps)
        """
        frames_path = os.path.join(self.base_path, exp_dir,'RLIO_1122test/LiDARs/Hesai')

        points_in_this_traj = []
        next_points_in_this_traj = []
        done = []

        # print(valid_steps.shape) 
        # alid_indices.shape: # of GT steps, valid_steps[t] -> incex of the real time in the poses.txt (= pcd file index)

        for current_step in selected_steps:

            if current_step != last_valid_step:

                idx_of_current_step_in_valid_steps = np.where(valid_steps == current_step)[0][0]
                next_step = valid_steps[idx_of_current_step_in_valid_steps + 1]

                path_to_pcd = os.path.join(frames_path, self.pcd_name_dict[exp_dir][current_step])
                path_to_next_pcd = os.path.join(frames_path, self.pcd_name_dict[exp_dir][next_step])

                if os.path.exists(path_to_pcd):
                    sampled_points = self._sample_for_fixed_num_points(path_to_pcd)
                    points_in_this_traj.append(sampled_points)
                else:
                    print(f"[ERROR] PCD File not found: {path_to_pcd}")

                if os.path.exists(path_to_pcd) and os.path.exists(path_to_next_pcd):
                    next_sampled_points = self._sample_for_fixed_num_points(path_to_next_pcd)
                    next_points_in_this_traj.append(next_sampled_points)
                else:
                    print(f"[ERROR] Next PCD File not found: {path_to_next_pcd}")

                done.append(False)
            else:  # Last state of each trajectory

                path_to_pcd = os.path.join(frames_path, self.pcd_name_dict[exp_dir][current_step])

                if os.path.exists(path_to_pcd):
                    sampled_points = self._sample_for_fixed_num_points(path_to_pcd)
                    points_in_this_traj.append(sampled_points)
                else:
                    print(f"[ERROR] PCD File not found: {path_to_pcd}")
                sampled_points = np.zeros((self.num_points_per_scan, 3))
                next_points_in_this_traj.append(sampled_points)
                done.append(True)

        return np.array(points_in_this_traj), np.array(next_points_in_this_traj), np.array(done)  # (#frames, 1024(#points), 3)
    
    def pointnet_preprocess(self, points):
        """
        Make the points compatible with PointNet input.

        input: (#frames, 1024(#points/scan), 3)
        output: (#frames, 3, 1024(#points/scan))

        """
        
        points = provider.random_point_dropout(points)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)  # (#frames, 1024(#points/scan), 3)
        points = points.transpose(2, 1) # (#frames, 3, 1024(#points/scan))

        return points
    
    def delete_invalid_errors(self, errors, valid_steps):
        return errors[valid_steps]
    
    def sub_dir_to_param(self, sub_dir):

        params_str = sub_dir.split('_')
        params = [float(p) for p in params_str]
        return np.array(params)

    # def calculate_reward(self, errors):
    #     coef = 50.0 #100.0

    #     # print((1-coef*errors).mean())
    #     return 1-coef*errors
    
    def select_steps(self, valid_steps, errors):
        """
        Select the steps to be used for training.
        """
        selections = random.sample(range(len(valid_steps)), self.num_steps)
        return valid_steps[selections], errors[selections]
    
    def get_valid_idices(self, exp_dir, sub_dir):
        restored_path = self.restore_path(exp_dir, sub_dir)

        poses_txt = os.path.join(restored_path, 'poses.txt')
        poses = np.loadtxt(poses_txt)  # (time, x, y, z, ...)
        poses_times = poses[:, 0]

        errors_zip = os.path.join(restored_path, 'errors', 'rpe_trans.zip')

        if os.path.exists(errors_zip):
            with zipfile.ZipFile(errors_zip, 'r') as zip_ref:

                if self.time_array_name in zip_ref.namelist():
                    with zip_ref.open(self.time_array_name) as file:
                        time_array = np.load(BytesIO(file.read()))
                        time_array = time_array[:-1]

            selected_steps = np.array([np.argmin(np.abs(poses_times - time)) for time in time_array])

        else:
            print(f"No rpe_trans error zip file: {errors_zip}")

        return selected_steps  # Usage: selected_steps[t] -> incex of the real time in the poses.txt (= pcd file index)

    def sample_steps(self, valid_steps):
        selections = random.sample(range(len(valid_steps)), self.num_steps)
        return valid_steps[selections]
    
    def calculate_reward(self, error):
        coef = 50.0
        return 1-coef*error

    def get_rewards(self, exp_dir, sub_dir, selected_steps, valid_steps, last_valid_step):
        # sub_dir: action parameters
        restored_path = self.restore_path(exp_dir, sub_dir)
        errors_zip = os.path.join(restored_path, 'errors', 'rpe_trans.zip')

        rewards = []

        for current_step in selected_steps:

            if current_step == last_valid_step:
                rewards.append(0)
            else:
                idx_of_current_step_in_valid_steps = np.where(valid_steps == current_step)[0][0]
                # idx_of_next_step_in_valid_steps = idx_of_current_step_in_valid_steps + 1

                if os.path.exists(errors_zip):
                    with zipfile.ZipFile(errors_zip, 'r') as zip_ref:
                        if self.error_array_name in zip_ref.namelist():
                            with zip_ref.open(self.error_array_name) as file:
                                error_array = np.load(BytesIO(file.read()))

                                # error = error_array[idx_of_next_step_in_valid_steps]
                                error = error_array[idx_of_current_step_in_valid_steps]

                                rewards.append(self.calculate_reward(error))

                        else: # maybe not called
                            rewards.append(-1)

                else: # diverge -> penalty largely
                    rewards.append(-1)
        
        return rewards
    

    def _get_reward(self, exp_dir, sub_dir, step, valid_steps, last_valid_step):
        """
        get a single reward for a single step (for the validation only!!!)
        """

        # sub_dir: action parameters
        restored_path = self.restore_path(exp_dir, sub_dir)
        errors_zip = os.path.join(restored_path, 'errors', 'rpe_trans.zip')

        # rewards = []

        current_step = step

        if current_step == last_valid_step:
            reward = 0
        else:
            idx_of_current_step_in_valid_steps = np.where(valid_steps == current_step)[0][0]
            idx_of_next_step_in_valid_steps = idx_of_current_step_in_valid_steps + 1

            if os.path.exists(errors_zip):
                with zipfile.ZipFile(errors_zip, 'r') as zip_ref:
                    if self.error_array_name in zip_ref.namelist():
                        with zip_ref.open(self.error_array_name) as file:
                            error_array = np.load(BytesIO(file.read()))

                            # error = error_array[idx_of_next_step_in_valid_steps]
                            error = error_array[idx_of_current_step_in_valid_steps]

                            reward = self.calculate_reward(error)

                    else: # maybe not called
                        reward = -1

            else: # diverge -> penalty largely
                reward = -1
        
        return reward
    
    def preprocess_trajectory(self):
        sampled_traj_name_pairs = self.random_select_trajectory()
        for exp_dir, sub_dir in sampled_traj_name_pairs:
            # Note: sub_dir itself is the action parameters

            # get a_t
            params = self.sub_dir_to_param(sub_dir) # action parameters

            valid_steps = self.get_valid_idices(exp_dir, sub_dir)  # Note: when we aquire pcd, it must pass through the self.pcd_name_dict
            last_valid_step = valid_steps[-1]
            selected_steps = self.sample_steps(valid_steps)

            # get s_t, s_t+1, done (Note: s_t -> s_t+1 is deterministic in this OFF RL formulation)
            points_in_this_traj, next_points_in_this_traj, done = self.fixed_num_sample_points(exp_dir, selected_steps, valid_steps, last_valid_step)
            processed_points = self.pointnet_preprocess(points_in_this_traj)
            processed_next_points = self.pointnet_preprocess(next_points_in_this_traj)

            # get r_t (no r_t+1) (the goal is minimize current error, and the current error is calculated by the current action)
            rewards = self.get_rewards(exp_dir, sub_dir, selected_steps, valid_steps, last_valid_step)

            self.rollout_storage.add_new_trajs_to_buffer(processed_points, processed_next_points, rewards, params, done)

        
