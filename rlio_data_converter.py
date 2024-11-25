import torch
import numpy as np
import os
import random
import zipfile
from io import BytesIO
import open3d as o3d
from multiprocessing import Pool

import provider
import rlio_rollout_stoage

import time

def sample_points_from_pcd(path_to_pcd, num_points_per_scan):
    if path_to_pcd is None:
        # For the last state, where there is no next point cloud
        return np.zeros((num_points_per_scan, 3))

    if os.path.exists(path_to_pcd):
        pcd = o3d.io.read_point_cloud(path_to_pcd)
        points = np.asarray(pcd.points)

        if points.shape[0] > num_points_per_scan:
            indices = np.random.choice(points.shape[0], num_points_per_scan, replace=False)
        else:
            indices = np.random.choice(points.shape[0], num_points_per_scan, replace=True)
        sampled_points = points[indices]

        return sampled_points
    else:
        print(f"[ERROR] PCD File not found: {path_to_pcd}")
        return np.zeros((num_points_per_scan, 3))  # Return zeros if file not found

class RLIODataConverter:
    def __init__(self, rollout_storage: rlio_rollout_stoage.RLIORolloutStorage, 
                mini_batch_size, num_epochs, num_ids, num_trajs, num_steps, num_points_per_scan, reward_scale):
        self.base_path = "/home/lim/HILTI22" 
        self.num_trajs = num_trajs
        self.num_points_per_scan = num_points_per_scan

        self.num_ids = num_ids
        self.num_trajs = num_trajs
        self.num_steps = num_steps

        self.rollout_storage = rollout_storage
        self.error_array_name = 'error_array.npy'
        self.time_array_name = 'time.npy'

        self.reward_scale = reward_scale

    def set_pcd_name_array_for_each_ids(self):
        self.pcd_name_dict = {}
        exp_dirs = [d for d in os.listdir(self.base_path) if d.startswith('exp') and os.path.isdir(os.path.join(self.base_path, d))]

        for exp_dir in exp_dirs:
            pcd_path = os.path.join(self.base_path, exp_dir, 'RLIO_1122test', 'LiDARs', 'Hesai')
            pcd_names = [f for f in os.listdir(pcd_path) if f.endswith('.pcd')]
            self.pcd_name_dict[exp_dir] = sorted(pcd_names)












            

    def random_select_trajectory(self):
        exp_dirs = [d for d in os.listdir(self.base_path) if d.startswith('exp') and os.path.isdir(os.path.join(self.base_path, d))]
        selected_exp_dirs = random.sample(exp_dirs, self.num_ids)

        valid_pairs = []

        for exp_dir in selected_exp_dirs:
            ours_path = os.path.join(self.base_path, exp_dir, 'RLIO_1122test', 'Hesai', 'ours')

            if os.path.exists(ours_path):
                sub_dirs = [d for d in os.listdir(ours_path) if os.path.isdir(os.path.join(ours_path, d))]
                sub_dirs_sampled = []

                while len(sub_dirs_sampled) < self.num_trajs:
                    # Remaining sub_dirs to sample from
                    remaining_sub_dirs = list(set(sub_dirs) - set(sub_dirs_sampled))
                    
                    if not remaining_sub_dirs:
                        print("[ERROR] the number of sub_dirs is less than the number of trajs")
                        break

                    sub_dir = random.choice(remaining_sub_dirs)
                    poses_path = os.path.join(ours_path, sub_dir, 'poses.txt')

                    if os.path.exists(poses_path):
                        # Check if poses.txt has enough lines
                        with open(poses_path, 'r') as f:
                            line_count = sum(1 for _ in f)
                        
                        if line_count >= self.num_steps*2:
                            valid_pairs.append((exp_dir, sub_dir))
                            sub_dirs_sampled.append(sub_dir)

        return valid_pairs

    def restore_path(self, exp_dir, sub_dir):
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
                    if len(time_array) == len(error_array) + 1:
                        time_array = time_array[:-1]

            selected_steps = np.array([np.argmin(np.abs(poses_times - time)) for time in time_array])
            errors = error_array

        else:
            print(f"No rpe_trans error zip file: {errors_zip}")

        return selected_steps, errors

    def fixed_num_sample_points(self, exp_dir, selected_steps, valid_steps, last_valid_step):
        """
        To make the number of points fixed, it randomly samples points from each frame.
        output: 
            points (#steps, num_points_per_scan, 3)
            next_points (#steps, num_points_per_scan, 3)
            done (#steps)
        """
        frames_path = os.path.join(self.base_path, exp_dir,'RLIO_1122test', 'LiDARs', 'Hesai')

        path_to_pcd_list = []
        path_to_next_pcd_list = []
        done_list = []

        for current_step in selected_steps:
            if current_step != last_valid_step:
                idx_of_current_step_in_valid_steps = np.where(valid_steps == current_step)[0][0]
                next_step = valid_steps[idx_of_current_step_in_valid_steps + 1]

                path_to_pcd = os.path.join(frames_path, self.pcd_name_dict[exp_dir][current_step])
                path_to_next_pcd = os.path.join(frames_path, self.pcd_name_dict[exp_dir][next_step])

                path_to_pcd_list.append(path_to_pcd)
                path_to_next_pcd_list.append(path_to_next_pcd)
                done_list.append(False)
            else:  # Last state of each trajectory
                path_to_pcd = os.path.join(frames_path, self.pcd_name_dict[exp_dir][current_step])

                path_to_pcd_list.append(path_to_pcd)
                path_to_next_pcd_list.append(None)  # No next point cloud
                done_list.append(True)

        num_points_per_scan = self.num_points_per_scan

        # Create argument tuples for starmap
        args_pcd = [(path, num_points_per_scan) for path in path_to_pcd_list]
        args_next_pcd = [(path, num_points_per_scan) for path in path_to_next_pcd_list]

        # Now, use Pool to process
        with Pool() as pool:
            points_in_this_traj = pool.starmap(sample_points_from_pcd, args_pcd)
            next_points_in_this_traj = pool.starmap(sample_points_from_pcd, args_next_pcd)

        points_in_this_traj = np.array(points_in_this_traj)
        next_points_in_this_traj = np.array(next_points_in_this_traj)
        done = np.array(done_list)

        return points_in_this_traj, next_points_in_this_traj, done

    def pointnet_preprocess(self, points):
        """
        Make the points compatible with PointNet input.
        input: (#frames, num_points_per_scan, 3)
        output: (#frames, 3, num_points_per_scan)
        """
        points = provider.random_point_dropout(points)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)  # (#frames, num_points_per_scan, 3)
        points = points.transpose(2, 1)  # (#frames, 3, num_points_per_scan)

        return points

    def delete_invalid_errors(self, errors, valid_steps):
        return errors[valid_steps]
    
    def sub_dir_to_param(self, sub_dir):
        params_str = sub_dir.split('_')
        params = [float(p) for p in params_str]
        return np.array(params)

    def select_steps(self, valid_steps):
        selections = random.sample(range(len(valid_steps)), self.num_steps)
        return valid_steps[selections]
    
    def get_valid_indices(self, exp_dir, sub_dir):
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
                        time_array = time_array[:-1]  # Delete last element

            selected_steps = np.array([np.argmin(np.abs(poses_times - time)) for time in time_array])
        else:
            print(f"No rpe_trans error zip file: {errors_zip}")

        return selected_steps

    def sample_steps(self, valid_steps):
        selections = random.sample(range(len(valid_steps)), self.num_steps)
        return valid_steps[selections]
    
    def calculate_reward(self, error):
        coef = 1.0
        return (1 - coef * error) * self.reward_scale

    def get_rewards(self, exp_dir, sub_dir, selected_steps, valid_steps, last_valid_step):
        restored_path = self.restore_path(exp_dir, sub_dir)
        trans_errors_zip = os.path.join(restored_path, 'errors', 'rpe_trans.zip')
        rot_errors_zip = os.path.join(restored_path, 'errors', 'rpe_rot.zip')

        rewards = []

        for current_step in selected_steps:
            if current_step == last_valid_step:
                rewards.append(0)
            else:
                idx_of_current_step_in_valid_steps = np.where(valid_steps == current_step)[0][0]

                if os.path.exists(trans_errors_zip) and os.path.exists(rot_errors_zip):
                    with zipfile.ZipFile(trans_errors_zip, 'r') as zip_ref:
                        with zip_ref.open(self.error_array_name) as file:
                            error_array = np.load(BytesIO(file.read()))
                            trans_error = error_array[idx_of_current_step_in_valid_steps]

                    with zipfile.ZipFile(rot_errors_zip, 'r') as zip_ref:
                        with zip_ref.open(self.error_array_name) as file:
                            error_array = np.load(BytesIO(file.read()))
                            rot_error = error_array[idx_of_current_step_in_valid_steps]

                    error = trans_error + rot_error
                    rewards.append(self.calculate_reward(error))
                else:
                    print("No error in the zip file, path:", trans_errors_zip)
                    rewards.append(-1)
        return rewards

    def _get_reward_for_valid(self, exp_dir, sub_dir, current_step, valid_steps, last_valid_step):
        restored_path = self.restore_path(exp_dir, sub_dir)
        trans_errors_zip = os.path.join(restored_path, 'errors', 'rpe_trans.zip')
        rot_errors_zip = os.path.join(restored_path, 'errors', 'rpe_rot.zip')

        if current_step == last_valid_step:
            reward = 0
        else:
            idx_of_current_step_in_valid_steps = np.where(valid_steps == current_step)[0]
            
            if len(idx_of_current_step_in_valid_steps) == 0:
                reward = -1
                return reward
            
            if os.path.exists(trans_errors_zip) and os.path.exists(rot_errors_zip):
                with zipfile.ZipFile(trans_errors_zip, 'r') as zip_ref:
                    with zip_ref.open(self.error_array_name) as file:
                        error_array = np.load(BytesIO(file.read()))
                        trans_error = error_array[idx_of_current_step_in_valid_steps]

                with zipfile.ZipFile(rot_errors_zip, 'r') as zip_ref:
                    with zip_ref.open(self.error_array_name) as file:
                        error_array = np.load(BytesIO(file.read()))
                        rot_error = error_array[idx_of_current_step_in_valid_steps]

                error = trans_error + rot_error
                reward = self.calculate_reward(error)[0]
            else:
                print("No zip file in", restored_path)
                reward = -1
        return reward

    def preprocess_trajectory(self):
        sampled_traj_name_pairs = self.random_select_trajectory()
        for exp_dir, sub_dir in sampled_traj_name_pairs:
            # print(exp_dir, sub_dir)

            # Get action parameters
            params = self.sub_dir_to_param(sub_dir)

            valid_steps = self.get_valid_indices(exp_dir, sub_dir)
            last_valid_step = valid_steps[-1]
            selected_steps = self.sample_steps(valid_steps)

            # Get states and next states
            points_in_this_traj, next_points_in_this_traj, done = self.fixed_num_sample_points(
                                                                exp_dir, selected_steps, valid_steps, last_valid_step)
            processed_points = self.pointnet_preprocess(points_in_this_traj)
            processed_next_points = self.pointnet_preprocess(next_points_in_this_traj)

            # Get rewards
            rewards = self.get_rewards(exp_dir, sub_dir, selected_steps, valid_steps, last_valid_step)

            self.rollout_storage.add_new_trajs_to_buffer(processed_points, processed_next_points, rewards, params, done)