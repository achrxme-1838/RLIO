import torch
import numpy as np
import os
import random
import zipfile
from io import BytesIO
import open3d as o3d
# import concurrent.futures

from concurrent.futures import ThreadPoolExecutor

import provider
import rlio_rollout_stoage

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.reward_sacle = reward_scale

        self.all_pcds = []
        self.exp_dirs = []

    def load_all_pcds(self):
        exp_dirs = [d for d in os.listdir(self.base_path) if d.startswith('exp') and os.path.isdir(os.path.join(self.base_path, d))]
        self.exp_dirs = exp_dirs
        print("exp_dirs : ", self.exp_dirs)

        def process_pcd(pcd_name, pcd_path):
            path_to_pcd = os.path.join(pcd_path, pcd_name)
            return self._sample_for_fixed_num_points(path_to_pcd)

        for exp_dir in exp_dirs:
            print("preprocess : ", exp_dir)
            pcd_path = os.path.join(self.base_path, exp_dir, 'RLIO_1122test', 'LiDARs', 'Hesai')
            pcd_names = [f for f in os.listdir(pcd_path) if f.endswith('.pcd')]

            points_in_this_exp = []

            with ThreadPoolExecutor() as executor:
                results = list(executor.map(lambda pcd_name: process_pcd(pcd_name, pcd_path), pcd_names))

            points_in_this_exp.extend(results)

            points_in_this_exp_np = np.array(points_in_this_exp)
            points_in_this_exp_tensor = torch.tensor(points_in_this_exp_np, device=device, requires_grad=False, dtype=torch.float32)

            self.all_pcds.append(points_in_this_exp_tensor)

    def set_pcd_name_array_for_each_ids(self):
        self.pcd_name_dict = {}
        exp_dirs = [d for d in os.listdir(self.base_path) if d.startswith('exp') and os.path.isdir(os.path.join(self.base_path, d))]

        for exp_dir in exp_dirs:
            pcd_path = os.path.join(self.base_path, exp_dir, 'RLIO_1122test', 'LiDARs', 'Hesai')
            pcd_names = [f for f in os.listdir(pcd_path) if f.endswith('.pcd')]

            self.pcd_name_dict[exp_dir] = pcd_names

    
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

    # def _sample_for_fixed_num_points(self, path_to_pcd):
    #     pcd = o3d.io.read_point_cloud(path_to_pcd)
    #     points = np.asarray(pcd.points)

    #     if points.shape[0] > self.num_points_per_scan:
    #         indices = np.random.choice(points.shape[0], self.num_points_per_scan, replace=False)
    #         sampled_points = points[indices]
    #     else:
    #         indices = np.random.choice(points.shape[0], self.num_points_per_scan, replace=True)
    #         sampled_points = points[indices]

    #     # Visualization
    #     point_cloud = o3d.geometry.PointCloud()
    #     point_cloud.points = o3d.utility.Vector3dVector(sampled_points)
    #     o3d.visualization.draw_geometries([point_cloud])

    #     return sampled_points


    def _sample_for_fixed_num_points(self, path_to_pcd):
        pcd = o3d.io.read_point_cloud(path_to_pcd)
        points = np.asarray(pcd.points)

        voxel_size = 0.1

        voxel_indices = np.floor(points / voxel_size).astype(int)

        unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)

        voxel_groups = {tuple(v): [] for v in unique_voxels}
        for idx, voxel in zip(range(len(points)), voxel_indices):
            voxel_groups[tuple(voxel)].append(idx)

        sampled_points = []
        for voxel, indices in voxel_groups.items():
            sampled_points.append(points[indices[0]])

        sampled_points = np.array(sampled_points)

        if sampled_points.shape[0] < self.num_points_per_scan:
            additional_points_needed = self.num_points_per_scan - sampled_points.shape[0]
            additional_indices = np.random.choice(sampled_points.shape[0], additional_points_needed, replace=True)
            additional_points = sampled_points[additional_indices]
            sampled_points = np.vstack([sampled_points, additional_points])

        if sampled_points.shape[0] > self.num_points_per_scan:
            sampled_points = sampled_points[:self.num_points_per_scan]

        # visualize
        # print(path_to_pcd)
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(sampled_points)
        # o3d.visualization.draw_geometries([point_cloud])

    

        return sampled_points
    

    
    def pointnet_preprocess(self, points):
        """
        Make the points compatible with PointNet input.

        input: (#frames, 1024(#points/scan), 3)
        output: (#frames, 3, 1024(#points/scan))

        """
        
        points = provider.random_point_dropout(points)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])

        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(points[0])
        # o3d.visualization.draw_geometries([point_cloud])

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
                        time_array = time_array[:-1]  # delete last element (because the error is calculated as relative manner)

            selected_steps = np.array([np.argmin(np.abs(poses_times - time)) for time in time_array])
        else:
            print(f"No rpe_trans error zip file: {errors_zip}")

        return selected_steps  # Usage: selected_steps[t] -> incex of the real time in the poses.txt (= pcd file index)

    def sample_steps(self, valid_steps):
        selections = random.sample(range(len(valid_steps)), self.num_steps)
        return valid_steps[selections]
    
    def calculate_reward(self, error):
        coef = 1.0
        
        # print(error, 1-coef*error)
        return (1-coef*error) * self.reward_sacle

    def get_rewards(self, exp_dir, sub_dir, selected_steps, valid_steps, last_valid_step):
        # sub_dir: action parameters
        restored_path = self.restore_path(exp_dir, sub_dir)
        trans_errors_zip = os.path.join(restored_path, 'errors', 'rpe_trans.zip')
        rot_errors_zip = os.path.join(restored_path, 'errors', 'rpe_rot.zip')


        rewards = []

        for current_step in selected_steps:

            if current_step == last_valid_step:
                rewards.append(0)
            else:
                idx_of_current_step_in_valid_steps = np.where(valid_steps == current_step)[0][0]
                # idx_of_next_step_in_valid_steps = idx_of_current_step_in_valid_steps + 1

                if os.path.exists(trans_errors_zip) and os.path.exists(rot_errors_zip):
                    with zipfile.ZipFile(trans_errors_zip, 'r') as zip_ref:
                        with zip_ref.open(self.error_array_name) as file:
                            error_array = np.load(BytesIO(file.read()))  # print(error_array.shape, valid_steps.shape) -> same : okay
                            trans_error = error_array[idx_of_current_step_in_valid_steps]

                    with zipfile.ZipFile(rot_errors_zip, 'r') as zip_ref:
                        with zip_ref.open(self.error_array_name) as file:
                            error_array = np.load(BytesIO(file.read()))
                            rot_error = error_array[idx_of_current_step_in_valid_steps]

                    error = trans_error + rot_error

                    # print(trans_error, rot_error)
                    rewards.append(self.calculate_reward(error))

                else: # diverge -> penalty largely
                    print("No error in the zip file, path : ", trans_errors_zip)
                    rewards.append(-1)
        
        return rewards
    

    def _get_reward_for_valid(self, exp_dir, sub_dir, current_step, valid_steps, last_valid_step):
        """
        get a single reward for a single step (for the validation only!!!)
        """

        # sub_dir: action parameters
        restored_path = self.restore_path(exp_dir, sub_dir)
        trans_errors_zip = os.path.join(restored_path, 'errors', 'rpe_trans.zip')
        rot_errors_zip = os.path.join(restored_path, 'errors', 'rpe_rot.zip')

        # rewards = []

        if current_step == last_valid_step:
            reward = 0
        else:
            idx_of_current_step_in_valid_steps = np.where(valid_steps == current_step)[0]
            
            if len(idx_of_current_step_in_valid_steps) == 0:
                # When it already diverged
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


            else: # diverge -> penalty largely
                print("No zip file in ", restored_path)
                reward = -1
        
        return reward
    
    def load_points_from_all_pcd(self, exp_dir, selected_steps, valid_steps, last_valid_step):
        exp_dir_index = self.exp_dirs.index(exp_dir)

        selected_next_steps = []
        dones = []

        for current_step in selected_steps:
            if current_step != last_valid_step:
                idx_of_current_step_in_valid_steps = np.where(valid_steps == current_step)[0][0]
                next_step = valid_steps[idx_of_current_step_in_valid_steps + 1]

                selected_next_steps.append(next_step)
                dones.append(False)
            else:

                selected_next_steps.append(last_valid_step)  # repeat the last step
                dones.append(True)
                
        points_in_this_traj = self.all_pcds[exp_dir_index][selected_steps]
        points_in_next = self.all_pcds[exp_dir_index][selected_next_steps]

        return points_in_this_traj, points_in_next ,dones

        # print(exp_dir_index)

        # return 

    
    def preprocess_trajectory(self):
        sampled_traj_name_pairs = self.random_select_trajectory()
        for exp_dir, sub_dir in sampled_traj_name_pairs:

            # print(exp_dir, sub_dir)

            # get a_t
            params = self.sub_dir_to_param(sub_dir) # action parameters

            valid_steps = self.get_valid_idices(exp_dir, sub_dir)  # Note: when we aquire pcd, it must pass through the self.pcd_name_dict
            # print(valid_steps)
            
            last_valid_step = valid_steps[-1]
            selected_steps = self.sample_steps(valid_steps)

            # get s_t, s_t+1, done (Note: s_t -> s_t+1 is deterministic in this OFF RL formulation)
            # points_in_this_traj, next_points_in_this_traj, done = self.fixed_num_sample_points(exp_dir, selected_steps, valid_steps, last_valid_step)
            points_in_this_traj, next_points_in_this_traj, done = self.load_points_from_all_pcd(exp_dir, selected_steps, valid_steps, last_valid_step) 

            points_in_this_traj_np = points_in_this_traj.cpu().numpy()
            next_points_in_this_traj_np = next_points_in_this_traj.cpu().numpy()
            
            processed_points = self.pointnet_preprocess(points_in_this_traj_np)
            processed_next_points = self.pointnet_preprocess(next_points_in_this_traj_np)

            # get r_t (no r_t+1) (the goal is minimize current error, and the current error is calculated by the current action)
            rewards = self.get_rewards(exp_dir, sub_dir, selected_steps, valid_steps, last_valid_step)

            self.rollout_storage.add_new_trajs_to_buffer(processed_points, processed_next_points, rewards, params, done)

        
