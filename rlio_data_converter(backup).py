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

            selected_indices = np.array([np.argmin(np.abs(poses_times - time)) for time in time_array])
            errors = error_array

        else:
            print(f"No rpe_trans error zip file: {errors_zip}")

        # print(selected_indices)

        return selected_indices, errors

    def fixed_num_sample_points(self, exp_dir, sub_dir, selected_indices, num_points=1024):
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

        for i in selected_indices:
            path_to_pcd = os.path.join(frames_path, self.pcd_name_dict[exp_dir][i])
            path_to_next_pcd = os.path.join(frames_path, self.pcd_name_dict[exp_dir][i+1])

            # print(i, path_to_pcd)
            if os.path.exists(path_to_pcd):
                pcd = o3d.io.read_point_cloud(path_to_pcd)
                points = np.asarray(pcd.points)

                if points.shape[0] > num_points:
                    indices = np.random.choice(points.shape[0], num_points, replace=False)
                    sampled_points = points[indices]
                else:
                    indices = np.random.choice(points.shape[0], num_points, replace=True)
                    sampled_points = points[indices]

                points_in_this_traj.append(sampled_points)
            else:
                print(f"PCD File not found: {path_to_pcd}")

            if os.path.exists(path_to_pcd) and os.path.exists(path_to_next_pcd):
                next_pcd = o3d.io.read_point_cloud(path_to_next_pcd)
                next_points = np.asarray(next_pcd.points)

                if next_points.shape[0] > num_points:
                    indices = np.random.choice(next_points.shape[0], num_points, replace=False)
                    sampled_points = next_points[indices]
                else:
                    indices = np.random.choice(next_points.shape[0], num_points, replace=True)
                    sampled_points = next_points[indices]

                next_points_in_this_traj.append(sampled_points)
                done.append(False)
            elif os.path.exists(path_to_pcd) and not os.path.exists(path_to_next_pcd):
                sampled_points = np.zeros((num_points, 3))
                next_points_in_this_traj.append(sampled_points)
                done.append(True)
            else:
                print(f"Next PCD File not found: {path_to_pcd}")       

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
    
    def delete_invalid_errors(self, errors, valid_indices):
        return errors[valid_indices]
    
    def sub_dir_to_param(self, sub_dir):

        params_str = sub_dir.split('_')
        params = [float(p) for p in params_str]
        return np.array(params)

    def calculate_reward(self, errors):
        coef = 50.0 #100.0

        # print((1-coef*errors).mean())
        return 1-coef*errors
    
    def select_steps(self, valid_indices, errors):
        """
        Select the steps to be used for training.
        """
        selections = random.sample(range(len(valid_indices)), self.num_steps)
        return valid_indices[selections], errors[selections]


    def preprocess_trajectory(self):
        sampled_traj_name_pairs = self.random_select_trajectory()
        for exp_dir, sub_dir in sampled_traj_name_pairs:

            # print(f"Processing: {exp_dir}/{sub_dir}")

            params = self.sub_dir_to_param(sub_dir) # action parameters

            valid_indices, errors = self.preprocess_errors(exp_dir, sub_dir)
            selected_indices, selected_errors = self.select_steps(valid_indices, errors)
            
            # start = time.time()
            points_in_this_traj, next_points_in_this_traj, done = self.fixed_num_sample_points(exp_dir, sub_dir, selected_indices, num_points=self.num_points_per_scan)
            # end = time.time()

            processed_points = self.pointnet_preprocess(points_in_this_traj)
            processed_next_points = self.pointnet_preprocess(next_points_in_this_traj)
            
            rewards = self.calculate_reward(selected_errors)

            self.rollout_storage.add_new_trajs_to_buffer(processed_points, processed_next_points, rewards, params, done)

            # end = time.time()

            # print(f"Time taken: {end-start}")

        
