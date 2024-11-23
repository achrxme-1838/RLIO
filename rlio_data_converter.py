import torch
import numpy as np
import os
import random
import zipfile
from io import BytesIO
import open3d as o3d

import provider
import rlio_rollout_stoage

class RLIODataConverter:
    def __init__(self, rollout_storage: rlio_rollout_stoage.RLIORolloutStorage, 
                 num_trajs, num_points_per_scan):
        self.base_path = "/home/lim/HILTI22" 
        self.num_trajs = num_trajs
        self.num_points_per_scan = num_points_per_scan

        self.rollout_storage = rollout_storage
        self.error_array_name = 'error_array.npy'
        self.time_array_name = 'time.npy'

    def random_select_trajectory(self):
        exp_dirs = [d for d in os.listdir(self.base_path) if d.startswith('exp') and os.path.isdir(os.path.join(self.base_path, d))]
        valid_pairs = []

        for exp_dir in exp_dirs:
            # ours_path = os.path.join(self.base_path, exp_dir, 'RLIO1122test_all_pvlio', 'Hesai', 'ours')
            ours_path = os.path.join(self.base_path, exp_dir, 'RLIO_1122test', 'Hesai', 'ours')

            if os.path.exists(ours_path):
                sub_dirs = [d for d in os.listdir(ours_path) if os.path.isdir(os.path.join(ours_path, d))]
                for sub_dir in sub_dirs:
                    status_file = os.path.join(ours_path, sub_dir, 'status.txt')
                    if os.path.exists(status_file):
                        with open(status_file, 'r') as f:
                            content = f.read().strip()
                        if content == "Finished":
                            valid_pairs.append((exp_dir, sub_dir))

        if len(valid_pairs) < self.num_trajs:
            raise ValueError(f"Not enough valid trajectories found. Requested: {self.num_trajs}, Available: {len(valid_pairs)}")

        return random.sample(valid_pairs, self.num_trajs)
    
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

            selected_indices = np.array([np.argmin(np.abs(poses_times - time)) for time in time_array])
            errors = error_array

        else:
            print(f"No rpe_trans error zip file: {errors_zip}")

        return selected_indices, errors

    def fixed_num_sample_points(self, exp_dir, sub_dir, selected_indices, num_points=1024):
        """
        To make the number of points fixed, it randomly samples points from each frame.
        output: 
            points (#frames, 1024(#points/scan), 3)
            next_points (#frames, 1024(#points/scan), 3)
            done (#frames)
        """
        restored_path = self.restore_path(exp_dir, sub_dir)
        frames_path = os.path.join(restored_path, 'frames')

        points_in_this_traj = []
        next_points_in_this_traj = []
        done = [] 

        valid_indices = list(range(len(selected_indices)))

        iter = 0
        for i in selected_indices:
            pcd_name = "F_" + str(i) + "_ours_" + sub_dir + ".pcd"
            next_pcd_name = "F_" + str(i+1) + "_ours_" + sub_dir + ".pcd"

            pcd_path = os.path.join(frames_path, pcd_name)
            next_pcd_path = os.path.join(frames_path, next_pcd_name)

            if os.path.exists(pcd_path):
                pcd = o3d.io.read_point_cloud(pcd_path)
                points = np.asarray(pcd.points)

                if points.shape[0] > num_points:
                    indices = np.random.choice(points.shape[0], num_points, replace=False)
                    sampled_points = points[indices]
                else:
                    indices = np.random.choice(points.shape[0], num_points, replace=True)
                    sampled_points = points[indices]

                points_in_this_traj.append(sampled_points)
            else:
                valid_indices.remove(iter)
                if i >= 10:
                    print(f"File not found: {pcd_path}")

            if os.path.exists(pcd_path) and os.path.exists(next_pcd_path):
                next_pcd = o3d.io.read_point_cloud(next_pcd_path)
                next_points = np.asarray(next_pcd.points)

                if next_points.shape[0] > num_points:
                    indices = np.random.choice(next_points.shape[0], num_points, replace=False)
                    sampled_points = next_points[indices]
                else:
                    indices = np.random.choice(next_points.shape[0], num_points, replace=True)
                    sampled_points = next_points[indices]

                next_points_in_this_traj.append(sampled_points)
                done.append(False)
                
            else:
                if i == selected_indices[-1]:  # Real end of the trajectory
                    sampled_points = np.zeros((num_points, 3))
                    next_points_in_this_traj.append(sampled_points)
                    done.append(True)
                else:   #  Maybe just the next frame is missing -> skip this frame
                    # done.append(False)
                    pass

            iter += 1

        return np.array(points_in_this_traj), np.array(next_points_in_this_traj), valid_indices, np.array(done)  # (#frames, 1024(#points), 3)
    
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


    def preprocess_trajectory(self):
        sampled_traj_name_pairs = self.random_select_trajectory()
        for exp_dir, sub_dir in sampled_traj_name_pairs:
            # print(f"Processing: {exp_dir}/{sub_dir}")sss

            selected_indices, errors = self.preprocess_errors(exp_dir, sub_dir)
            
            params = self.sub_dir_to_param(sub_dir)  # action parameters

            points_in_this_traj, next_points_in_this_traj, valid_indices, dones = self.fixed_num_sample_points(exp_dir, sub_dir, selected_indices, num_points=1024)

            processed_points = self.pointnet_preprocess(points_in_this_traj)
            processed_next_points = self.pointnet_preprocess(next_points_in_this_traj)

            valid_errors = self.delete_invalid_errors(errors, valid_indices)
            
            self.rollout_storage.add_new_trajs_to_buffer(processed_points, processed_next_points, valid_errors, params, dones)




