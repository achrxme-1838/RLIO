import numpy as np
import os
import random
import zipfile
from io import BytesIO
import open3d as o3d
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool  # For threading

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

        # print(points.shape)

        if points.shape[0] > num_points_per_scan:
            indices = np.random.choice(points.shape[0], num_points_per_scan, replace=False)
        else:
            indices = np.random.choice(points.shape[0], num_points_per_scan, replace=True)
        sampled_points = points[indices]

        return sampled_points
    else:
        print(f"[ERROR] PCD File not found: {path_to_pcd}")
        return np.zeros((num_points_per_scan, 3))  # Return zeros if file not found

def pointnet_preprocess(points):
    """
    Make the points compatible with PointNet input.
    input: (#frames, num_points_per_scan, 3)
    output: (#frames, 3, num_points_per_scan)
    """
    points = provider.random_point_dropout(points)
    points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
    points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
    points = points.transpose(0, 2, 1)  # (#frames, 3, num_points_per_scan)
    return points

def process_single_trajectory(args):
    (exp_dir, sub_dir, base_path, num_steps, num_points_per_scan, reward_scale, pcd_name_dict) = args

    def sub_dir_to_param(sub_dir):
        params_str = sub_dir.split('_')
        params = [float(p) for p in params_str]
        return np.array(params)

    def get_valid_indices(exp_dir, sub_dir):
        restored_path = os.path.join(base_path, exp_dir, 'RLIO_1122test', 'Hesai', 'ours', sub_dir)
        poses_txt = os.path.join(restored_path, 'poses.txt')
        poses = np.loadtxt(poses_txt)  # (time, x, y, z, ...)
        poses_times = poses[:, 0]

        errors_zip = os.path.join(restored_path, 'errors', 'rpe_trans.zip')

        if os.path.exists(errors_zip):
            with zipfile.ZipFile(errors_zip, 'r') as zip_ref:
                if 'time.npy' in zip_ref.namelist():
                    with zip_ref.open('time.npy') as file:
                        time_array = np.load(BytesIO(file.read()))
                        time_array = time_array[:-1]  # Delete last element

            selected_steps = np.array([np.argmin(np.abs(poses_times - time)) for time in time_array])
        else:
            print(f"No rpe_trans error zip file: {errors_zip}")
            selected_steps = np.array([])

        return selected_steps

    def sample_steps(valid_steps):
        if len(valid_steps) >= num_steps:
            selections = random.sample(range(len(valid_steps)), num_steps)
            return valid_steps[selections]
        else:
            print("[ERROR] Not enough valid steps to sample.")
            return valid_steps

    def fixed_num_sample_points(exp_dir, selected_steps, valid_steps, last_valid_step, num_points_per_scan):
        frames_path = os.path.join(base_path, exp_dir, 'RLIO_1122test', 'LiDARs', 'Hesai')

        path_to_pcd_list = []
        path_to_next_pcd_list = []
        done_list = []

        for current_step in selected_steps:
            if current_step != last_valid_step:
                idx_of_current_step_in_valid_steps = np.where(valid_steps == current_step)[0][0]
                next_step = valid_steps[idx_of_current_step_in_valid_steps + 1]

                path_to_pcd = os.path.join(frames_path, pcd_name_dict[exp_dir][current_step])
                path_to_next_pcd = os.path.join(frames_path, pcd_name_dict[exp_dir][next_step])

                path_to_pcd_list.append(path_to_pcd)
                path_to_next_pcd_list.append(path_to_next_pcd)
                done_list.append(False)
            else:  # Last state of each trajectory
                path_to_pcd = os.path.join(frames_path, pcd_name_dict[exp_dir][current_step])

                path_to_pcd_list.append(path_to_pcd)
                path_to_next_pcd_list.append(None)  # No next point cloud
                done_list.append(True)

        # Process the point clouds sequentially or with a ThreadPool
        # Sequential processing
        points_in_this_traj = [sample_points_from_pcd(path, num_points_per_scan) for path in path_to_pcd_list]
        next_points_in_this_traj = [sample_points_from_pcd(path, num_points_per_scan) for path in path_to_next_pcd_list]

        points_in_this_traj = np.array(points_in_this_traj)
        next_points_in_this_traj = np.array(next_points_in_this_traj)
        done = np.array(done_list)

        return points_in_this_traj, next_points_in_this_traj, done

    def calculate_reward(error):
        coef = 1.0
        return (1 - coef * error) * reward_scale

    def get_rewards(exp_dir, sub_dir, selected_steps, valid_steps, last_valid_step):
        restored_path = os.path.join(base_path, exp_dir, 'RLIO_1122test', 'Hesai', 'ours', sub_dir)
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
                        with zip_ref.open('error_array.npy') as file:
                            error_array = np.load(BytesIO(file.read()))
                            trans_error = error_array[idx_of_current_step_in_valid_steps]

                    with zipfile.ZipFile(rot_errors_zip, 'r') as zip_ref:
                        with zip_ref.open('error_array.npy') as file:
                            error_array = np.load(BytesIO(file.read()))
                            rot_error = error_array[idx_of_current_step_in_valid_steps]

                    error = trans_error + rot_error
                    rewards.append(calculate_reward(error))
                else:
                    print("No error in the zip file, path:", trans_errors_zip)
                    rewards.append(-1)
        return rewards

    # Start processing the trajectory
    params = sub_dir_to_param(sub_dir)
    valid_steps = get_valid_indices(exp_dir, sub_dir)
    if len(valid_steps) == 0:
        print(f"No valid steps for {exp_dir}/{sub_dir}")
        return None
    last_valid_step = valid_steps[-1]
    selected_steps = sample_steps(valid_steps)

    points_in_this_traj, next_points_in_this_traj, done = fixed_num_sample_points(
        exp_dir, selected_steps, valid_steps, last_valid_step, num_points_per_scan)
    processed_points = pointnet_preprocess(points_in_this_traj)
    processed_next_points = pointnet_preprocess(next_points_in_this_traj)
    rewards = get_rewards(exp_dir, sub_dir, selected_steps, valid_steps, last_valid_step)

    return processed_points, processed_next_points, np.array(rewards), params, done

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
        self.set_pcd_name_array_for_each_ids()

    #--------------------------------------------------
    # Those functions are for the out of parallel computation
    def _restore_path(self, exp_dir, sub_dir):
        """
        for the out of parallel computation
        """
        restored_path = os.path.join(self.base_path, exp_dir, 'RLIO_1122test', 'Hesai', 'ours', sub_dir)
        return restored_path

    def _sample_steps(self, valid_steps):
        """
        for the out of parallel computation
        """
        if len(valid_steps) >= self.num_steps:
            selections = random.sample(range(len(valid_steps)), self.num_steps)
            return valid_steps[selections]
        else:
            print("[ERROR] Not enough valid steps to sample.")
            return valid_steps

    def _fixed_num_sample_points(self, exp_dir, selected_steps, valid_steps, last_valid_step):
        """
        for the out of parallel computation
        """
        frames_path = os.path.join(self.base_path, exp_dir, 'RLIO_1122test', 'LiDARs', 'Hesai')

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

        points_in_this_traj = [sample_points_from_pcd(path, self.num_points_per_scan) for path in path_to_pcd_list]
        next_points_in_this_traj = [sample_points_from_pcd(path, self.num_points_per_scan) for path in path_to_next_pcd_list]

        points_in_this_traj = np.array(points_in_this_traj)
        next_points_in_this_traj = np.array(next_points_in_this_traj)
        done = np.array(done_list)

        return points_in_this_traj, next_points_in_this_traj, done

    def _pointnet_preprocess(self, points):
        """
        for the out of parallel computation
        """
        points = provider.random_point_dropout(points)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points = points.transpose(0, 2, 1)  # (#frames, 3, num_points_per_scan)
        return points

    def _get_reward_for_valid(self, exp_dir, sub_dir, current_step, valid_steps, last_valid_step):
        """
        for the out of parallel computation
        """
        restored_path = self._restore_path(exp_dir, sub_dir)
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
                    if self.error_array_name in zip_ref.namelist():
                        with zip_ref.open(self.error_array_name) as file:
                            error_array = np.load(BytesIO(file.read()))
                            trans_error = error_array[idx_of_current_step_in_valid_steps]
                    else:
                        print(f"{self.error_array_name} not found in {trans_errors_zip}")
                        return -1

                with zipfile.ZipFile(rot_errors_zip, 'r') as zip_ref:
                    if self.error_array_name in zip_ref.namelist():
                        with zip_ref.open(self.error_array_name) as file:
                            error_array = np.load(BytesIO(file.read()))
                            rot_error = error_array[idx_of_current_step_in_valid_steps]
                    else:
                        print(f"{self.error_array_name} not found in {rot_errors_zip}")
                        return -1

                error = trans_error + rot_error
                reward = self._calculate_reward(error)[0]
            else:
                print(f"No zip file in {restored_path}")
                reward = -1
        return reward

    def _calculate_reward(self, error):
        """
        for the out of parallel computation
        """
        coef = 1.0
        return (1 - coef * error) * self.reward_scale

    def _get_valid_indices(self, exp_dir, sub_dir):
        """
        for the out of parallel computation
        """
        restored_path = os.path.join(self.base_path, exp_dir, 'RLIO_1122test', 'Hesai', 'ours', sub_dir)
        poses_txt = os.path.join(restored_path, 'poses.txt')
        poses = np.loadtxt(poses_txt)  # (time, x, y, z, ...)
        # poses_times = poses[:, 0]

        if poses.ndim == 1:
            print("Warning: Poses array is 1-dimensional. Unable to index with 2D slicing.")
            poses_times = poses 
        elif poses.ndim == 2:
            poses_times = poses[:, 0]
        else:
            print(f"Warning: Poses array has {poses.ndim} dimensions. Unable to index with 2D slicing.")
            poses_times = poses

        errors_zip = os.path.join(restored_path, 'errors', 'rpe_trans.zip')

        if os.path.exists(errors_zip):
            with zipfile.ZipFile(errors_zip, 'r') as zip_ref:
                if 'time.npy' in zip_ref.namelist():
                    with zip_ref.open('time.npy') as file:
                        time_array = np.load(BytesIO(file.read()))
                        time_array = time_array[:-1]  # Delete last element

            selected_steps = np.array([np.argmin(np.abs(poses_times - time)) for time in time_array])
        else:
            print(f"No rpe_trans error zip file: {errors_zip}")
            selected_steps = np.array([])

        return selected_steps

    #--------------------------------------------------

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
                    remaining_sub_dirs = list(set(sub_dirs) - set(sub_dirs_sampled))

                    if not remaining_sub_dirs:
                        print("[ERROR] the number of sub_dirs is less than the number of trajs")
                        break

                    sub_dir = random.choice(remaining_sub_dirs)
                    poses_path = os.path.join(ours_path, sub_dir, 'poses.txt')

                    if os.path.exists(poses_path):
                        with open(poses_path, 'r') as f:
                            line_count = sum(1 for _ in f)

                        if line_count >= self.num_steps * 2:
                            valid_pairs.append((exp_dir, sub_dir))
                            sub_dirs_sampled.append(sub_dir)

        return valid_pairs

    def preprocess_trajectory(self):
        sampled_traj_name_pairs = self.random_select_trajectory()
        args_list = [(exp_dir, sub_dir, self.base_path, self.num_steps, self.num_points_per_scan, self.reward_scale, self.pcd_name_dict)
                     for exp_dir, sub_dir in sampled_traj_name_pairs]

        with Pool() as pool:
            results = pool.map(process_single_trajectory, args_list)

        # Filter out any None results
        results = [res for res in results if res is not None]

        if not results:
            print("No valid trajectories were processed.")
            return

        all_processed_points = []
        all_processed_next_points = []
        all_rewards = []
        all_params = []
        all_done = []

        for processed_points, processed_next_points, rewards, params, done in results:
            all_processed_points.append(processed_points)
            all_processed_next_points.append(processed_next_points)
            all_rewards.append(rewards)
            # Repeat params for each sample in the trajectory
            params_repeated = np.tile(params, (self.num_steps, 1))
            all_params.append(params_repeated)
            all_done.append(done)

        # Convert lists to arrays
        all_processed_points = np.concatenate(all_processed_points, axis=0)
        all_processed_next_points = np.concatenate(all_processed_next_points, axis=0)
        all_rewards = np.concatenate(all_rewards)
        # all_params = np.stack(all_params)
        all_params = np.concatenate(all_params, axis=0)
        all_done = np.concatenate(all_done)

        self.rollout_storage.add_new_trajs_to_buffer(
            all_processed_points, all_processed_next_points, all_rewards, all_params, all_done)