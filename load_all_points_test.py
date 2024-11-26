import open3d as o3d
import numpy as np

import os

import torch

def main():


    pcd_folder_path = "/home/lim/HILTI22/exp04_construction_upper_level/RLIO_1122test/LiDARs/Hesai"
    num_points_per_scan = 256

    pcd_files = [f for f in os.listdir(pcd_folder_path) if f.endswith('.pcd')]

    sampled_points_list = []

    for pcd_file in pcd_files:
        pcd_path = os.path.join(pcd_folder_path, pcd_file)

        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)

        if points.shape[0] > num_points_per_scan:
            indices = np.random.choice(points.shape[0], num_points_per_scan, replace=False)
        else:
            indices = np.random.choice(points.shape[0], num_points_per_scan, replace=True)

        sampled_points = points[indices]
        sampled_points_list.append(sampled_points)

    sampled_points_array = np.array(sampled_points_list)

    sampled_points_tensor = torch.tensor(sampled_points_array, device='cuda')

    while(1):
        print(sampled_points_tensor.shape)
    


if __name__ == "__main__":
    main()
