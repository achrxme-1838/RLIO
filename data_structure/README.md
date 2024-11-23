# lio_benchmark

### Directory Structure

- Data Directory
    ```
    ${HOME}/lbench 
    ____{DATASET_NAME} (심볼릭 링크로 실제 파일 KITTI dataset과 연결)
        |___{SCENE_NAME}
            |___Hesai (Hilti: Hesai, Mulran: Ouster ...)
                |___{TIMESTAMP}.pcd
                |___...
            |___imu.txt
            |___poses.txt (if exists)
    ```
- Result Directory
    ```
    ${HOME}/lbench/results
    ____{DATASET_NAME}
        |___{SCENE_NAME}
            |___{LiDAR_Combination}
                |___{Algorithm_Name}
                    |___{Parameter_Set_Name}
                        |___frames(# = N)
                            |___F_{idx}_{Algorithm_Name}_{Parameter_Set_Name}.pcd
                            |___...
                        |___errors
                            |___{Error_Metric}.zip
                                |___time.npy (# <= N)
                                |___error_array.npy (# <= N)
                                |___stats.json
                                |___info.json
                            |___...
                        |___params_set_desc.txt
                    |___...  
                |___...
            |___...
        |___...  
    |___...    

    ```