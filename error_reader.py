import os
import zipfile
import numpy as np
from io import BytesIO

def get_stepwise_min_max(error_dir):
    results = {i: {"min_error": float('inf'), "max_error": float('-inf'), "min_dir": "", "max_dir": ""} for i in range(1000)}  # Assuming max 1000 steps
    
    for root, dirs, files in os.walk(error_dir):
        for dir_name in dirs:
            error_zip_path = os.path.join(root, dir_name, 'rpe_trans.zip')
            if os.path.exists(error_zip_path):
                try:
                    with zipfile.ZipFile(error_zip_path, 'r') as zip_ref:
                        with zip_ref.open('error_array.npy') as file:
                            error_array = np.load(BytesIO(file.read()))
                            
                            # 각 스텝별로 최소값, 최대값 구하기
                            for step in range(error_array.shape[0]):
                                step_min = np.min(error_array[step])
                                step_max = np.max(error_array[step])
                                
                                # 최소값이 현재 step에서 더 작으면 업데이트
                                if step_min < results[step]["min_error"]:
                                    results[step]["min_error"] = step_min
                                    results[step]["min_dir"] = dir_name
                                
                                # 최대값이 현재 step에서 더 크면 업데이트
                                if step_max > results[step]["max_error"]:
                                    results[step]["max_error"] = step_max
                                    results[step]["max_dir"] = dir_name
                except Exception as e:
                    print(f"Error processing {error_zip_path}: {e}")
                    continue

    return results

def main():
    # 베이스 디렉토리
    base_dir = "/home/lim/HILTI22/exp18_corridor_lower_gallery_2/RLIO_1122test/Hesai/ours/"
    
    # error_dir_1, error_dir_2 하위 폴더들에 대해서 처리
    results = []
    results.extend(get_stepwise_min_max(os.path.join(base_dir, "0.2_2_0.2_0.1")))
    results.extend(get_stepwise_min_max(os.path.join(base_dir, "0.6_2_0.2_0.1")))

    # 결과 출력
    for step, data in results[0]:
        min_error = data["min_error"]
        max_error = data["max_error"]
        min_dir = data["min_dir"]
        max_dir = data["max_dir"]
        
        print(f"Step {step}: Min Error = {min_error} at {min_dir}, Max Error = {max_error} at {max_dir}")
        print('-' * 50)
    # x_1 = np.arange(len(error_array_1))
    # plt.plot(x_1, error_array_1, marker='o', linestyle='-', color='b')


    # x_2 = np.arange(len(error_array_2))
    # plt.plot(x_2, error_array_2, marker='o', linestyle='-', color='r')

    # print(len(x_1), len(x_2))

    # plt.title('Line Plot of Data')
    # plt.xlabel('Index')
    # plt.ylabel('Value')

    # plt.show()


if __name__ == "__main__":
    main()