import os
import subprocess
import re

traj_dir = "../saved_trajectories"
gt_dir = "/home/geneta/dataset/source_dataset/euroc/ground_truth"

# 精准过滤：只保留以 _vio_init_dpvo_trial01.txt 结尾的文件
traj_files = [f for f in os.listdir(traj_dir) if f.endswith("_vio_init_dpvo_trial01.txt")]
all_gt_files = os.listdir(gt_dir)

print(f"{'Sequence':<20} | {'Scale':<10} | {'APE Mean':<10}")
print("-" * 45)

for file in traj_files:
    # 从文件名提取 MH02 或 V101 这种格式
    # 匹配模式：(MH|V1|V2) + _ + 两位数字
    match = re.search(r"(MH|V1|V2)_(\d{2})", file)
    
    if match:
        # 将 MH 和 02 拼起来 -> MH02
        key = f"{match.group(1)}{match.group(2)}"
        
        # 查找 GT 文件名中包含 key 的
        gt_file = next((f for f in all_gt_files if key in f), None)
        
        if gt_file:
            traj_path = os.path.join(traj_dir, file)
            gt_path = os.path.join(gt_dir, gt_file)
            
            # 执行评估
            cmd = f"evo_ape tum {gt_path} {traj_path} -a -s -v"
            try:
                output = subprocess.check_output(cmd, shell=True, text=True)
                scale_match = re.search(r"Scale correction:\s+([\d.]+)", output)
                mean_match = re.search(r"mean\s+([\d.]+)", output)
                
                scale = scale_match.group(1) if scale_match else "N/A"
                mean = mean_match.group(1) if mean_match else "N/A"
                
                print(f"{file[:18]+'...':<20} | {scale:<10} | {mean:<10}")
            except Exception as e:
                print(f"Error processing {file}: {e}")
        else:
            print(f"WARNING: No GT file for key: {key} (Derived from {file})")
    else:
        print(f"DEBUG: Skipping file (doesn't match sequence pattern): {file}")
