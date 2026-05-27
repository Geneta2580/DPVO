import os
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
from tqdm import tqdm
import evo.main_ape as main_ape
import numpy as np
import torch
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import plot_trajectory
from dpvo.stream import euroc_frame_count, euroc_stream
from dpvo.utils import Timer

# ==================== 偷天换日：用你写的函数覆盖 evo 的函数 ====================
def my_write_tum_trajectory_file(file_path, traj, confirm_overwrite=False):
    # 确保在这里面加上你的硬核转换，彻底断了科学计数法的后路
    stamps = traj.timestamps
    xyz = traj.positions_xyz
    quat = np.roll(traj.orientations_quat_wxyz, -1, axis=1)
    
    # 强制转为 float64 并应用你的 %.10f 格式
    mat = np.column_stack((stamps, xyz, quat)).astype(np.float64)
    fmt = ['%.10f'] * 8
    
    np.savetxt(file_path, mat, delimiter=" ", fmt=fmt)
    print(f"🔥 2. 成功调用了修改后的保存逻辑！保存到了: {file_path}")

# 核心：强行替换掉 file_interface 里的函数
import evo.tools.file_interface as file_interface
file_interface.write_tum_trajectory_file = my_write_tum_trajectory_file

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

@torch.no_grad()
def run(cfg, network, imagedir, imudir, calib, stride=1, viz=False, show_img=False,
        desc=None, vio_init_dump_path=None):

    slam = None

    queue = Queue(maxsize=8)
    reader = Process(target=euroc_stream, args=(queue, imagedir, imudir, calib, stride, 0))
    reader.start()

    image_timestamps = []
    pbar = tqdm(total=euroc_frame_count(imagedir, stride), desc=desc or "Processing", unit="frame")
    try:
        while 1:
            t, tstamp_ns, image, intrinsics, imu_meas = queue.get()
            if t < 0:
                break

            tstamp_sec = tstamp_ns * 1e-9
            image_timestamps.append(tstamp_ns)

            image = torch.from_numpy(image).permute(2, 0, 1).cuda()
            intrinsics = torch.from_numpy(intrinsics).cuda()

            if show_img:
                show_image(image, 1)

            if slam is None:
                slam = DPVO(cfg, network, ht=image.shape[1], wd=image.shape[2], viz=viz)
                slam.vio_init_dump_path = vio_init_dump_path

            with Timer("SLAM", enabled=False):
                slam(t, image, intrinsics, imu_meas, tstamp_sec)

            pbar.update(1)
    finally:
        pbar.close()

    reader.join()

    poses, _ = slam.terminate()
    return poses, np.array(image_timestamps, dtype=np.float64)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--config', default="config/fast.yaml")
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--show_img', action="store_true")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--eurocdir', default="datasets/EUROC")
    parser.add_argument('--backend_thresh', type=float, default=64.0)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--dump_vio_init_traj', action="store_true",
                        help="Save DPVO poses in the VIO init window as TUM when init succeeds")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.BACKEND_THRESH = args.backend_thresh
    cfg.merge_from_list(args.opts)

    print("\nRunning with config...")
    print(cfg, "\n")

    torch.manual_seed(1234)

    euroc_scenes = [
        "MH_01_easy",
        # "MH_02_easy",
        # "MH_03_medium",
        # "MH_04_difficult",
        # "MH_05_difficult",
        # "V1_01_easy",
        # "V1_02_medium",
        # "V1_03_difficult",
        # "V2_01_easy",
        # "V2_02_medium",
        # "V2_03_difficult",
    ]

    results = {}
    for scene in euroc_scenes:
        scenedir = os.path.join(args.eurocdir, scene)
        imagedir = os.path.join(scenedir, "mav0/cam0/data")
        imudir = os.path.join(scenedir, "mav0/imu0")
        groundtruth = "datasets/euroc_groundtruth/{}.txt".format(scene)

        scene_results = []
        for i in range(args.trials):
            vio_init_dump_path = None
            if args.dump_vio_init_traj:
                Path("saved_trajectories").mkdir(exist_ok=True)
                vio_init_dump_path = (
                    f"saved_trajectories/{scene}_vio_init_dpvo_trial{i + 1:02d}.txt"
                )

            traj_est, tstamps = run(
                cfg, args.network, imagedir, imudir,
                "calib/euroc.txt", args.stride, args.viz, args.show_img,
                desc=f"{scene} trial {i + 1}",
                vio_init_dump_path=vio_init_dump_path,
            )
            # Camera filename timestamps are in nanoseconds (same as EuRoC GT file header).
            tstamps_sec = tstamps * 1e-9
            assert len(traj_est) == len(tstamps_sec), (
                f"pose count {len(traj_est)} != timestamp count {len(tstamps_sec)}"
            )

            traj_est = PoseTrajectory3D(
                positions_xyz=traj_est[:, :3],
                orientations_quat_wxyz=traj_est[:, [6, 3, 4, 5]],
                timestamps=np.array(tstamps_sec))

            traj_ref = file_interface.read_tum_trajectory_file(groundtruth)
            # datasets/euroc_groundtruth stores timestamp [ns]; evo TUM reader expects seconds.
            traj_ref.timestamps = traj_ref.timestamps * 1e-9

            max_diff = 0.05 * args.stride
            traj_ref, traj_est = sync.associate_trajectories(
                traj_ref, traj_est, max_diff=max_diff)

            result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
                pose_relation=PoseRelation.translation_part, align=True, correct_scale=False)
            ate_score = result.stats["rmse"]

            Path("trajectory_plots").mkdir(exist_ok=True)
            tum_path = f"trajectory_plots/Euroc_{scene}_Trial{i+1:02d}.txt"
            print(f"1. Saving trajectory to {tum_path}")
            file_interface.write_tum_trajectory_file(tum_path, traj_est)

            if args.plot:
                plot_trajectory(traj_est, traj_ref, f"Euroc {scene} Trial #{i+1} (ATE: {ate_score:.03f})",
                                f"trajectory_plots/Euroc_{scene}_Trial{i+1:02d}.pdf", align=True, correct_scale=True)

            if args.save_trajectory:
                Path("saved_trajectories").mkdir(exist_ok=True)
                file_interface.write_tum_trajectory_file(
                    f"saved_trajectories/Euroc_{scene}_Trial{i+1:02d}.txt", traj_est)

            scene_results.append(ate_score)

        results[scene] = np.median(scene_results)
        print(scene, sorted(scene_results))

    xs = []
    for scene in results:
        print(scene, results[scene])
        xs.append(results[scene])

    print("AVG: ", np.mean(xs))

    

    
