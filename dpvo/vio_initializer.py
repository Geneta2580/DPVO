from fileinput import filename
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, V, B
from scipy.spatial.transform import Rotation
from .geometry import calculate_preintegration_and_jacobian, pose_matrix_to_tum_format

class VIOInitFrame:
    def __init__(self, stamp_id, t_sec, pose):
        self.stamp_id = stamp_id
        self.t_sec = float(t_sec)
        self.pose = pose.copy()

    def get_timestamp(self):
        return self.t_sec

    def get_global_pose(self):
        return self.pose

    def set_global_pose(self, T):
        self.pose = T.copy()

class VIOInitializer:
    @staticmethod
    def make_init_frames(vio_init_buffer):
        return [
            VIOInitFrame(
                item["stamp_id"],
                item["t_sec"],
                item["pose"],
            )
            for item in vio_init_buffer
        ]

    @staticmethod
    def solve_gyro_bias(keyframes, imu_factors, T_bc):
        H_b = np.zeros((3, 3))
        Z_b = np.zeros((3, 1))

        # 假设零偏为0
        initial_gyro_bias = np.zeros(3)

        for factor_info in imu_factors:
            start_ts = factor_info['start_kf_timestamp']
            end_ts = factor_info['end_kf_timestamp']
            raw_measurements = factor_info['imu_measurements']

            # 获取对应视觉KF
            kf_start = next((kf for kf in keyframes if kf.get_timestamp() == start_ts), None)
            kf_end = next((kf for kf in keyframes if kf.get_timestamp() == end_ts), None)

            if kf_start and kf_end:
                T_b0_bi_start = kf_start.get_global_pose() @ np.linalg.inv(T_bc)
                T_b0_bi_end = kf_end.get_global_pose() @ np.linalg.inv(T_bc)

                R_i_vis = gtsam.Rot3(T_b0_bi_start[:3, :3])
                R_j_vis = gtsam.Rot3(T_b0_bi_end[:3, :3])
                
                delta_R_mat, _, _, J_R_bg = calculate_preintegration_and_jacobian(
                    raw_measurements, start_ts, initial_gyro_bias
                )

                if delta_R_mat is None: continue

                delta_R_imu = gtsam.Rot3(delta_R_mat)
                
                # error_R = gtsam.Rot3.Logmap((R_i_vis.inverse().compose(R_j_vis).compose(delta_R_imu.inverse())))
                R_ij_vis = R_i_vis.inverse().compose(R_j_vis)

                error_R = gtsam.Rot3.Logmap(
                    delta_R_imu.inverse().compose(R_ij_vis)
                )
                
                H_b += J_R_bg.T @ J_R_bg
                Z_b += J_R_bg.T @ error_R.reshape(3, 1)
        try:
            bg = np.linalg.solve(H_b, Z_b)
            print(f"【System Init】: Gyro bias solved: {bg.flatten()}")            
            return bg.flatten()
        except np.linalg.LinAlgError:
            print("【System Init】: Failed to solve gyro bias")
            return None

    @staticmethod
    def repropagate_imu(imu_factors, imu_processor, gyro_bias):
        new_bias = gtsam.imuBias.ConstantBias(np.zeros(3), gyro_bias) # Accel bias暂时为0

        for factor_info in imu_factors:
            repropagated_result = imu_processor.pre_integration(
                factor_info['imu_measurements'],
                factor_info['start_kf_timestamp'],
                factor_info['end_kf_timestamp'],
                override_bias = new_bias
            )
            if repropagated_result:
                factor_info['imu_preintegration'] = repropagated_result

        return imu_factors

    @staticmethod
    def linear_alignment(keyframes, imu_factors, gravity_magnitude, T_bc):
        num_frames = len(keyframes)
        dim = 3 * num_frames + 3 + 1 # 每帧的速度、重力、尺度
        
        # 这里创建的是全局的Hessian矩阵和resisual向量
        H = np.zeros([dim, dim])
        b = np.zeros(dim)

        for i, factor_info in enumerate(imu_factors):
            pim = factor_info['imu_preintegration']
            start_ts = factor_info['start_kf_timestamp']
            end_ts = factor_info['end_kf_timestamp']

            kf_start = next((kf for kf in keyframes if kf.get_timestamp() == start_ts), None)
            kf_end = next((kf for kf in keyframes if kf.get_timestamp() == end_ts), None)

            if not kf_start or not kf_end:
                continue

            # 获取预积分时间间隔
            dt = pim.deltaTij()

            # 获取视觉KF的旋转和平移(注意这里是带外参的)
            T_i_pose = kf_start.get_global_pose()
            # print(f"【System Init】: T_i_pose: {T_i_pose}")
            R_i = (T_i_pose @ np.linalg.inv(T_bc))[:3, :3]  # R_i = R_c0_bi (IMU旋转)
            t_i = T_i_pose[:3, 3]                         # t_i = p_c0_ci (相机平移)

            T_j_pose = kf_end.get_global_pose()
            # print(f"【System Init】: T_j_pose: {T_j_pose}")
            R_j = (T_j_pose @ np.linalg.inv(T_bc))[:3, :3]  # R_j = R_c0_bj (IMU旋转)
            t_j = T_j_pose[:3, 3]                         # t_j = p_c0_cj (相机平移)

            # ======================= VIO scale debug =======================
            # if i < 10 or i % 5 == 0:
            #     vis_dp_c0 = t_j - t_i
            #     vis_dp_bi = R_i.T @ vis_dp_c0
            #     ext_term = -T_bc[:3, 3] + (R_i.T @ R_j) @ T_bc[:3, 3]
            #     imu_dp = np.asarray(pim.deltaPij()).reshape(3)
            #     imu_dv = np.asarray(pim.deltaVij()).reshape(3)
            #     vis_norm = np.linalg.norm(vis_dp_bi)
            #     imu_norm = np.linalg.norm(imu_dp)
            #     ext_norm = np.linalg.norm(ext_term)
            #     dot = float(np.dot(vis_dp_bi, imu_dp))
            #     cos = dot / (vis_norm * imu_norm + 1e-12)
            #     print(
            #         f"[ScaleDebug] factor {i}: "
            #         f"dt={dt:.4f}, "
            #         f"|vis_dp_bi|={vis_norm:.6f}, "
            #         f"|imu_dp|={imu_norm:.6f}, "
            #         f"|ext|={ext_norm:.6f}, "
            #         f"imu/vis={imu_norm / (vis_norm + 1e-12):.6f}, "
            #         f"cos={cos:.6f}"
            #     )
            #     print(f"  vis_dp_bi = {vis_dp_bi}")
            #     print(f"  imu_dp     = {imu_dp}")
            #     print(f"  imu_dv     = {imu_dv}")
            #     print(f"  ext_term   = {ext_term}")
            # ======================= VIO scale debug =======================

            # print(f"【System Init】: t_i: {t_i}")
            # print(f"【System Init】: t_j: {t_j}")

            # 创建局部的最小二乘矩阵块
            # (0:3) KF_i的速度，(3:6) KF_j的速度，(6:9) KF的重力，(9) KF的尺度
            tmp_A = np.zeros([6, 10]) 
            # (0:3) 位置误差 (3:6) 速度误差
            tmp_b = np.zeros(6)

            # 位置误差对KF_i速度的雅可比？
            tmp_A[0:3, 0:3] = -dt * np.eye(3)

            # 位置误差对重力雅可比?
            tmp_A[0:3, 6:9] = R_i.T * dt * dt / 2

            # 位置误差对尺度的雅可比,100是为了防止矩阵因为数值问题奇异
            tmp_A[0:3, 9] =  np.matmul(R_i.T, t_j - t_i) / 100.0

            # 位置误差（这里似乎和VINS-Fusion有出入，看起来是有没有加入外参的问题）
            t_ext = T_bc[:3, 3]
            tmp_b[0:3] = pim.deltaPij() + t_ext - np.matmul(R_i.T, R_j) @ t_ext

            # 速度误差对KF_i速度的雅可比？
            tmp_A[3:6, 0:3] = -np.eye(3)

            # 速度误差对KF_j速度的雅可比?
            tmp_A[3:6, 3:6] = np.matmul(R_i.T, R_j)

            # 位置误差对尺度的雅可比?
            tmp_A[3:6, 6:9] = R_i.T * dt

            # 速度误差
            tmp_b[3:6] = pim.deltaVij()

            # 创建局部的最小二乘Hessian矩阵块
            r_H = np.matmul(tmp_A.T, tmp_A)
            r_b = np.matmul(tmp_A.T, tmp_b)

            # 将局部块添加到全局Hessian矩阵
            H[i*3:i*3+6, i*3:i*3+6] += r_H[0:6, 0:6]
            b[i*3:i*3+6] += r_b[0:6]

            # 实际上是重力和尺度的Hessian部分堆叠（3+1）
            H[-4:, -4:] += r_H[-4:, -4:]
            b[-4:] += r_b[-4:]
            
            # 重力和尺度与速度交叉项的堆叠
            H[i*3:i*3+6, dim-4:] += r_H[0:6, -4:]
            H[dim-4:, i*3:i*3+6] += r_H[-4:, 0:6]
        
        # print(f"【System Init】: H: {H}")
        # print(f"【System Init】: b: {b}")
        # Debugger.visualize_matrix(H, title="Hessian Matrix", save_path="hessian_matrix.png")
        # Debugger.save_full_matrix_python(H)
        
        H = H * 1000.0
        b = b * 1000.0
        x = np.matmul(np.linalg.inv(H), b)

        scale = x[dim-1] / 100.0 # 这里就补偿了前面的100
        gravity = x[-4:-1]
        velocities = x[:3*num_frames]

        # ======================= VIO scale debug =======================
        # eigvals = np.linalg.eigvalsh(H)
        # cond = np.linalg.cond(H)
        # vel = velocities.reshape(-1, 3)
        # vel_norms = np.linalg.norm(vel, axis=1)
        # print("[ScaleDebug] linear_alignment result:")
        # print(f"  scale = {scale}")
        # print(f"  gravity = {gravity}, |g|={np.linalg.norm(gravity)}")
        # print(f"  H cond = {cond:.3e}")
        # print(f"  H eig min/max = {eigvals[0]:.3e} / {eigvals[-1]:.3e}")
        # print(
        #     f"  velocity norm min/max = "
        #     f"{vel_norms.min():.6f} / {vel_norms.max():.6f}"
        # )

        # def debug_residual_for_scale(test_scale, velocities_vec, gravity_vec):
        #     total_pos = 0.0
        #     total_vel = 0.0
        #     count = 0

        #     for k, factor_info_dbg in enumerate(imu_factors):
        #         pim_dbg = factor_info_dbg['imu_preintegration']
        #         start_ts_dbg = factor_info_dbg['start_kf_timestamp']
        #         end_ts_dbg = factor_info_dbg['end_kf_timestamp']

        #         kf_start_dbg = next(
        #             (kf for kf in keyframes if kf.get_timestamp() == start_ts_dbg), None
        #         )
        #         kf_end_dbg = next(
        #             (kf for kf in keyframes if kf.get_timestamp() == end_ts_dbg), None
        #         )
        #         if not kf_start_dbg or not kf_end_dbg:
        #             continue

        #         dt_dbg = pim_dbg.deltaTij()
        #         T_i_dbg = kf_start_dbg.get_global_pose()
        #         T_j_dbg = kf_end_dbg.get_global_pose()
        #         R_i_dbg = (T_i_dbg @ np.linalg.inv(T_bc))[:3, :3]
        #         R_j_dbg = (T_j_dbg @ np.linalg.inv(T_bc))[:3, :3]
        #         t_i_dbg = T_i_dbg[:3, 3]
        #         t_j_dbg = T_j_dbg[:3, 3]

        #         v_i = velocities_vec[k * 3:k * 3 + 3]
        #         v_j = velocities_vec[(k + 1) * 3:(k + 1) * 3 + 3]

        #         pos_pred = (
        #             -dt_dbg * v_i
        #             + R_i_dbg.T @ gravity_vec * dt_dbg * dt_dbg / 2.0
        #             + (R_i_dbg.T @ (t_j_dbg - t_i_dbg)) * test_scale
        #         )
        #         pos_b = (
        #             np.asarray(pim_dbg.deltaPij()).reshape(3)
        #             - T_bc[:3, 3]
        #             + (R_i_dbg.T @ R_j_dbg) @ T_bc[:3, 3]
        #         )
        #         vel_pred = (
        #             -v_i
        #             + (R_i_dbg.T @ R_j_dbg) @ v_j
        #             + R_i_dbg.T @ gravity_vec * dt_dbg
        #         )
        #         vel_b = np.asarray(pim_dbg.deltaVij()).reshape(3)

        #         pos_res = pos_pred - pos_b
        #         vel_res = vel_pred - vel_b
        #         total_pos += float(pos_res @ pos_res)
        #         total_vel += float(vel_res @ vel_res)
        #         count += 1

        #     if count > 0:
        #         print(
        #             f"[ScaleDebug] residual test scale={test_scale:.6f}: "
        #             f"pos_rmse={np.sqrt(total_pos / count):.6f}, "
        #             f"vel_rmse={np.sqrt(total_vel / count):.6f}, "
        #             f"count={count}"
        #         )

        # debug_residual_for_scale(scale, velocities, gravity)
        # debug_residual_for_scale(1.0, velocities, gravity)
        # ======================= VIO scale debug =======================

        return scale, gravity, velocities

    @staticmethod
    def refine_gravity(keyframes, imu_factors, gravity, gravity_magnitude, T_bc):
        g0 = gravity / np.linalg.norm(gravity) * gravity_magnitude
        
        # 初始化切平面一组基向量
        num_frames = len(keyframes)
        dim = 3 * num_frames + 2 + 1 # 每帧的速度、重力、尺度
        
        H = np.zeros([dim, dim])
        b = np.zeros(dim)

        for k in range(4):
            # 构造切平面基向量
            aa = g0 / np.linalg.norm(g0)
            tmp = np.array([.0,.0,1.0])

            bb = (tmp - np.dot(aa,tmp) * aa)
            bb /= np.linalg.norm(bb)
            cc = np.cross(aa, bb)
            bc = np.zeros([3, 2])
            bc[0:3, 0] = bb
            bc[0:3, 1] = cc
            lxly = bc

            for i, factor_info in enumerate(imu_factors):
                pim = factor_info['imu_preintegration']
                start_ts = factor_info['start_kf_timestamp']
                end_ts = factor_info['end_kf_timestamp']

                kf_start = next((kf for kf in keyframes if kf.get_timestamp() == start_ts), None)
                kf_end = next((kf for kf in keyframes if kf.get_timestamp() == end_ts), None)

                if not kf_start or not kf_end:
                    continue

                # 获取预积分时间间隔
                dt = pim.deltaTij()

                # 获取视觉KF的旋转和平移
                T_i_pose = kf_start.get_global_pose()
                R_i = (T_i_pose @ np.linalg.inv(T_bc))[:3, :3]  # R_i = R_c0_bi (IMU旋转)
                t_i = T_i_pose[:3, 3]                         # t_i = p_c0_ci (相机平移)

                T_j_pose = kf_end.get_global_pose()
                R_j = (T_j_pose @ np.linalg.inv(T_bc))[:3, :3]  # R_j = R_c0_bj (IMU旋转)
                t_j = T_j_pose[:3, 3]                         # t_j = p_c0_cj (相机平移)

                # 创建局部的最小二乘雅可比块
                # (0:3) KF_i的速度，(3:6) KF_j的速度，(6:8) KF的重力(参数化了)，(8) KF的尺度
                tmp_A = np.zeros([6, 9]) 
                # (0:3) 位置误差 (3:6) 速度误差
                tmp_b = np.zeros(6)

                # 位置误差对KF_i速度的雅可比？
                tmp_A[0:3, 0:3] = -dt * np.eye(3)

                # 位置误差对重力雅可比?
                tmp_A[0:3, 6:8] = np.matmul(R_i.T, lxly) * dt * dt / 2

                # 位置误差对尺度的雅可比?
                tmp_A[0:3, 8]=  np.matmul(R_i.T, t_j - t_i) / 100.0

                # 位置误差（这里似乎和VINS-Fusion有出入）
                t_ext = T_bc[:3, 3]
                tmp_b[0:3] = pim.deltaPij() - np.matmul(R_i.T, g0) * dt * dt / 2 + t_ext - np.matmul(R_i.T, R_j) @ t_ext

                # 速度误差对KF_i速度的雅可比？
                tmp_A[3:6, 0:3] = -np.eye(3)

                # 速度误差对KF_j速度的雅可比?
                tmp_A[3:6, 3:6] = np.matmul(R_i.T, R_j)

                # 位置误差对尺度的雅可比?
                tmp_A[3:6, 6:8] = np.matmul(R_i.T, lxly) * dt

                # 速度误差
                tmp_b[3:6] = pim.deltaVij() - np.matmul(R_i.T, g0) * dt

                r_H = np.matmul(tmp_A.T, tmp_A)
                r_b = np.matmul(tmp_A.T, tmp_b)

                # 将局部块添加到全局矩阵
                H[i*3:i*3+6, i*3:i*3+6] += r_H[0:6, 0:6]
                b[i*3:i*3+6] += r_b[0:6]
                H[-3:, -3:] += r_H[-3:, -3:]
                b[-3:] += r_b[-3:]
                
                H[i*3:i*3+6, dim-3:] += r_H[0:6, -3:]
                H[dim-3:, i*3:i*3+6] += r_H[-3:, 0:6]
        
        H = H * 1000.0
        b = b * 1000.0
        x = np.matmul(np.linalg.inv(H), b)
        delta_g = x[-3:-1]
        g0 += np.matmul(lxly, delta_g)
        g0 = g0 / np.linalg.norm(g0) * gravity_magnitude

        scale = x[dim-1] / 100.0
        velocities = x[:3*num_frames]

        print(f"【System Init】: refine scale: {scale}")
        print(f"【System Init】: refine result: {np.linalg.norm(g0)} refine gravity: {g0}")

        return scale, g0, velocities

    @staticmethod
    def align_to_world_frame(keyframes, velocities, refine_gravity, refine_scale, T_bc):
        
        final_trajectory = []

        # 外参矩阵的逆
        T_cb = np.linalg.inv(T_bc)
        R_cb = T_cb[:3, :3]

        pose_c0_c0_R = keyframes[0].get_global_pose()[:3, :3]
        pose_c0_c0_t = keyframes[0].get_global_pose()[:3, 3]

        # 应用计算的尺度因子，转换速度
        print(f"【Initializer】: Applying solved scale ({refine_scale:.4f}) to all poses and velocities...")
        for i in range(len(keyframes)):
            pose_c0_bi_with_scale = np.eye(4)
            pose_c0_ci_R = keyframes[i].get_global_pose()[:3, :3]
            pose_c0_ci_t = keyframes[i].get_global_pose()[:3, 3]

            # R_c0_ci转换为R_c0_bi
            pose_c0_bi_with_scale[:3, :3] = pose_c0_ci_R @ R_cb
            
            # 尺度因子写入pose_c0_bi
            pose_c0_bi_with_scale[:3, 3] = refine_scale * pose_c0_ci_t - pose_c0_ci_R @ R_cb @ T_bc[:3, 3] - \
                                            (refine_scale * pose_c0_c0_t - pose_c0_c0_R @ R_cb @ T_bc[:3, 3])
            keyframes[i].set_global_pose(pose_c0_bi_with_scale)

            # 速度转换到c0系下
            # velocities[i*3 : i*3+3] *= refine_scale # 这里不需要乘尺度因子，因为速度是相对的
            v_bi = velocities[i*3 : i*3+3]
            v_ci = pose_c0_ci_R @ R_cb @ v_bi
            velocities[i*3 : i*3+3] = v_ci

        # 已知两重力，求c0到w系的变换矩阵
        ng1 = refine_gravity / np.linalg.norm(refine_gravity)
        ng2 = np.array([0.0, 0.0, 1.0])

        R_w_c0_obj, _ = Rotation.align_vectors(a=[ng2], b=[ng1])
        R_w_c0 = R_w_c0_obj.as_matrix()

        # 提取其 yaw, pitch, roll
        ypr = Rotation.from_matrix(R_w_c0).as_euler('zyx', degrees=False)
        yaw = ypr[0]
        
        # 构建一个只用于消除 yaw 的旋转
        R_yaw_correction = Rotation.from_euler('z', -yaw, degrees=False).as_matrix()
        
        # 得到最终的、无yaw变化的变换矩阵
        R_final_w_c0 = R_yaw_correction @ R_w_c0

        gravity_w = R_final_w_c0 @ refine_gravity


        for i, kf in enumerate(keyframes):
            pose_c0_bi_with_scale= kf.get_global_pose() # 这里是带尺度了的
            pose_c0_bi_R_with_scale = pose_c0_bi_with_scale[:3, :3]
            pose_c0_bi_t_with_scale = pose_c0_bi_with_scale[:3, 3]
            
            # 更新KeyFrame中的全局位姿，这里不需要乘外参
            # 因为后续还可以使用global_pose来计算新帧的global_pose(T_w_ci*T_ci_cj)
            T_w_bi = np.eye(4)
            T_w_bi[:3, :3] = R_final_w_c0 @ pose_c0_bi_R_with_scale
            T_w_bi[:3, 3] = R_final_w_c0 @ pose_c0_bi_t_with_scale
            
            kf.set_global_pose(T_w_bi)

            # TODO?:对齐到世界系原点

            # 打印初始化的轨迹
            # timestamp = kf.get_timestamp()
            # pose = kf.get_global_pose()
            # T_cb = np.linalg.inv(T_bc)
            # T_w_bi = pose @ T_cb
            # tx, ty, tz, qx, qy, qz, qw = pose_matrix_to_tum_format(T_w_bi)
            # ts_sec = timestamp
            # tum_line = f"{ts_sec:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}"
            # final_trajectory.append(tum_line)
            # 打印初始化的轨迹

            # 变换速度到世界系下 (速度向量也需要旋转)，这里执行了一次原地修改
            velocities[i*3 : i*3+3] = R_final_w_c0 @ velocities[i*3 : i*3+3]

            # print(f"【Initializer】: KF {i}, Velocity: {velocities[i*3 : i*3+3]}")
        
        # 打印初始化的轨迹
        # output_path = 'test_trajectory.txt'
        # with open(output_path, "w") as f:
        #     for line in final_trajectory:
        #         f.write(line + "\n")
        # 打印初始化的轨迹
        
        print(f"【Initializer】: Alignment to world frame complete, gravity_w: {gravity_w}")

        return gravity_w, R_final_w_c0

    @staticmethod # 静态方法，不需要实例化，不需要传递self
    def initialize(keyframes, imu_factors, imu_processor, gravity_magnitude, T_bc):
        bg0 = VIOInitializer.solve_gyro_bias(keyframes, imu_factors, T_bc)
        # if bg0 is None:
        #     print("【System Init】: Failed to solve gyro bias, using zero bias")
        # bg0 = np.zeros(3)  # 失败时使用零偏置作为后备
        
        repropagated_imu_factors = VIOInitializer.repropagate_imu(imu_factors, imu_processor, bg0)
        if not repropagated_imu_factors:
            print("【System Init】: Repropagation failed.")
            return False, None, None, None, None, None

        scale, gravity, velocities = VIOInitializer.linear_alignment(keyframes, repropagated_imu_factors, gravity_magnitude, T_bc)
        if scale is None or gravity is None:
            print("【System Init】: Failed to intialize scale and gravity")
            return False, None, None, None, None, None

        refine_scale, refine_gravity, refine_velocities = VIOInitializer.refine_gravity(keyframes, repropagated_imu_factors, gravity, gravity_magnitude, T_bc)
        if refine_scale is None or refine_gravity is None:
            print("【System Init】: Failed to refine scale and gravity")
            return False, None, None, None, None, None

        gravity_w, R_final_w_c0 = VIOInitializer.align_to_world_frame(keyframes, refine_velocities, refine_gravity, refine_scale, T_bc)

        return True, refine_scale, bg0, refine_velocities, gravity_w, R_final_w_c0