from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from . import altcorr, fastba, lietorch
from . import projective_ops as pops
from .lietorch import SE3
from .net import VONet
from .patchgraph import PatchGraph
from .utils import *
from .imu_processor import IMUProcessor, prepare_measurements, preintegration_summary
from .vio_initializer import VIOInitializer
from .VIOBackend import VIOBackend
from .geometry import matrix_to_lietorch_se3_data, pose_matrix_to_tum_format

mp.set_start_method('spawn', True)


autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device="cuda")


class DPVO:

    def __init__(self, cfg, network, ht=480, wd=640, viz=False):
        self.cfg = cfg
        self.load_weights(network)
        self.is_initialized = False
        self.enable_timing = False
        torch.set_num_threads(2)

        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.ht = ht    # image height
        self.wd = wd    # image width

        DIM = self.DIM
        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = 0
        # pg.tstamps_[i] stores internal stamp id (counter); stamp_to_time maps id -> wall time (s).
        self.stamp_to_time = {}

        # keep track of global-BA calls
        self.ran_global_ba = np.zeros(100000, dtype=bool)

        ht = ht // RES
        wd = wd // RES

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        ### network attributes ###
        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}

        ### frame memory size ###
        self.pmem = self.mem = 36 # 32 was too small given default settings
        if self.cfg.LOOP_CLOSURE:
            self.last_global_ba = -1000 # keep track of time since last global opt
            self.pmem = self.cfg.MAX_EDGE_AGE # patch memory

        self.imap_ = torch.zeros(self.pmem, self.M, DIM, **kwargs)
        self.gmap_ = torch.zeros(self.pmem, self.M, 128, self.P, self.P, **kwargs)

        self.pg = PatchGraph(self.cfg, self.P, self.DIM, self.pmem, **kwargs)

        # classic backend
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.load_long_term_loop_closure()

        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.viewer = None
        if viz:
            self.start_viewer()

        # =========================== VIO 相关变量初始化 ===========================
        # IMU数据处理配置
        imu_cfg = {
            "gravity": cfg.IMU_GRAVITY,
            "accel_noise_sigma": cfg.IMU_ACCEL_NOISE,
            "gyro_noise_sigma": cfg.IMU_GYRO_NOISE,
            "accel_bias_rw_sigma": cfg.IMU_ACCEL_BIAS_RW,
            "gyro_bias_rw_sigma": cfg.IMU_GYRO_BIAS_RW,
        }
        self.imu_processor = IMUProcessor(imu_cfg)
        self.T_bc = np.asarray(cfg.T_BC, dtype=np.float64).reshape(4, 4)
        self.last_cam_t_sec = None
        self.imu_preintegrations = []
        # Global IMU buffer: (t_sec, reading) sorted by timestamp (streaming append).
        self.imu_buffer = deque()

        # V-I 初始化数据
        self.vio_initialized = False
        self.vio_init_buffer = deque(maxlen=60)
        self.vio_init_T_wc0 = None
        self.vio_init_result = None
        accept_n = int(cfg.VIO_INIT_ACCEPT_COUNT)
        self.vio_init_scale_history = deque(maxlen=accept_n)
        self.vio_init_bg_history = deque(maxlen=accept_n)
        self.vio_init_dump_path = None

        # V-I 优化后端
        self.vio_backend = None

    # ============================================= VIO 相关添加函数 =============================================
    # ====================== 辅助函数 ======================
    # ====== 辅助函数：：IMU相关函数 ======
    def push_imu_measurements(self, imu_meas):
        """Cache all IMU samples from the reader into imu_buffer (time-ordered)."""
        if not imu_meas:
            return
        for meas in prepare_measurements(imu_meas):
            t_sec = meas[0]
            if self.imu_buffer and t_sec < self.imu_buffer[-1][0]:
                # Rare out-of-order sample: keep buffer sorted by timestamp.
                self.imu_buffer.append(meas)
                self.imu_buffer = deque(sorted(self.imu_buffer, key=lambda x: x[0]))
            else:
                self.imu_buffer.append(meas)

    def pop_imu_until(self, end_time):
        """Pop and return IMU samples with timestamp <= end_time (for preintegration)."""
        return IMUProcessor.get_imu_interval_with(self.imu_buffer, end_time)

    def get_imu_between(self, t0, t1, include_start_prev=True):
        """
        Non-destructive query of IMU samples in interval (t0, t1].
        Optionally prepend the last IMU sample before or at t0.

        Returned format matches IMUProcessor.pre_integration():
            List[(t_sec, Reading(gyro, accel))]
        """
        if t1 <= t0:
            return []

        selected = []
        prev = None

        for item in self.imu_buffer:
            ts = item[0]

            if ts <= t0:
                prev = item
                continue

            if ts <= t1:
                selected.append(item)
            else:
                break

        # 取前一个值方便IMU插值，精度更高
        if include_start_prev and prev is not None:
            return [prev] + selected

        return selected

    def build_single_imu_factor(self, kf_i, kf_j):
        """
        Preintegrate all IMU samples between two consecutive init keyframes.

        Uses the full (non-destructive) imu_buffer interval (t_i, t_j] plus one
        sample at or before t_i for interpolation at segment start.
        """
        ts_i = kf_i.get_timestamp()
        ts_j = kf_j.get_timestamp()

        if ts_j <= ts_i:
            print(f"[VIO Init] invalid keyframe interval: {ts_i:.6f} -> {ts_j:.6f}")
            return None

        imu_meas = self.get_imu_between(ts_i, ts_j, include_start_prev=True)

        if len(imu_meas) < 2:
            print(
                f"[VIO Init] not enough IMU between "
                f"{ts_i:.6f} -> {ts_j:.6f}, n={len(imu_meas)}"
            )
            return None

        pim = self.imu_processor.pre_integration(
            imu_meas,
            ts_i,
            ts_j,
        )

        if pim is None:
            print(
                f"[VIO Init] preintegration failed between "
                f"{ts_i:.6f} -> {ts_j:.6f}, n_imu={len(imu_meas)}"
            )
            return None

        return {
            "start_kf_timestamp": ts_i,
            "end_kf_timestamp": ts_j,
            "imu_measurements": imu_meas,
            "imu_preintegration": pim,
        }
    
    def build_imu_factors_for_init(self, keyframes):
        """
        Build one IMU preintegration factor per consecutive init keyframe pair.

        keyframes are time-decimated by VIO_INIT_KEYFRAME_INTERVAL; each factor
        integrates the complete IMU stream between its two timestamps.
        """
        if len(keyframes) < 2:
            return None

        imu_factors = []

        for kf_i, kf_j in zip(keyframes[:-1], keyframes[1:]):
            factor = self.build_single_imu_factor(kf_i, kf_j)
            if factor is None:
                return None

            imu_factors.append(factor)

        return imu_factors

    # ====== 辅助函数：：视觉相关函数 ======
    def register_stamp_time(self, stamp_id, tstamp_sec=None, fallback=None):
        """Record wall-clock time (seconds) for an internal stamp id."""
        t = tstamp_sec if tstamp_sec is not None else fallback
        if t is not None:
            self.stamp_to_time[int(stamp_id)] = float(t)

    def get_stamp_time(self, stamp_id):
        """Query wall-clock time (seconds) from internal stamp id in pg.tstamps_."""
        return self.stamp_to_time.get(int(stamp_id))

    def get_frame_time(self, frame_idx):
        """Query wall-clock time (seconds) for active buffer slot frame_idx in [0, n)."""
        if frame_idx < 0 or frame_idx >= self.n:
            return None
        return self.get_stamp_time(self.pg.tstamps_[frame_idx])

    def get_T_wc_from_pg(self, frame_idx):
        """Return DPVO internal pose as 4x4 T_wc."""
        T = SE3(self.pg.poses_[frame_idx]).inv().matrix().detach().cpu().numpy()
        if T.ndim == 3:
            T = T[0]
        return T

    def get_T_c0_ci_from_pg(self, frame_idx):
        """
        Convert DPVO pose to VIOInitializer pose T_c0_ci.
        """
        T_wc_i = self.get_T_wc_from_pg(frame_idx)

        if self.vio_init_T_wc0 is None:
            self.vio_init_T_wc0 = T_wc_i.copy()

        return np.linalg.inv(self.vio_init_T_wc0) @ T_wc_i

    def find_active_frame_idx_by_stamp(self, stamp_id):
        """Return active pg index for a stamp id, or None if the frame was removed."""
        stamp_id = int(stamp_id)
        for i in range(self.n):
            if int(self.pg.tstamps_[i]) == stamp_id:
                return i
        return None

    def snapshot_to_active_motionmag(self, anchor_item, curr_frame_idx, robust=True):
        """
        Compute geometric patch-flow magnitude from frozen snapshot anchor to
        current active frame.

        Build a tiny 2-frame temporary tensors:
            frame 0: anchor snapshot
            frame 1: current active frame

        Then project anchor patches from frame 0 to frame 1.
        """
        if curr_frame_idx < 0 or curr_frame_idx >= self.n:
            return None

        try:
            with torch.no_grad():
                device = self.pg.poses_.device

                pose0 = anchor_item["pose_data"].to(device=device).view(1, 7)
                pose1 = self.pg.poses_[curr_frame_idx].detach().clone().to(device=device).view(1, 7)
                poses2 = torch.cat([pose0, pose1], dim=0).view(1, 2, 7)

                patch0 = anchor_item["patches"].to(device=device).view(1, self.M, 3, 3, 3)
                # dummy second-frame patches; not used because kk only indexes anchor patches
                patch1 = self.pg.patches_[curr_frame_idx].detach().clone().to(device=device).view(1, self.M, 3, 3, 3)
                patches2 = torch.cat([patch0, patch1], dim=1)

                intr0 = anchor_item["intrinsics"].to(device=device).view(1, 4)
                intr1 = self.pg.intrinsics_[curr_frame_idx].detach().clone().to(device=device).view(1, 4)
                intrinsics2 = torch.cat([intr0, intr1], dim=0).view(1, 2, 4)

                kk = torch.arange(0, self.M, device=device, dtype=torch.long)
                ii = torch.zeros_like(kk)       # source frame index 0
                jj = torch.ones_like(kk)        # target frame index 1

                flow, _ = pops.flow_mag(
                    SE3(poses2),
                    patches2,
                    intrinsics2,
                    ii,
                    jj,
                    kk,
                    beta=0.5,
                )

                if flow.numel() == 0:
                    return None

                flow = flow.float().reshape(-1)

                if robust:
                    score = torch.quantile(flow, 0.5)
                else:
                    score = flow.mean()

                score = float(score.detach().cpu().item())
                if not np.isfinite(score):
                    return None

                return score

        except Exception as e:
            print(f"[VIO Init] snapshot_to_active_motionmag failed: {e}")
            return None

    def make_vio_frame_snapshot(self, frame_idx, stamp_id, t_sec, T_c0_ci, parallax=np.nan, anchor_stamp=-1):
        """
        Snapshot all information needed by VIO initialization and future
        parallax checks. This snapshot remains valid after DPVO keyframe culling.
        """
        return {
            "stamp_id": int(stamp_id),
            "t_sec": float(t_sec),

            # Pose used by VIOInitializer.
            # Convention: T_c0_ci = inv(T_wc0) @ T_wc_i
            "pose": np.asarray(T_c0_ci, dtype=np.float64).copy(),

            # Geometric snapshot for future parallax check.
            # DPVO internal pose data is T_cw-like lietorch SE3 data.
            "pose_data": self.pg.poses_[frame_idx].detach().clone(),
            "patches": self.pg.patches_[frame_idx].detach().clone(),
            "intrinsics": self.pg.intrinsics_[frame_idx].detach().clone(),

            # Debug metadata.
            "parallax": float(parallax) if np.isfinite(parallax) else np.nan,
            "anchor_stamp": int(anchor_stamp),
        }

    def append_vio_init_frame(self, frame_idx):
        """
        Append only frames passing parallax selection into vio_init_buffer.

        The last accepted init frame is stored as a full snapshot, so it can be
        used as anchor even after DPVO keyframe culling.

        Returns:
            True if a new frame was appended, False otherwise.
        """
        if self.vio_initialized:
            return False

        if frame_idx < 0 or frame_idx >= self.n:
            return False

        stamp_id = int(self.pg.tstamps_[frame_idx])
        t_sec = self.get_stamp_time(stamp_id)
        if t_sec is None:
            return False

        T_c0_ci = self.get_T_c0_ci_from_pg(frame_idx)

        # First frame: accept directly as anchor snapshot.
        if len(self.vio_init_buffer) == 0:
            item = self.make_vio_frame_snapshot(
                frame_idx=frame_idx,
                stamp_id=stamp_id,
                t_sec=t_sec,
                T_c0_ci=T_c0_ci,
                parallax=np.nan,
                anchor_stamp=-1,
            )
            self.vio_init_buffer.append(item)
            print(f"[VIO Init] append anchor snapshot: stamp={stamp_id}")
            return True

        anchor_item = self.vio_init_buffer[-1]
        anchor_stamp = int(anchor_item["stamp_id"])
        anchor_t = float(anchor_item["t_sec"])

        # Time decimation relative to last accepted snapshot.
        min_dt = float(self.cfg.VIO_INIT_KEYFRAME_INTERVAL)
        dt = float(t_sec - anchor_t)
        if dt < min_dt:
            return False

        # Cross-window parallax check using frozen snapshot anchor.
        parallax = self.snapshot_to_active_motionmag(anchor_item, frame_idx, robust=True)

        if parallax is None:
            print(
                f"[VIO Init] snapshot parallax unavailable: "
                f"anchor_stamp={anchor_stamp}, curr_stamp={stamp_id}, dt={dt:.3f}"
            )
            return False

        min_parallax = float(self.cfg.VIO_INIT_MIN_PAIR_PARALLAX)
        if parallax < min_parallax:
            print(
                f"[VIO Init] parallax too low: "
                f"anchor_stamp={anchor_stamp}, curr_stamp={stamp_id}, "
                f"dt={dt:.3f}, {parallax:.3f} < {min_parallax:.3f}"
            )
            return False

        item = self.make_vio_frame_snapshot(
            frame_idx=frame_idx,
            stamp_id=stamp_id,
            t_sec=t_sec,
            T_c0_ci=T_c0_ci,
            parallax=parallax,
            anchor_stamp=anchor_stamp,
        )
        self.vio_init_buffer.append(item)

        print(
            f"[VIO Init] append snapshot: n={len(self.vio_init_buffer)}, "
            f"anchor_stamp={anchor_stamp}, curr_stamp={stamp_id}, "
            f"dt={dt:.3f}, parallax={parallax:.3f}"
        )
        return True

    def save_vio_init_dpvo_tum(self, path):
        """
        Save frozen DPVO visual poses in the VIO init sliding window as TUM.

        Uses self.vio_init_buffer only. This guarantees the saved trajectory is
        exactly the same visual pose set used by VIOInitializer.make_init_frames().
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.vio_init_T_wc0 is None:
            print("[VIO Init] cannot save frozen init trajectory: vio_init_T_wc0 is None")
            return

        lines = []
        for item in self.vio_init_buffer:
            T_c0_ci = np.asarray(item["pose"], dtype=np.float64)

            # Convert relative visual pose back to DPVO/world T_wc.
            # Since item["pose"] = T_c0_ci = inv(T_wc0) @ T_wc_i:
            T_wc = self.vio_init_T_wc0 @ T_c0_ci

            tx, ty, tz, qx, qy, qz, qw = pose_matrix_to_tum_format(T_wc)
            lines.append(
                f"{item['t_sec']:.10f} {tx:.10f} {ty:.10f} {tz:.10f} "
                f"{qx:.10f} {qy:.10f} {qz:.10f} {qw:.10f}"
            )

        path.write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"[VIO Init] saved FROZEN DPVO window trajectory to {path} ({len(lines)} poses)")

    # ====================== V-I初始化函数 ======================
    def check_imu_coverage(self, t0, t1):
        if len(self.imu_buffer) < 2:
            return False, None, None

        imu_t0 = self.imu_buffer[0][0]
        imu_t1 = self.imu_buffer[-1][0]

        return imu_t0 <= t0 and imu_t1 >= t1, imu_t0, imu_t1

    def check_imu_excitation(self, imu_factors):
        """
        VINS-Fusion style IMU excitation check.

        For each visual keyframe interval:
            tmp_g_i = delta_v_ij / dt_ij

        Then compute:
            excitation = sqrt( sum_i ||tmp_g_i - mean(tmp_g)||^2 / (N - 1) )

        This measures how much the preintegrated average acceleration changes
        across visual intervals.
        """
        if imu_factors is None or len(imu_factors) < 2:
            return 0.0, None

        tmp_g_list = []

        for factor_info in imu_factors:
            pim = factor_info.get("imu_preintegration", None)
            if pim is None:
                continue

            dt = float(pim.deltaTij())
            if dt <= 1e-6 or not np.isfinite(dt):
                continue

            dv = np.asarray(pim.deltaVij(), dtype=np.float64).reshape(3)
            tmp_g_list.append(dv / dt)

        if len(tmp_g_list) < 2:
            return 0.0, None

        tmp_g = np.asarray(tmp_g_list, dtype=np.float64)
        avg_g = np.mean(tmp_g, axis=0)

        excitation = np.sqrt(
            np.sum(np.linalg.norm(tmp_g - avg_g, axis=1) ** 2)
            / max(1, len(tmp_g) - 1)
        )

        return float(excitation), avg_g

    def reset_vio_init_state(self):
        """Clear VIO init buffer and acceptance history (e.g. after VO bootstrap)."""
        self.vio_init_buffer.clear()
        self.vio_init_T_wc0 = None
        self.vio_init_scale_history.clear()
        self.vio_init_bg_history.clear()
        self.vio_init_result = None

    def try_vio_initialization(self):
        """Try V-I initialization using selected VIO init frames only."""
        if self.vio_initialized:
            return True

        keyframes_all = VIOInitializer.make_init_frames(self.vio_init_buffer)
        keyframes = keyframes_all[1:] 
        n_kf = len(keyframes)

        min_frames = int(self.cfg.VIO_INIT_MIN_FRAMES)
        if n_kf < min_frames:
            return False

        t0 = keyframes[0].get_timestamp()
        t1 = keyframes[-1].get_timestamp()
        duration = float(t1 - t0)

        # 1. IMU coverage
        ok, imu_t0, imu_t1 = self.check_imu_coverage(t0, t1)
        if not ok:
            print(
                f"[VIO Init] waiting IMU coverage: "
                f"imu=[{imu_t0}, {imu_t1}], visual=[{t0}, {t1}]"
            )
            return False

        # 2. build IMU factors
        imu_factors = self.build_imu_factors_for_init(keyframes)
        if imu_factors is None or len(imu_factors) != len(keyframes) - 1:
            print("[VIO Init] failed to build imu_factors")
            return False

        # 3. IMU excitation check, VINS-Fusion style
        imu_exc, avg_g = self.check_imu_excitation(imu_factors)

        min_imu_exc = float(self.cfg.VIO_INIT_MIN_ACC_VAR)
        require_imu_exc = True

        print(
            f"[VIO Init] imu excitation={imu_exc:.4f}, "
            f"threshold={min_imu_exc:.4f}, "
            f"avg_g={avg_g if avg_g is not None else None}"
        )

        if imu_exc < min_imu_exc:
            print(
                f"[VIO Init] IMU excitation not enough: "
                f"{imu_exc:.4f} < {min_imu_exc:.4f}"
            )
            if require_imu_exc:
                return False

        # 4. solve V-I init
        ok, scale, bg, velocities, gravity_w, R_w_c0 = VIOInitializer.initialize(
            keyframes=keyframes,
            imu_factors=imu_factors,
            imu_processor=self.imu_processor,
            gravity_magnitude=self.cfg.IMU_GRAVITY,
            T_bc=self.T_bc,
        )

        if not ok or scale is None or scale <= 0 or not np.isfinite(scale):
            print(
                f"[VIO Init] solve failed: ok={ok}, scale={scale}, "
                f"frames={n_kf}, bg={bg}"
            )
            return False

        # 5. scale stability
        self.vio_init_scale_history.append(float(scale))
        self.vio_init_bg_history.append(np.asarray(bg, dtype=np.float64).reshape(-1))

        parallaxes = [
            item.get("parallax", np.nan)
            for item in self.vio_init_buffer
            if np.isfinite(item.get("parallax", np.nan))
        ]
        mean_parallax = float(np.mean(parallaxes)) if len(parallaxes) > 0 else float("nan")

        print(
            f"[VIO Init] solve ok: frames={n_kf}, duration={duration:.3f}s, "
            f"scale={scale:.4f}, mean_parallax={mean_parallax:.3f}, "
            f"bg={np.asarray(bg).reshape(-1)}, gravity_w={gravity_w}"
        )

        need = int(self.cfg.VIO_INIT_ACCEPT_COUNT)
        if len(self.vio_init_scale_history) < need:
            print(
                f"[VIO Init] collecting scale history "
                f"({len(self.vio_init_scale_history)}/{need})"
            )
            return False

        recent = np.asarray(self.vio_init_scale_history, dtype=np.float64)
        mean_scale = float(np.mean(recent))
        rel_std = float(np.std(recent) / (mean_scale + 1e-12))

        if rel_std >= float(self.cfg.VIO_INIT_ACCEPT_REL_STD):
            print(
                f"[VIO Init] scale not stable yet: "
                f"recent={recent}, rel_std={rel_std:.4f}"
            )
            return False

        # 6. accept
        self.vio_initialized = True
        self.vio_init_result = {
            "scale": float(scale),
            "bg": np.asarray(bg, dtype=np.float64).reshape(-1),
            "velocities": velocities,
            "gravity_w": gravity_w,
            "R_w_c0": R_w_c0,
            "keyframes": keyframes,
            "mean_parallax": mean_parallax,
        }

        # 暂时不建议直接写回 DPVO
        # self.apply_vio_sim3_to_dpvo_state(scale, R_w_c0)

        print(
            f"[VIO Init] accepted: scale={scale:.4f}, rel_std={rel_std:.4f}, "
            f"frames={n_kf}, duration={duration:.3f}s, "
            f"mean_parallax={mean_parallax:.3f}"
        )

        if self.vio_init_dump_path:
            self.save_vio_init_dpvo_tum(self.vio_init_dump_path)

        return True

    def _maybe_init_vio_backend(self):
        """Create VIO backend once visual-inertial initialization is accepted."""
        if not self.vio_initialized or self.vio_init_result is None:
            return

        if self.vio_backend is not None:
            return

        self.vio_backend = VIOBackend(
            cfg=self.cfg,
            T_bc=self.T_bc,
            imu_processor=self.imu_processor,
        )
        self.vio_backend.initialize_from_vio_result(
            dpvo=self,
            vio_init_result=self.vio_init_result,
        )

    def preintegrate_imu(self, imu_meas, tstamp_sec):
        measurements = prepare_measurements(imu_meas)
        if len(measurements) < 2:
            return None, None

        if self.last_cam_t_sec is not None:
            start_time = self.last_cam_t_sec
        else:
            start_time = measurements[0][0] - 1e-6

        end_time = tstamp_sec if tstamp_sec is not None else measurements[-1][0]
        pim = self.imu_processor.pre_integration(measurements, start_time, end_time)
        if pim is None:
            return None, None

        if tstamp_sec is not None:
            self.last_cam_t_sec = tstamp_sec
        else:
            self.last_cam_t_sec = end_time

        self.imu_preintegrations.append(pim)
        return pim, preintegration_summary(pim)

    def apply_vio_sim3_to_dpvo_state(self, scale, R_w_c0, t_w_c0=None):
        """
        Apply VIO initialization Sim(3) to DPVO visual state.

        Original DPVO pose convention through get_T_wc_from_pg():
            X_c0 = T_c0_ci @ X_ci

        After this function:
            X_w = T_w_ci @ X_ci

        where:
            R_w_ci = R_w_c0 @ R_c0_ci
            p_w_ci = R_w_c0 @ (scale * p_c0_ci) + t_w_c0

        Patch channel 2 is treated as inverse depth/disparity, so:
            idepth_metric = idepth_visual / scale
        """
        if getattr(self, "dpvo_metric_scaled", False):
            return

        scale = float(scale)
        R_w_c0 = np.asarray(R_w_c0, dtype=np.float64).reshape(3, 3)

        if t_w_c0 is None:
            t_w_c0 = np.zeros(3, dtype=np.float64)
        else:
            t_w_c0 = np.asarray(t_w_c0, dtype=np.float64).reshape(3)

        if scale <= 0 or not np.isfinite(scale):
            print(f"[VIO Init] invalid Sim3 scale: {scale}")
            return

        with torch.no_grad():
            for i in range(self.n):
                # 注意这里DPVO将第一帧c0作为视觉的world系
                T_c0_ci = self.get_T_wc_from_pg(i)

                T_w_ci = np.eye(4, dtype=np.float64)
                T_w_ci[:3, :3] = R_w_c0 @ T_c0_ci[:3, :3]
                T_w_ci[:3, 3] = R_w_c0 @ (scale * T_c0_ci[:3, 3]) + t_w_c0

                T_ci_w = np.linalg.inv(T_w_ci)

                # Convert 4x4 T_ci_w back to SE3 7D data.
                self.pg.poses_[i] = matrix_to_lietorch_se3_data(T_ci_w, ref_tensor=self.pg.poses_)

            # DPVO patch geometry is inverse depth / disparity.
            self.pg.patches_[:self.n, :, 2] /= scale

            # 处理相对位姿的平移部分
            for k, value in list(self.pg.delta.items()):
                t0, dP = value
                dP_data = dP.data.clone()
                dP_data[..., :3] *= scale
                self.pg.delta[k] = (t0, SE3(dP_data))

        self.dpvo_metric_scaled = True
        print(f"[VIO Init] applied Sim3 to DPVO state: scale={scale:.6f}")
    # ============================================= VIO 相关添加函数 =============================================


    def load_long_term_loop_closure(self):
        try:
            from .loop_closure.long_term import LongTermLoopClosure
            self.long_term_lc = LongTermLoopClosure(self.cfg, self.pg)
        except ModuleNotFoundError as e:
            self.cfg.CLASSIC_LOOP_CLOSURE = False
            print(f"WARNING: {e}")

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            from collections import OrderedDict
            state_dict = torch.load(network)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace('module.', '')] = v
            
            self.network = VONet()
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

    def start_viewer(self):
        from dpviewer import Viewer

        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

        self.viewer = Viewer(
            self.image_,
            self.pg.poses_,
            self.pg.points_,
            self.pg.colors_,
            intrinsics_)

    @property
    def poses(self):
        return self.pg.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.pg.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.pg.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.pg.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.pmem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.pmem * self.M, 128, 3, 3)

    @property
    def n(self):
        return self.pg.n

    @n.setter
    def n(self, val):
        self.pg.n = val

    @property
    def m(self):
        return self.pg.m

    @m.setter
    def m(self, val):
        self.pg.m = val

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.pg.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.terminate(self.n)

        if self.cfg.LOOP_CLOSURE:
            self.append_factors(*self.pg.edges_loop())

        for _ in range(12):
            self.ran_global_ba[self.n] = False
            self.update()

        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.pg.tstamps_[i]] = self.pg.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=np.float64)
        if self.viewer is not None:
            self.viewer.join()

        # Poses: x y z qx qy qz qw
        return poses, tstamps

    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.pg.kk, self.pg.jj)
        ii1 = ii % (self.M * self.pmem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.pg.ii, self.pg.jj, self.pg.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        self.pg.jj = torch.cat([self.pg.jj, jj]) # 目标帧
        self.pg.kk = torch.cat([self.pg.kk, ii]) # 全局patch索引
        self.pg.ii = torch.cat([self.pg.ii, self.ix[ii]]) # 源帧

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs) # 384维隐状态特征
        self.pg.net = torch.cat([self.pg.net, net], dim=1)

    def remove_factors(self, m, store: bool):
        assert self.pg.ii.numel() == self.pg.weight.shape[1]
        if store:
            self.pg.ii_inac = torch.cat((self.pg.ii_inac, self.pg.ii[m]))
            self.pg.jj_inac = torch.cat((self.pg.jj_inac, self.pg.jj[m]))
            self.pg.kk_inac = torch.cat((self.pg.kk_inac, self.pg.kk[m]))
            self.pg.weight_inac = torch.cat((self.pg.weight_inac, self.pg.weight[:,m]), dim=1)
            self.pg.target_inac = torch.cat((self.pg.target_inac, self.pg.target[:,m]), dim=1)
        self.pg.weight = self.pg.weight[:,~m]
        self.pg.target = self.pg.target[:,~m]

        self.pg.ii = self.pg.ii[~m]
        self.pg.jj = self.pg.jj[~m]
        self.pg.kk = self.pg.kk[~m]
        self.pg.net = self.pg.net[:,~m]
        assert self.pg.ii.numel() == self.pg.weight.shape[1]

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:,kk % (self.M * self.pmem)]
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.pg.ii == i) & (self.pg.jj == j)
        ii = self.pg.ii[k]
        jj = self.pg.jj[k]
        kk = self.pg.kk[k]

        flow, _ = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):

        i = self.n - self.cfg.KEYFRAME_INDEX - 1 # n - 5 向前5帧
        j = self.n - self.cfg.KEYFRAME_INDEX + 1 # n - 3 向前3帧
        m = self.motionmag(i, j) + self.motionmag(j, i)
 
        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.pg.tstamps_[k-1]
            t1 = self.pg.tstamps_[k]

            dP = SE3(self.pg.poses_[k]) * SE3(self.pg.poses_[k-1]).inv()
            self.pg.delta[t1] = (t0, dP)

            to_remove = (self.pg.ii == k) | (self.pg.jj == k)
            self.remove_factors(to_remove, store=False)

            self.pg.kk[self.pg.ii > k] -= self.M
            self.pg.ii[self.pg.ii > k] -= 1
            self.pg.jj[self.pg.jj > k] -= 1

            for i in range(k, self.n-1):
                self.pg.tstamps_[i] = self.pg.tstamps_[i+1]
                self.pg.colors_[i] = self.pg.colors_[i+1]
                self.pg.poses_[i] = self.pg.poses_[i+1]
                self.pg.patches_[i] = self.pg.patches_[i+1]
                self.pg.intrinsics_[i] = self.pg.intrinsics_[i+1]

                self.imap_[i % self.pmem] = self.imap_[(i+1) % self.pmem]
                self.gmap_[i % self.pmem] = self.gmap_[(i+1) % self.pmem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

            self.n -= 1
            self.m-= self.M

            if self.cfg.CLASSIC_LOOP_CLOSURE:
                self.long_term_lc.keyframe(k)

        to_remove = self.ix[self.pg.kk] < self.n - self.cfg.REMOVAL_WINDOW # Remove edges falling outside the optimization window
        if self.cfg.LOOP_CLOSURE:
            # ...unless they are being used for loop closure
            lc_edges = ((self.pg.jj - self.pg.ii) > 30) & (self.pg.jj > (self.n - self.cfg.OPTIMIZATION_WINDOW))
            to_remove = to_remove & ~lc_edges
        self.remove_factors(to_remove, store=True)

    def __run_global_BA(self):
        """ Global bundle adjustment
         Includes both active and inactive edges """
        full_target = torch.cat((self.pg.target_inac, self.pg.target), dim=1)
        full_weight = torch.cat((self.pg.weight_inac, self.pg.weight), dim=1)
        full_ii = torch.cat((self.pg.ii_inac, self.pg.ii))
        full_jj = torch.cat((self.pg.jj_inac, self.pg.jj))
        full_kk = torch.cat((self.pg.kk_inac, self.pg.kk))

        self.pg.normalize()
        lmbda = torch.as_tensor([1e-4], device="cuda")
        t0 = self.pg.ii.min().item()
        fastba.BA(self.poses, self.patches, self.intrinsics,
            full_target, full_weight, lmbda, full_ii, full_jj, full_kk, t0, self.n, M=self.M, iterations=2, eff_impl=True)
        self.ran_global_ba[self.n] = True

    def update(self):
        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject()

            with autocast(enabled=True):
                corr = self.corr(coords)
                ctx = self.imap[:, self.pg.kk % (self.M * self.pmem)]
                self.pg.net, (delta, weight, _) = \
                    self.network.update(self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk)

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[...,self.P//2,self.P//2] + delta.float()

        self.pg.target = target
        self.pg.weight = weight

        with Timer("BA", enabled=self.enable_timing):
            try:
                # run global bundle adjustment if there exist long-range edges
                if (self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1).any() and not self.ran_global_ba[self.n]:
                    self.__run_global_BA()
                else:
                    t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
                    t0 = max(t0, 1)
                    fastba.BA(self.poses, self.patches, self.intrinsics, 
                        target, weight, lmbda, self.pg.ii, self.pg.jj, self.pg.kk, t0, self.n, M=self.M, iterations=2, eff_impl=False)
            except:
                print("Warning BA failed...")

            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.pg.points_[:len(points)] = points[:]

    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')

    def __call__(self, tstamp, image, intrinsics, imu_meas=None, tstamp_sec=None):
        """ track new frame """

        # 加入IMU量测
        if imu_meas is not None:
            self.push_imu_measurements(imu_meas)

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc(image, self.n)

        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--opts BUFFER_SIZE={self.N*2}"')

        if self.viewer is not None:
            self.viewer.update_image(image.contiguous())

        image = 2 * (image[None,None] / 255.0) - 0.5
        
        with autocast(enabled=self.cfg.MIXED_PRECISION):
            # 提取patch特征，同时初始化patch的几何参数（x,y,d,全1）
            fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                    centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT, 
                    return_color=True)

        ### update state attributes ###
        stamp_id = self.counter
        wall_t = tstamp_sec if tstamp_sec is not None else float(tstamp)
        self.tlist.append(wall_t)
        self.pg.tstamps_[self.n] = stamp_id
        # 将时间戳和帧id对应映射
        self.register_stamp_time(stamp_id, tstamp_sec=tstamp_sec, fallback=tstamp)
        self.pg.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization
        clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
        self.pg.colors_[self.n] = clr.to(torch.uint8)

        self.pg.index_[self.n + 1] = self.n + 1
        self.pg.index_map_[self.n + 1] = self.m + self.M

        # 初始化外推当前帧位姿（后续使用IMU预积分外推）
        if self.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.pg.poses_[self.n-1]) # 上一帧
                P2 = SE3(self.pg.poses_[self.n-2]) # 上上一帧

                # To deal with varying camera hz
                *_, a,b,c = [1]*3 + self.tlist
                fac = (c-b) / (b-a) # 用 tlist 里最近三帧时间戳，把步长缩放到：当前间隔 / 上一间隔

                xi = self.cfg.MOTION_DAMPING * fac * (P1 * P2.inv()).log() # 外推一半位姿变化量的位姿
                tvec_qvec = (SE3.exp(xi) * P1).data # 李代数exp映射
                self.pg.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.pg.poses_[self.n] = tvec_qvec

        # TODO better depth initialization
        patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
        if self.is_initialized:
            # 对最近3帧中的所有patch的深度求中值
            # 为当前帧所有patch（Nx3x3）都赋统一值
            s = torch.median(self.pg.patches_[self.n-3:self.n,:,2])
            patches[:,:,2] = s

        self.pg.patches_[self.n] = patches

        ### update network attributes ###
        self.imap_[self.n % self.pmem] = imap.squeeze()
        self.gmap_[self.n % self.pmem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1        
        if self.n > 0 and not self.is_initialized:
            # 初始化运动检查，投影后计算相关性，然后输入更新模块预测光流运动
            if self.motion_probe() < 2.0:
                self.pg.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1
        self.m += self.M

        if self.cfg.LOOP_CLOSURE:
            if self.n - self.last_global_ba >= self.cfg.GLOBAL_OPT_FREQ:
                """ Add loop closure factors """
                lii, ljj = self.pg.edges_loop()
                if lii.numel() > 0:
                    self.last_global_ba = self.n
                    self.append_factors(lii, ljj)

        # Add forward and backward factors
        # 这里将所有满足条件的前向边以及反向边加入到patch graph中
        # 对于枚举的所有__edges_forw/back都执行append_factors操作
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        # 初始化逻辑
        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True

            for itr in range(12):
                self.update()

            self.reset_vio_init_state()
            for i in range(self.n):
                self.append_vio_init_frame(i)
            self.try_vio_initialization()
            self._maybe_init_vio_backend()

        elif self.is_initialized:
            self.update()
            # V-I初始化逻辑
            if not self.vio_initialized:
                appended = self.append_vio_init_frame(self.n - 1)
                if appended:
                    self.try_vio_initialization()
                    self._maybe_init_vio_backend()

            # 后端优化逻辑
            if self.vio_backend is not None:
                if self.counter % self.cfg.VIO_BACKEND_OPT_INTERVAL == 0:
                    self.vio_backend.optimize_current_dpvo_window_batch(self)
            
            self.keyframe()


        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.attempt_loop_closure(self.n)
            self.long_term_lc.lc_callback()
