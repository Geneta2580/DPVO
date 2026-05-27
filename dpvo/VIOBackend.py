# dpvo/VIOBackend.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import gtsam
from gtsam.symbol_shorthand import X, V, B


@dataclass
class BackendFrame:
    stamp_id: int
    frame_idx: int
    t_sec: float
    T_wc_visual: np.ndarray
    T_wb_init: np.ndarray


@dataclass
class PatchObservation:
    stamp_id: int
    frame_idx: int
    xy: np.ndarray
    weight: float
    is_anchor: bool = False


@dataclass
class PatchTrack:
    patch_id: int
    anchor_stamp_id: int
    idepth: float
    observations: List[PatchObservation]


class VIOBackend:
    """
    GTSAM backend for DPVO + IMU.

    Design goal:
        - dpvo.py only triggers backend update.
        - This file owns GTSAM states, key mapping, factor construction,
          smart factor extraction, optimization and marginalization.
    """

    def __init__(self, cfg, T_bc, imu_processor):
        self.cfg = cfg
        self.T_bc = np.asarray(T_bc, dtype=np.float64).reshape(4, 4)
        self.T_cb = np.linalg.inv(self.T_bc)
        self.imu_processor = imu_processor

        # -----------------------------
        # GTSAM active state
        # -----------------------------
        self.active_values = gtsam.Values()
        self.marg_factor = None
        self.init_priors = gtsam.NonlinearFactorGraph()

        # -----------------------------
        # ID management
        # -----------------------------
        self.stamp_to_gid: Dict[int, int] = {}
        self.gid_to_stamp: Dict[int, int] = {}
        self.active_stamps: List[int] = []
        self.next_gid: int = 0

        # -----------------------------
        # Patch / smart factor state
        # -----------------------------
        self.active_patch_ids: Set[int] = set()
        self.marginalized_patch_ids: Set[int] = set()
        self.current_smart_factors = {}

        # -----------------------------
        # Latest optimized state
        # -----------------------------
        self.latest_result = None
        self.latest_pose = None
        self.latest_velocity = None
        self.latest_bias = gtsam.imuBias.ConstantBias()

        # -----------------------------
        # Init result
        # -----------------------------
        self.initialized = False
        self.metric_scale = None
        self.gravity_w = None
        self.R_w_c0 = None

        # -----------------------------
        # Camera model
        # K will be initialized from DPVO intrinsics / RES
        # -----------------------------
        self.K = None
        self.body_T_cam = gtsam.Pose3(self.T_bc)

        # -----------------------------
        # Debug statistics
        # -----------------------------
        self.last_stats = {}

        self._setup_noise_models()

        # 使用IMU因子
        self.use_imu_factors = False

    # ============================================================
    # Basic setup
    # ============================================================

    def _setup_noise_models(self):
        self.pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3], dtype=np.float64)
        )

        self.velocity_prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.5)

        self.bias_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.5, 0.5, 0.5, 0.05, 0.05, 0.05], dtype=np.float64)
        )

        self.smart_base_sigma = float(
            getattr(self.cfg, "VIO_BACKEND_SMART_BASE_SIGMA", 1.5)
        )

    def _create_smart_params(self):
        params = gtsam.SmartProjectionParams()
        params.setLinearizationMode(gtsam.LinearizationMode.HESSIAN)
        params.setDegeneracyMode(gtsam.DegeneracyMode.ZERO_ON_DEGENERACY)
        return params

    def _get_gid(self, stamp_id: int) -> int:
        stamp_id = int(stamp_id)

        if stamp_id not in self.stamp_to_gid:
            gid = self.next_gid
            self.next_gid += 1
            self.stamp_to_gid[stamp_id] = gid
            self.gid_to_stamp[gid] = stamp_id

        return self.stamp_to_gid[stamp_id]

    def _ensure_calibration_from_dpvo(self, dpvo):
        if self.K is not None:
            return

        # Use DPVO feature-resolution intrinsics.
        # dpvo.pg.intrinsics_ has already been divided by RES.
        if dpvo.n <= 0:
            raise RuntimeError("Cannot initialize backend calibration: dpvo.n <= 0")

        intr = dpvo.pg.intrinsics_[0].detach().cpu().numpy().reshape(-1)
        fx, fy, cx, cy = map(float, intr[:4])

        self.K = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)

    def _remove_orphan_values(self, graph, values):
        graph_keys = set()

        for i in range(graph.size()):
            factor = graph.at(i)
            if factor is None:
                continue
            for key in factor.keys():
                graph_keys.add(key)

        keys_to_remove = []
        for key in values.keys():
            if key not in graph_keys:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            values.erase(key)

        if keys_to_remove:
            print(f"[VIOBackend] removed orphan values: {len(keys_to_remove)}")

        return values

    # ============================================================
    # Init from VIO initializer
    # ============================================================

    def initialize_from_vio_result(self, dpvo, vio_init_result):
        """
        Called once after VIOInitializer is accepted.
        This does not write back to DPVO frontend.
        """

        self._ensure_calibration_from_dpvo(dpvo)

        if vio_init_result is None:
            raise RuntimeError(
                "VIOBackend.initialize_from_vio_result called before VIO init succeeded"
            )

        self.metric_scale = float(vio_init_result["scale"])
        self.gravity_w = np.asarray(vio_init_result["gravity_w"], dtype=np.float64).reshape(3)
        self.R_w_c0 = np.asarray(vio_init_result["R_w_c0"], dtype=np.float64).reshape(3, 3)

        bg = np.asarray(vio_init_result["bg"], dtype=np.float64).reshape(3)
        ba = np.zeros(3, dtype=np.float64)
        self.latest_bias = gtsam.imuBias.ConstantBias(ba, bg)

        self.initialized = True

        print(
            f"[VIOBackend] initialized from VIO init: "
            f"scale={self.metric_scale:.6f}, bg={bg}"
        )

    # ============================================================
    # DPVO snapshot extraction
    # ============================================================

    def extract_frames_from_dpvo(self, dpvo) -> List[BackendFrame]:
        frames = []

        if self.metric_scale is None:
            raise RuntimeError("VIOBackend metric_scale is not initialized")

        for frame_idx in range(dpvo.n):
            stamp_id = int(dpvo.pg.tstamps_[frame_idx])
            t_sec = dpvo.get_stamp_time(stamp_id)

            if t_sec is None:
                continue

            T_wc_visual = dpvo.get_T_wc_from_pg(frame_idx)

            # First version:
            #   Use VIO scale to create metric initial pose.
            T_wc_metric = T_wc_visual.copy()
            T_wc_metric[:3, 3] *= self.metric_scale

            # camera pose -> body pose
            T_wb_init = T_wc_metric @ self.T_cb

            frames.append(
                BackendFrame(
                    stamp_id=stamp_id,
                    frame_idx=frame_idx,
                    t_sec=float(t_sec),
                    T_wc_visual=T_wc_visual,
                    T_wb_init=T_wb_init,
                )
            )

        return frames

    def extract_patch_tracks_from_dpvo(self, dpvo) -> Dict[int, PatchTrack]:
        """
        Convert DPVO patch graph into SmartFactor-style tracks.

        One DPVO patch id -> one smart factor candidate.
        """

        tracks: Dict[int, PatchTrack] = {}

        if dpvo.n <= 0 or dpvo.m <= 0:
            return tracks

        with torch.no_grad():
            ii = dpvo.pg.ii.detach().cpu().numpy()
            jj = dpvo.pg.jj.detach().cpu().numpy()
            kk = dpvo.pg.kk.detach().cpu().numpy()

            target = dpvo.pg.target.detach().cpu().numpy()
            weight = dpvo.pg.weight.detach().cpu().numpy()
            patches = dpvo.pg.patches_.detach().cpu().numpy()

            print("[VIOBackend] ii", ii.shape, "jj", jj.shape, "kk", kk.shape)
            print("[VIOBackend] target", target.shape, "weight", weight.shape)
            print("[VIOBackend] patches", patches.shape)
            print("[VIOBackend] n/m/M", dpvo.n, dpvo.m, dpvo.M)

        # Expected shapes:
        #   target: [1, E, 2]
        #   weight: [1, E, 2] or [1, E, ...]
        target = np.asarray(target)
        weight = np.asarray(weight)

        if target.ndim == 3:
            target_e = target[0]
        else:
            target_e = target

        if weight.ndim >= 3:
            weight_e = weight[0]
        else:
            weight_e = weight

        num_edges = len(kk)

        for e in range(num_edges):
            src_idx = int(ii[e])
            tgt_idx = int(jj[e])
            patch_id = int(kk[e])

            if patch_id in self.marginalized_patch_ids:
                continue

            if src_idx < 0 or src_idx >= dpvo.n:
                continue
            if tgt_idx < 0 or tgt_idx >= dpvo.n:
                continue

            anchor_idx = patch_id // dpvo.M
            patch_local = patch_id % dpvo.M

            if anchor_idx < 0 or anchor_idx >= dpvo.n:
                continue

            anchor_stamp = int(dpvo.pg.tstamps_[anchor_idx])
            tgt_stamp = int(dpvo.pg.tstamps_[tgt_idx])

            patch = patches[anchor_idx, patch_local]

            u0 = float(patch[0, 1, 1])
            v0 = float(patch[1, 1, 1])
            idepth = float(patch[2, 1, 1])

            if not np.isfinite(idepth) or idepth <= 0:
                continue

            if patch_id not in tracks:
                tracks[patch_id] = PatchTrack(
                    patch_id=patch_id,
                    anchor_stamp_id=anchor_stamp,
                    idepth=idepth,
                    observations=[
                        PatchObservation(
                            stamp_id=anchor_stamp,
                            frame_idx=anchor_idx,
                            xy=np.array([u0, v0], dtype=np.float64),
                            weight=1.0,
                            is_anchor=True,
                        )
                    ],
                )

            xy = np.asarray(target_e[e]).reshape(-1)[:2].astype(np.float64)

            if not np.all(np.isfinite(xy)):
                continue

            w = float(np.mean(np.asarray(weight_e[e]).reshape(-1)))
            if not np.isfinite(w):
                continue

            self._append_or_replace_observation(
                tracks[patch_id],
                PatchObservation(
                    stamp_id=tgt_stamp,
                    frame_idx=tgt_idx,
                    xy=xy,
                    weight=w,
                    is_anchor=False,
                ),
            )

        return tracks

    def _append_or_replace_observation(
        self, track: PatchTrack, new_obs: PatchObservation
    ) -> None:
        """
        Keep at most one observation per stamp_id inside a PatchTrack.

        DPVO patch graph is an edge list; it's possible to have multiple edges
        from the same patch to the same target frame. GTSAM smart factors allow
        at most one measurement per pose key, so we merge duplicates here.

        Strategy:
            - Prefer the observation with larger weight.
            - If weights are very close, prefer anchor observation.
        """
        sid = int(new_obs.stamp_id)

        for i, old_obs in enumerate(track.observations):
            if int(old_obs.stamp_id) != sid:
                continue

            old_score = float(old_obs.weight)
            new_score = float(new_obs.weight)

            # Tiny bias to prefer anchor when otherwise similar.
            if new_obs.is_anchor and not old_obs.is_anchor:
                new_score += 1e-3
            if old_obs.is_anchor and not new_obs.is_anchor:
                old_score += 1e-3

            if new_score > old_score:
                track.observations[i] = new_obs
            return

        track.observations.append(new_obs)

    # ============================================================
    # Filtering
    # ============================================================

    def _track_parallax(self, track: PatchTrack) -> float:
        if len(track.observations) < 2:
            return 0.0

        xy = np.stack([obs.xy for obs in track.observations], axis=0)
        center = xy.mean(axis=0)
        return float(np.max(np.linalg.norm(xy - center[None, :], axis=1)))

    def filter_patch_tracks(self, tracks: Dict[int, PatchTrack]) -> Dict[int, PatchTrack]:
        min_track = int(getattr(self.cfg, "VIO_BACKEND_SMART_MIN_TRACK", 3))
        min_parallax = float(getattr(self.cfg, "VIO_BACKEND_SMART_MIN_PARALLAX", 1.0))
        min_weight = float(getattr(self.cfg, "VIO_BACKEND_SMART_MIN_WEIGHT", 0.05))
        max_factors = int(getattr(self.cfg, "VIO_BACKEND_SMART_MAX_FACTORS", 2000))

        valid = []

        for patch_id, track in tracks.items():
            if len(track.observations) < min_track:
                continue

            mean_w = np.mean([obs.weight for obs in track.observations])
            if not np.isfinite(mean_w) or mean_w < min_weight:
                continue

            parallax = self._track_parallax(track)
            if parallax < min_parallax:
                continue

            valid.append((patch_id, track, mean_w, parallax))

        # Prefer higher weight and larger parallax tracks.
        valid.sort(key=lambda x: x[2] * x[3], reverse=True)

        valid = valid[:max_factors]

        return {patch_id: track for patch_id, track, _, _ in valid}

    # ============================================================
    # Graph construction
    # ============================================================

    def build_batch_graph_from_dpvo(self, dpvo):
        """
        First verification version:
            Build one batch graph from current DPVO window.
            No marginalization yet.
        """

        if not self.initialized:
            raise RuntimeError("VIOBackend is not initialized")

        self._ensure_calibration_from_dpvo(dpvo)

        frames = self.extract_frames_from_dpvo(dpvo)
        tracks = self.extract_patch_tracks_from_dpvo(dpvo)
        tracks = self.filter_patch_tracks(tracks)

        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        stamp_to_frame = {f.stamp_id: f for f in frames}

        # -----------------------------
        # Insert frame states
        # -----------------------------
        for i, frame in enumerate(frames):
            gid = self._get_gid(frame.stamp_id)

            pose = gtsam.Pose3(frame.T_wb_init)

            if self.latest_velocity is not None:
                vel = np.asarray(self.latest_velocity, dtype=np.float64).reshape(3)
            else:
                vel = np.zeros(3, dtype=np.float64)

            bias = self.latest_bias

            values.insert(X(gid), pose)

            if i == 0:
                graph.add(gtsam.PriorFactorPose3(X(gid), pose, self.pose_prior_noise))

            if self.use_imu_factors:
                if self.latest_velocity is not None:
                    vel = np.asarray(self.latest_velocity, dtype=np.float64).reshape(3)
                else:
                    vel = np.zeros(3, dtype=np.float64)

                bias = self.latest_bias

                values.insert(V(gid), vel)
                values.insert(B(gid), bias)

                if i == 0:
                    graph.add(gtsam.PriorFactorVector(V(gid), vel, self.velocity_prior_noise))
                    graph.add(gtsam.PriorFactorConstantBias(B(gid), bias, self.bias_prior_noise))

        # -----------------------------
        # Visual smart factors
        # -----------------------------
        smart_params = self._create_smart_params()
        self.current_smart_factors.clear()

        num_smart = 0

        for patch_id, track in tracks.items():
            obs_valid = [
                obs for obs in track.observations
                if obs.stamp_id in stamp_to_frame
            ]

            if len(obs_valid) < 2:
                continue

            mean_w = np.mean([obs.weight for obs in obs_valid])
            sigma = self.smart_base_sigma / np.sqrt(np.clip(mean_w, 0.05, 10.0))

            noise = gtsam.noiseModel.Isotropic.Sigma(2, float(sigma))

            smart_factor = gtsam.SmartProjectionPose3Factor(
                noise,
                self.K,
                self.body_T_cam,
                smart_params,
            )

            added = 0
            added_gids: Set[int] = set()
            for obs in obs_valid:
                gid = self._get_gid(obs.stamp_id)
                if gid in added_gids:
                    continue
                if not values.exists(X(gid)):
                    continue

                measurement = gtsam.Point2(float(obs.xy[0]), float(obs.xy[1]))
                smart_factor.add(measurement, X(gid))
                added_gids.add(gid)
                added += 1

            if added >= 2:
                graph.push_back(smart_factor)
                self.current_smart_factors[patch_id] = smart_factor
                num_smart += 1

        self.last_stats = {
            "num_frames": len(frames),
            "num_raw_tracks": len(tracks),
            "num_smart_factors": num_smart,
            "graph_size": graph.size(),
            "values_size": values.size(),
        }

        return graph, values, frames, tracks

    # ============================================================
    # Optimization
    # ============================================================

    def optimize_current_dpvo_window_batch(self, dpvo):
        graph, values, frames, tracks = self.build_batch_graph_from_dpvo(dpvo)
        values = self._remove_orphan_values(graph, values)

        if graph.size() == 0 or values.size() == 0:
            print("[VIOBackend] empty graph, skip optimization")
            return None

        try:
            initial_error = graph.error(values)

            params = gtsam.LevenbergMarquardtParams()
            params.setVerbosityLM("SILENT")

            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
            result = optimizer.optimize()

            final_error = graph.error(result)

        except RuntimeError as e:
            print(f"[VIOBackend] batch optimization failed: {e}")
            return None

        self.latest_result = result
        self.active_values = result

        if frames:
            latest_gid = self._get_gid(frames[-1].stamp_id)
            if result.exists(X(latest_gid)):
                self.latest_pose = result.atPose3(X(latest_gid))
            if self.use_imu_factors:
                if result.exists(V(latest_gid)):
                    self.latest_velocity = result.atVector(V(latest_gid))
                if result.exists(B(latest_gid)):
                    self.latest_bias = result.atConstantBias(B(latest_gid))

        self.last_stats.update({
            "initial_error": float(initial_error),
            "final_error": float(final_error),
        })

        print(
            "[VIOBackend] batch optimize ok: "
            f"frames={self.last_stats['num_frames']}, "
            f"smart={self.last_stats['num_smart_factors']}, "
            f"error={initial_error:.3f}->{final_error:.3f}"
        )

        return result