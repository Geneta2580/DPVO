#!/usr/bin/env python3
"""Compare path length of VIO-init DPVO window trajectory vs EuRoC GT (time-synced)."""

import argparse

import numpy as np


def load_tum(path, gt_ns=False):
    data = np.loadtxt(path)
    ts = data[:, 0]
    if gt_ns:
        ts = ts * 1e-9
    xyz = data[:, 1:4]
    return ts, xyz


def path_len(xyz):
    if len(xyz) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(xyz[1:] - xyz[:-1], axis=1)))


def sync_pairs(gt_ts, gt_xyz, est_ts, est_xyz, max_diff=0.01):
    pairs_gt, pairs_est = [], []
    for t, p in zip(est_ts, est_xyz):
        idx = int(np.argmin(np.abs(gt_ts - t)))
        if abs(gt_ts[idx] - t) < max_diff:
            pairs_gt.append(gt_xyz[idx])
            pairs_est.append(p)
    return np.asarray(pairs_gt), np.asarray(pairs_est)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="EuRoC GT TUM (timestamps in ns)")
    parser.add_argument("--est", required=True, help="VIO init DPVO window TUM (timestamps in s)")
    parser.add_argument("--max-diff", type=float, default=0.01, help="Max timestamp diff (s)")
    args = parser.parse_args()

    gt_ts, gt_xyz = load_tum(args.gt, gt_ns=True)
    est_ts, est_xyz = load_tum(args.est, gt_ns=False)

    pairs_gt, pairs_est = sync_pairs(gt_ts, gt_xyz, est_ts, est_xyz, args.max_diff)
    if len(pairs_gt) < 2:
        raise SystemExit(f"Too few synced pairs: {len(pairs_gt)}")

    len_gt = path_len(pairs_gt)
    len_est = path_len(pairs_est)

    print(f"Synced poses: {len(pairs_gt)}")
    print(f"GT path length:  {len_gt:.6f} m")
    print(f"EST path length: {len_est:.6f} m")
    print(f"GT/EST ratio:    {len_gt / (len_est + 1e-12):.6f}")
    print(f"EST/GT ratio:    {len_est / (len_gt + 1e-12):.6f}")


if __name__ == "__main__":
    main()
