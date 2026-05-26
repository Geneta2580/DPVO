import os
import cv2
import numpy as np
from collections import namedtuple
from multiprocessing import Process, Queue
from pathlib import Path
from itertools import chain

ImuSample = namedtuple("ImuSample", ["gyro", "accel"])

def image_stream(queue, imagedir, calib, stride, skip=0):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]
    assert os.path.exists(imagedir), imagedir

    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        if 0:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])

        else:
            intrinsics = np.array([fx, fy, cx, cy])
            
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        queue.put((t, image, intrinsics))

    queue.put((-1, image, intrinsics))


def load_euroc_imu(imudir):
    """Load EuRoC IMU CSV (header line starts with #)."""
    imu_csv = Path(imudir) / "data.csv"
    assert imu_csv.exists(), imu_csv
    data = np.loadtxt(imu_csv, delimiter=",", comments="#")
    timestamps_ns = data[:, 0].astype(np.int64)
    gyro = data[:, 1:4]
    accel = data[:, 4:7]
    return timestamps_ns, gyro, accel


def collect_imu_interval(timestamps_ns, gyro, accel, imu_idx, t_start_ns, t_end_ns):
    """IMU samples with t_start_ns < t <= t_end_ns; returns updated imu_idx."""
    measurements = []
    n = len(timestamps_ns)
    while imu_idx < n and timestamps_ns[imu_idx] <= t_end_ns:
        if timestamps_ns[imu_idx] > t_start_ns:
            t_sec = timestamps_ns[imu_idx] * 1e-9
            sample = ImuSample(gyro[imu_idx], accel[imu_idx])
            measurements.append((t_sec, sample))
        imu_idx += 1
    return measurements, imu_idx


def euroc_frame_count(imagedir, stride, skip=0):
    """Number of frames euroc_stream will emit (same indexing as image list)."""
    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    return len(sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride])


def euroc_stream(queue, imagedir, imudir, calib, stride, skip=0):
    """EuRoC image + IMU generator (IMU synced to cam0 frame timestamps)."""

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]
    assert os.path.exists(imagedir), imagedir

    imu_ts_ns, imu_gyro, imu_accel = load_euroc_imu(imudir)
    imu_idx = 0
    prev_tstamp_ns = None

    image = intrinsics = None
    for t, imfile in enumerate(image_list):
        tstamp_ns = int(imfile.stem)

        image = cv2.imread(str(imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        intrinsics = np.array([fx, fy, cx, cy])

        h, w, _ = image.shape
        image = image[: h - h % 16, : w - w % 16]

        t_start_ns = prev_tstamp_ns if prev_tstamp_ns is not None else (imu_ts_ns[0] - 1)
        imu_meas, imu_idx = collect_imu_interval(
            imu_ts_ns, imu_gyro, imu_accel, imu_idx, t_start_ns, tstamp_ns
        )
        prev_tstamp_ns = tstamp_ns

        # IMU数据为时间戳/accel/gyro
        queue.put((t, tstamp_ns, image, intrinsics, imu_meas))

    queue.put((-1, -1, image, intrinsics, []))


def video_stream(queue, imagedir, calib, stride, skip=0):
    """ video generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    assert os.path.exists(imagedir), imagedir
    cap = cv2.VideoCapture(imagedir)

    t = 0

    for _ in range(skip):
        ret, image = cap.read()

    while True:
        # Capture frame-by-frame
        for _ in range(stride):
            ret, image = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                break

        if not ret:
            break

        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        intrinsics = np.array([fx*.5, fy*.5, cx*.5, cy*.5])
        queue.put((t, image, intrinsics))

        t += 1

    queue.put((-1, image, intrinsics))
    cap.release()

