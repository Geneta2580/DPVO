import os
import cv2
import numpy as np
from collections import namedtuple
from multiprocessing import Process, Queue
from pathlib import Path
from itertools import chain

ImuSample = namedtuple("ImuSample", ["accel", "gyro"])
# (timestamp in seconds, ImuSample) — compatible with IMUProcessor.pre_integration
ImuMeasurement = tuple


def load_euroc_imu(imu_dir):
    """Load EuRoC imu0/data.csv. First row is the header."""
    csv_path = os.path.join(imu_dir, "data.csv")
    assert os.path.exists(csv_path), csv_path

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    timestamps_s = data[:, 0] * 1e-9
    gyro = data[:, 1:4]
    accel = data[:, 4:7]
    return timestamps_s, gyro, accel


def _load_calib(calib):
    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy
    return calib, K, fx, fy, cx, cy


def _read_image_frame(imfile, calib, K, fx, fy, cx, cy):
    image = cv2.imread(str(imfile))
    if len(calib) > 4:
        image = cv2.undistort(image, K, calib[4:])

    intrinsics = np.array([fx, fy, cx, cy])
    h, w, _ = image.shape
    image = image[: h - h % 16, : w - w % 16]
    return image, intrinsics


def image_stream(queue, imagedir, calib, stride, skip=0):
    """ image generator """

    calib, K, fx, fy, cx, cy = _load_calib(calib)

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]
    assert os.path.exists(imagedir), imagedir

    for t, imfile in enumerate(image_list):
        image, intrinsics = _read_image_frame(imfile, calib, K, fx, fy, cx, cy)
        queue.put((t, image, intrinsics))

    queue.put((-1, image, intrinsics))


def euroc_stream(queue, imagedir, imu_dir, calib, stride, skip=0):
    """EuRoC image + IMU stream. Image timestamps come from filenames (ns)."""
    calib, K, fx, fy, cx, cy = _load_calib(calib)

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]
    assert os.path.exists(imagedir), imagedir
    assert os.path.exists(imu_dir), imu_dir

    imu_ts, imu_gyro, imu_accel = load_euroc_imu(imu_dir)
    imu_idx = 0

    image = intrinsics = None
    for t, imfile in enumerate(image_list):
        timestamp_ns = int(imfile.stem)
        timestamp_s = timestamp_ns * 1e-9

        imu_measurements = []
        while imu_idx < len(imu_ts) and imu_ts[imu_idx] <= timestamp_s:
            sample = ImuSample(
                accel=imu_accel[imu_idx],
                gyro=imu_gyro[imu_idx],
            )
            imu_measurements.append((imu_ts[imu_idx], sample))
            imu_idx += 1

        image, intrinsics = _read_image_frame(imfile, calib, K, fx, fy, cx, cy)
        queue.put((t, timestamp_ns, image, intrinsics, imu_measurements))

    queue.put((-1, timestamp_ns, image, intrinsics, []))


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

