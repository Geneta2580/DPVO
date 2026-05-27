from yacs.config import CfgNode as CN

_C = CN()

# max number of keyframes
_C.BUFFER_SIZE = 4096

# bias patch selection towards high gradient regions?
_C.CENTROID_SEL_STRAT = 'RANDOM'

# VO config (increase for better accuracy)
_C.PATCHES_PER_FRAME = 80
_C.REMOVAL_WINDOW = 20
_C.OPTIMIZATION_WINDOW = 12
_C.PATCH_LIFETIME = 12

# threshold for keyframe removal
_C.KEYFRAME_INDEX = 4
_C.KEYFRAME_THRESH = 12.5

# camera motion model
_C.MOTION_MODEL = 'DAMPED_LINEAR'
_C.MOTION_DAMPING = 0.5

_C.MIXED_PRECISION = True

# Loop closure
_C.LOOP_CLOSURE = False
_C.BACKEND_THRESH = 64.0
_C.MAX_EDGE_AGE = 1000
_C.GLOBAL_OPT_FREQ = 15

# Classic loop closure
_C.CLASSIC_LOOP_CLOSURE = False
_C.LOOP_CLOSE_WINDOW_SIZE = 3
_C.LOOP_RETR_THRESH = 0.04

# IMU (defaults match config/default.yaml)
_C.IMU_GRAVITY = 9.81
_C.IMU_ACCEL_NOISE = 2.0e-3
_C.IMU_GYRO_NOISE = 1.0e-4
_C.IMU_ACCEL_BIAS_RW = 1.0e-4
_C.IMU_GYRO_BIAS_RW = 1.0e-5

# VIO initialization (overridden by config/*.yaml)
_C.VIO_INIT_MIN_FRAMES = 15
_C.VIO_INIT_KEYFRAME_INTERVAL = 0.1
_C.VIO_INIT_MIN_PAIR_PARALLAX = 3.0
_C.VIO_INIT_MIN_ACC_VAR = 0.25
_C.VIO_INIT_ACCEPT_COUNT = 5
_C.VIO_INIT_ACCEPT_REL_STD = 0.05

# VIO backend config
_C.VIO_BACKEND_OPT_INTERVAL = 5
_C.VIO_BACKEND_SMART_BASE_SIGMA = 1.5
_C.VIO_BACKEND_SMART_MIN_TRACK = 3
_C.VIO_BACKEND_SMART_MIN_PARALLAX = 1.0
_C.VIO_BACKEND_SMART_MIN_WEIGHT = 0.05
_C.VIO_BACKEND_SMART_MAX_FACTORS = 2000

# Body (IMU) to camera extrinsic T_bc, 4x4 row-major (default: identity)
_C.T_BC = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]

cfg = _C
