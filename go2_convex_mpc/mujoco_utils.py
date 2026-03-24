from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def quat_to_euler_zyx_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    q_xyzw = quat_wxyz_to_xyzw(q_wxyz)
    rot = R.from_quat(q_xyzw)
    return rot.as_euler("ZYX", degrees=False).astype(float)


def rotmat_from_quat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    q_xyzw = quat_wxyz_to_xyzw(q_wxyz)
    return R.from_quat(q_xyzw).as_matrix().astype(float)


def yaw_rot(yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def foot_jacobian_world(model, data, mujoco, body_id: int, dof_indices: list[int]) -> np.ndarray:
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    return jacp[:, dof_indices]


def body_velocity_world(model, data, mujoco, body_id: int):
    vel = np.zeros(6, dtype=float)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel, 0)
    return vel[:3].copy(), vel[3:].copy()
