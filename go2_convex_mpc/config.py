from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


LEG_ORDER = ("FL", "FR", "RL", "RR")


@dataclass
class MPCConfig:
    horizon: int = 10
    dt_mpc: float = 0.025
    mu: float = 0.4
    fz_max: float = 660
    alpha: float = 1e-6
    x_drag: float = 0.0

    Q_diag: np.ndarray = field(
        default_factory=lambda: np.array(
            [   1.0, 1.0, 1.0,  # [0, 1, 2] Theta (yaw, pitch, roll) weight: 1
                0.0, 0.0, 50.0, # [3, 4, 5] Position (px, py, pz): z weight 50
                0.5, 0.5, 1.0,  # [6, 7, 8] Angular velocity (wx, wy, wz): yaw rate weight 1
                1.0, 1.0, 1.0   # [9, 10, 11] Linear velocity (vx, vy, vz): v weight 1
            ],
            dtype=float,
        )
    )

    mass: float = 9.0
    I_body: np.ndarray = field(
        default_factory=lambda: np.diag([0.07, 0.26, 0.242]).astype(float)
    )

    max_qp_iter: int = 80
    trust_constr_verbose: int = 0


@dataclass
class ControlConfig:
    sim_dt: float = 0.002
    body_height: float = 0.27

    swing_kp: np.ndarray = field(
        default_factory=lambda: np.diag([100.0, 100.0, 50.0]).astype(float)
    )
    swing_kd: np.ndarray = field(
        default_factory=lambda: np.diag([10.0, 10.0, 10.0]).astype(float)
    )

    stance_joint_kd: float = 1.5

    swing_height: float = 0.10
    cycle_time: float = 0.40
    duty_factor: float = 0.5

    raibert_kv: float = 0.25
    raibert_kyaw: float = 0.05

    use_home_keyframe: bool = True
    viewer_realtime: bool = True
