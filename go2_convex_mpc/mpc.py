from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.linalg
import scipy.sparse as sp
import osqp

from .config import MPCConfig
from .mujoco_utils import yaw_rot, skew


@dataclass
class MPCResult:
    forces_world: np.ndarray
    success: bool
    status: str
    objective: float


class ConvexMPC:
    """
    Python implementation of the MIT Cheetah 3 convex MPC structure.

    State order:
        [yaw, pitch, roll, px, py, pz, wx, wy, wz, vx, vy, vz, g]
    """

    def __init__(self, cfg: MPCConfig):
        self.cfg = cfg
        self.prev_solution = np.zeros(12 * cfg.horizon, dtype=float)

    def _continuous_mats(self, yaw: float, foot_pos_world_rel: np.ndarray):
        A = np.zeros((13, 13), dtype=float)
        B = np.zeros((13, 12), dtype=float)

        R_yaw = yaw_rot(yaw)
        I_world = R_yaw @ self.cfg.I_body @ R_yaw.T
        I_inv = np.linalg.inv(I_world)

        A[3, 9] = 1.0
        A[4, 10] = 1.0
        A[5, 11] = 1.0
        A[11, 12] = 1.0
        A[11, 9] = self.cfg.x_drag
        A[0:3, 6:9] = R_yaw.T

        for leg in range(4):
            r = foot_pos_world_rel[leg]
            B[6:9, 3 * leg: 3 * leg + 3] = I_inv @ skew(r)
            B[9:12, 3 * leg: 3 * leg + 3] = np.eye(3) / self.cfg.mass

        return A, B

    def _discretize(self, A: np.ndarray, B: np.ndarray):
        dt = self.cfg.dt_mpc
        # [수정됨] 무거운 scipy.linalg.expm 대신 1차 테일러 전개(Euler Method) 적용
        Ad = np.eye(13, dtype=float) + A * dt
        Bd = B * dt
        return Ad, Bd

    def _stack_prediction(self, Ad: np.ndarray, Bd: np.ndarray):
        N = self.cfg.horizon
        nx = 13
        nu = 12

        Aqp = np.zeros((nx * N, nx), dtype=float)
        Bqp = np.zeros((nx * N, nu * N), dtype=float)

        powers = [np.eye(nx)]
        for _ in range(N):
            powers.append(Ad @ powers[-1])

        for r in range(N):
            Aqp[nx * r:nx * (r + 1), :] = powers[r + 1]
            for c in range(r + 1):
                Bqp[nx * r:nx * (r + 1), nu * c:nu * (c + 1)] = powers[r - c] @ Bd

        return Aqp, Bqp

    def _qp_cost(self, Aqp: np.ndarray, Bqp: np.ndarray, x0: np.ndarray, x_ref: np.ndarray):
        N = self.cfg.horizon
        Qfull = np.zeros(13, dtype=float)
        Qfull[:12] = self.cfg.Q_diag
        S = np.diag(np.tile(Qfull, N))

        Xd = np.zeros((13 * N,), dtype=float)
        for k in range(N):
            Xd[13 * k:13 * k + 12] = x_ref[k]

        H = 2.0 * (Bqp.T @ S @ Bqp + self.cfg.alpha * np.eye(12 * N))
        g = 2.0 * Bqp.T @ S @ (Aqp @ x0 - Xd)
        return 0.5 * (H + H.T), g

    def _qp_constraints(self, contact_table: np.ndarray):
        N = self.cfg.horizon
        mu_inv = 1.0 / self.cfg.mu

        f_block = np.array(
            [
                [ mu_inv, 0.0, 1.0],
                [-mu_inv, 0.0, 1.0],
                [0.0,  mu_inv, 1.0],
                [0.0, -mu_inv, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        A = np.zeros((20 * N, 12 * N), dtype=float)
        lb = np.zeros((20 * N,), dtype=float)
        ub = np.full((20 * N,), np.inf, dtype=float)

        for k in range(N):
            for leg in range(4):
                row = 20 * k + 5 * leg
                col = 12 * k + 3 * leg
                A[row:row + 5, col:col + 3] = f_block
                ub[row + 4] = float(contact_table[k, leg]) * self.cfg.fz_max
        return A, lb, ub

    def solve(
        self,
        yaw: float,
        base_pos: np.ndarray,
        base_rpy_ypr: np.ndarray,
        base_omega_world: np.ndarray,
        base_vel_world: np.ndarray,
        foot_pos_world: np.ndarray,
        x_ref: np.ndarray,
        contact_table: np.ndarray,
    ) -> MPCResult:
        foot_rel = np.asarray(foot_pos_world, dtype=float) - np.asarray(base_pos, dtype=float)[None, :]

        x0 = np.zeros((13,), dtype=float)
        x0[0] = base_rpy_ypr[0]
        x0[1] = base_rpy_ypr[1]
        x0[2] = base_rpy_ypr[2]
        x0[3:6] = base_pos
        x0[6:9] = base_omega_world
        x0[9:12] = base_vel_world
        x0[12] = -9.81

        A, B = self._continuous_mats(yaw, foot_rel)
        Ad, Bd = self._discretize(A, B)
        Aqp, Bqp = self._stack_prediction(Ad, Bd)
        H, g = self._qp_cost(Aqp, Bqp, x0, x_ref)
        Aineq, lb, ub = self._qp_constraints(contact_table)

        # OSQP에 맞게 행렬을 희소 행렬(Sparse Matrix)로 변환
        P = sp.csc_matrix(H)
        q = g
        A_osqp = sp.csc_matrix(Aineq)
        l = lb
        u_b = ub

        # OSQP 문제 객체 생성 및 설정
        prob = osqp.OSQP()
        prob.setup(
            P=P, 
            q=q, 
            A=A_osqp, 
            l=l, 
            u=u_b, 
            verbose=False,               # 터미널 출력 끄기 (실시간 제어 필수)
            max_iter=self.cfg.max_qp_iter,
            warm_start=True              # 이전 솔루션을 활용하여 연산 속도 향상
        )

        # Warm start 적용 (이전 최적 해가 있는 경우)
        prob.warm_start(x=self.prev_solution)

        # 최적화 풀기
        res = prob.solve()

        # 결과 처리
        # OSQP 상태 코드 1이나 2는 성공적으로 풀렸음을 의미합니다.
        if res.info.status_val == 1 or res.info.status_val == 2: 
            u = np.asarray(res.x, dtype=float)
            success = True
            status = res.info.status
            obj = res.info.obj_val
            self.prev_solution = u.copy()
        else:
            u = self.prev_solution.copy()
            success = False
            status = res.info.status
            obj = float("nan")

        return MPCResult(
            forces_world=u[:12].reshape(4, 3),
            success=success,
            status=status,
            objective=obj,
        )