from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np



from .config import MPCConfig, ControlConfig, LEG_ORDER
from .gait import GaitScheduler
from .swing import FootSwingTrajectory
from .mujoco_utils import body_velocity_world, foot_jacobian_world, quat_to_euler_zyx_wxyz, rotmat_from_quat_wxyz
from .mpc import ConvexMPC


@dataclass
class Command:
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    body_height: float = 0.27


class Go2ConvexMPCController:
    def __init__(self, model, data, mujoco, mpc_cfg: MPCConfig, ctrl_cfg: ControlConfig, mode: str = "stand", command: Command | None = None):
        self.model = model
        self.data = data
        self.mujoco = mujoco
        self.mpc_cfg = mpc_cfg
        self.ctrl_cfg = ctrl_cfg
        self.command = command if command is not None else Command(body_height=ctrl_cfg.body_height)

        self.scheduler = GaitScheduler(mode=mode, cycle_time=ctrl_cfg.cycle_time, duty_factor=ctrl_cfg.duty_factor, horizon=mpc_cfg.horizon, dt_mpc=mpc_cfg.dt_mpc)
        self.mpc = ConvexMPC(mpc_cfg)

        self.base_body_id = self._body_id("base_link")
        self.foot_body_ids = {leg: self._body_id(f"{leg}_foot") for leg in LEG_ORDER}

        self.leg_joint_names = {leg: [f"{leg}_hip_joint", f"{leg}_thigh_joint", f"{leg}_calf_joint"] for leg in LEG_ORDER}
        self.leg_actuator_names = {leg: [f"{leg}_hip", f"{leg}_thigh", f"{leg}_calf"] for leg in LEG_ORDER}

        self.leg_dof_indices = {leg: [int(self.model.jnt_dofadr[self._joint_id(name)]) for name in names] for leg, names in self.leg_joint_names.items()}
        self.leg_actuator_ids = {leg: [self._actuator_id(name) for name in names] for leg, names in self.leg_actuator_names.items()}

        self.home_foot_positions_body = self._measure_home_foot_positions_body()
        self.swing_traj = {leg: FootSwingTrajectory(height=ctrl_cfg.swing_height) for leg in LEG_ORDER}
        self.swing_active_prev = {leg: False for leg in LEG_ORDER}
        self.world_position_desired = None
        self.last_mpc_forces_world = np.zeros((4, 3), dtype=float)

        self.mpc_steps = max(1, int(round(self.mpc_cfg.dt_mpc / self.ctrl_cfg.sim_dt)))
        self.sim_step_counter = 0

    def _body_id(self, name: str) -> int:
        bid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise RuntimeError(f"Body '{name}' not found in model.")
        return int(bid)

    def _joint_id(self, name: str) -> int:
        jid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Joint '{name}' not found in model.")
        return int(jid)

    def _actuator_id(self, name: str) -> int:
        aid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise RuntimeError(f"Actuator '{name}' not found in model.")
        return int(aid)

    def _measure_home_foot_positions_body(self):
        tmp = self.mujoco.MjData(self.model)
        if self.ctrl_cfg.use_home_keyframe:
            kid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_KEY, "home")
            if kid >= 0:
                self.mujoco.mj_resetDataKeyframe(self.model, tmp, int(kid))
            else:
                self.mujoco.mj_resetData(self.model, tmp)
        else:
            self.mujoco.mj_resetData(self.model, tmp)

        self.mujoco.mj_forward(self.model, tmp)

        # [수정됨] xpos 대신 xipos(관성 중심) 사용
        base_pos = tmp.xipos[self.base_body_id].copy()
        base_quat = tmp.xquat[self.base_body_id].copy()
        Rwb = rotmat_from_quat_wxyz(base_quat)

        out = {}
        for leg, bid in self.foot_body_ids.items():
            pf = tmp.xpos[bid].copy()
            out[leg] = Rwb.T @ (pf - base_pos)
            
        return out

    def reset_pose(self):
        if self.ctrl_cfg.use_home_keyframe:
            kid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_KEY, "home")
            if kid >= 0:
                self.mujoco.mj_resetDataKeyframe(self.model, self.data, int(kid))
            else:
                self.mujoco.mj_resetData(self.model, self.data)
        else:
            self.mujoco.mj_resetData(self.model, self.data)

        self.mujoco.mj_forward(self.model, self.data)
        base_pos, base_quat, ypr, omega_w, vel_w = self._base_state()
        self.world_position_desired = np.array([base_pos[0], base_pos[1], self.command.body_height], dtype=float)
        self.sim_step_counter = 0
        for leg in LEG_ORDER:
            self.swing_active_prev[leg] = False

    def _base_state(self):
        # [수정됨] xpos 대신 xipos(관성 중심) 사용
        base_pos = self.data.xipos[self.base_body_id].copy()
        base_quat = self.data.xquat[self.base_body_id].copy()
        ypr = quat_to_euler_zyx_wxyz(base_quat)
        omega_w, vel_w = body_velocity_world(self.model, self.data, self.mujoco, self.base_body_id)
        return base_pos, base_quat, ypr, omega_w, vel_w

    def _foot_positions_world(self):
        return {leg: self.data.xpos[bid].copy() for leg, bid in self.foot_body_ids.items()}

    def _foot_vel_world(self, leg: str):
        jac = foot_jacobian_world(self.model, self.data, self.mujoco, self.foot_body_ids[leg], self.leg_dof_indices[leg])
        qd_leg = np.array([self.data.qvel[idx] for idx in self.leg_dof_indices[leg]], dtype=float)
        return jac @ qd_leg

    def _build_reference(self, base_pos: np.ndarray, ypr: np.ndarray):
        N = self.mpc_cfg.horizon
        dt = self.mpc_cfg.dt_mpc
        xref = np.zeros((N, 12), dtype=float)

        yaw = ypr[0]
        for k in range(N):
            t = k * dt
            xref[k, 0] = yaw + self.command.yaw_rate * t
            xref[k, 1] = 0.0
            xref[k, 2] = 0.0
            xref[k, 3] = self.world_position_desired[0] + self.command.vx * t
            xref[k, 4] = self.world_position_desired[1] + self.command.vy * t
            xref[k, 5] = self.command.body_height
            xref[k, 6] = 0.0
            xref[k, 7] = 0.0
            xref[k, 8] = self.command.yaw_rate
            xref[k, 9] = self.command.vx
            xref[k, 10] = self.command.vy
            xref[k, 11] = 0.0
        return xref

    def _swing_target_world(self, leg: str, base_pos: np.ndarray, base_quat: np.ndarray, base_vel_world: np.ndarray):
        Rwb = rotmat_from_quat_wxyz(base_quat)
        nominal_body = self.home_foot_positions_body[leg].copy()
        stance_time = self.scheduler.current_stance_time()

        # [추가됨] 시뮬레이션 경과 시간 계산 및 램프(Ramp) 계수 생성
        # 0.0초에서 2.0초 사이에 0.0 -> 1.0으로 서서히 증가합니다.
        t = self.sim_step_counter * self.ctrl_cfg.sim_dt
        ramp = min(1.0, max(0.0, t / 2.0))

        # 1. 어깨(고관절)의 현재 월드 좌표 계산
        p_shoulder_world = base_pos + Rwb @ nominal_body

        # 2. Raibert Heuristic 적용
        # 목표 속도(v_des)에도 ramp를 곱해 급발진을 방지합니다.
        v_des = np.array([self.command.vx, self.command.vy], dtype=float) * ramp
        v_err = base_vel_world[:2] - v_des  

        pw = p_shoulder_world.copy()
        
        # 보폭 계산: [대칭항] + [피드백항]
        step_x = 0.5 * stance_time * base_vel_world[0] + self.ctrl_cfg.raibert_kv * v_err[0]
        step_y = 0.5 * stance_time * base_vel_world[1] + self.ctrl_cfg.raibert_kv *0.5* v_err[1]
        max_step = 0.22  # Go2 로봇의 다리 길이를 고려한 최대 보폭 제한 (미터)
        step_x = float(np.clip(step_x, -max_step, max_step))
        step_y = float(np.clip(step_y, -max_step, max_step))
        
        # [추가됨] 계산된 보폭(step)에도 ramp를 곱해 초반에 발을 너무 멀리 뻗는 것을 방지합니다.
        pw[0] += step_x * ramp
        pw[1] += step_y * ramp
        
        # Yaw 회전 보상항에도 ramp 적용
        yaw_offset_body = np.array([0.0, self.ctrl_cfg.raibert_kyaw * self.command.yaw_rate, 0.0], dtype=float)
        pw += (Rwb @ yaw_offset_body) * ramp
        
        pw[2] = 0.0
        return pw

    def _stance_torque(self, leg: str, f_world_on_body: np.ndarray):
        J = foot_jacobian_world(self.model, self.data, self.mujoco, self.foot_body_ids[leg], self.leg_dof_indices[leg])
        tau = J.T @ (-np.asarray(f_world_on_body, dtype=float))
        qd_leg = np.array([self.data.qvel[idx] for idx in self.leg_dof_indices[leg]], dtype=float)
        return tau - self.ctrl_cfg.stance_joint_kd * qd_leg

    def _swing_torque(self, leg: str, p_des_w: np.ndarray, v_des_w: np.ndarray):
        J = foot_jacobian_world(self.model, self.data, self.mujoco, self.foot_body_ids[leg], self.leg_dof_indices[leg])
        p_now = self.data.xpos[self.foot_body_ids[leg]].copy()
        v_now = self._foot_vel_world(leg)
        f_des = self.ctrl_cfg.swing_kp @ (p_des_w - p_now) + self.ctrl_cfg.swing_kd @ (v_des_w - v_now)
        return J.T @ f_des

    def _apply_leg_torque(self, leg: str, tau: np.ndarray):
        tau = np.asarray(tau, dtype=float).copy()
        for j, aid in enumerate(self.leg_actuator_ids[leg]):
            low, high = self.model.actuator_ctrlrange[aid]
            self.data.ctrl[aid] = np.clip(tau[j], low, high)

    def control_step(self):
        t = self.sim_step_counter * self.ctrl_cfg.sim_dt

        base_pos, base_quat, ypr, omega_w, vel_w = self._base_state()
        foot_positions_world = self._foot_positions_world()

        if self.world_position_desired is None:
            self.world_position_desired = np.array([base_pos[0], base_pos[1], self.command.body_height], dtype=float)

        if self.scheduler.mode != "stand":
            self.world_position_desired[0] += self.command.vx * self.ctrl_cfg.sim_dt
            self.world_position_desired[1] += self.command.vy * self.ctrl_cfg.sim_dt

        contact_now = self.scheduler.contact_now(t)
        swing_phase = self.scheduler.swing_phase_now(t)

        if self.sim_step_counter % self.mpc_steps == 0:
            xref = self._build_reference(base_pos, ypr)
            contact_table = self.scheduler.mpc_contact_table(t)
            foot_array = np.stack([foot_positions_world[leg] for leg in LEG_ORDER], axis=0)
            result = self.mpc.solve(
                yaw=ypr[0],
                base_pos=base_pos,
                base_rpy_ypr=ypr,
                base_omega_world=omega_w,
                base_vel_world=vel_w,
                foot_pos_world=foot_array,
                x_ref=xref,
                contact_table=contact_table,
            )
            self.last_mpc_forces_world = result.forces_world

        swing_time = self.scheduler.current_swing_time()

        for i, leg in enumerate(LEG_ORDER):
            in_contact = bool(contact_now[i])

            if in_contact:
                self._apply_leg_torque(leg, self._stance_torque(leg, self.last_mpc_forces_world[i]))
                self.swing_active_prev[leg] = False
            else:
                if not self.swing_active_prev[leg]:
                    self.swing_traj[leg].set_start(foot_positions_world[leg])
                    self.swing_traj[leg].set_goal(self._swing_target_world(leg, base_pos, base_quat, vel_w))
                    self.swing_active_prev[leg] = True

                p_des, v_des = self.swing_traj[leg].sample(float(swing_phase[i]), swing_time)
                self._apply_leg_torque(leg, self._swing_torque(leg, p_des, v_des))

        self.sim_step_counter += 1

    def run(self, viewer=None, steps: int = 5000, push_force_max: float = 0.0, push_interval_sec: float = 3.0, push_duration_sec: float = 0.15, record_path: str = None):
        import time
        import mujoco  # 카메라 객체 생성을 위해 명시적 임포트
        
        # 녹화를 위한 설정
        frames = []
        fps = 30
        render_step = max(1, int(1.0 / (fps * self.ctrl_cfg.sim_dt)))
        renderer = None
        cam = None
        opt = None
        
        if record_path:
            import imageio
            print(f"🎥 녹화를 시작합니다... (목표 파일: {record_path})")
            # 640x480 해상도로 렌더러 초기화
            renderer = mujoco.Renderer(self.model, 480, 640)
            
            # [추가됨] 로봇(base_body_id)을 따라다니는 트래킹 카메라 생성
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, cam)
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = self.base_body_id  # 몸통을 추적 대상으로 설정
            cam.distance = 2.5     # 카메라와 로봇 사이의 거리 (원하면 조절 가능)
            cam.elevation = -20    # 카메라의 상하 각도 (약간 내려다보는 뷰)
            cam.azimuth = 130      # 카메라의 좌우 각도 (우측 대각선 뒤에서 보는 뷰)
            
            opt = mujoco.MjvOption()

        push_interval = max(1, int(push_interval_sec / self.ctrl_cfg.sim_dt))
        push_duration = max(1, int(push_duration_sec / self.ctrl_cfg.sim_dt))
        push_force = np.zeros(6, dtype=float)

        for step in range(steps):
            if push_force_max > 0.0:
                if step > 0 and step % push_interval == 0:
                    theta = np.random.uniform(0.0, 2.0 * np.pi)
                    fx = push_force_max * np.cos(theta)
                    fy = push_force_max * np.sin(theta)
                    push_force[:3] = [fx, fy, 0.0]
                    print(f"💥 [Push 발생!] X: {fx:.1f} N, Y: {fy:.1f} N")
                
                if step > 0 and (step % push_interval) < push_duration:
                    self.data.xfrc_applied[self.base_body_id, :] = push_force
                else:
                    self.data.xfrc_applied[self.base_body_id, :] = 0.0
            
            self.control_step()
            self.mujoco.mj_step(self.model, self.data)
            
            # [수정됨] 일반 update_scene 대신 트래킹 카메라(cam)를 적용하여 씬을 그립니다.
            if record_path and (step % render_step == 0):
                mujoco.mjv_updateScene(
                    self.model, self.data, opt, None, cam, 
                    mujoco.mjtCatBit.mjCAT_ALL, renderer.scene
                )
                frames.append(renderer.render())
            
            # 뷰어 동기화 (기존)
            if viewer is not None and step % 15 == 0:
                viewer.sync()
                if self.ctrl_cfg.viewer_realtime:
                    time.sleep(self.ctrl_cfg.sim_dt * 15)
                    
        # 시뮬레이션 종료 후 GIF 저장
        if record_path and len(frames) > 0:
            import imageio
            print(f"💾 GIF 저장 중... (총 {len(frames)} 프레임)")
            # loop=0은 GIF 무한 반복을 의미합니다.
            imageio.mimsave(record_path, frames, fps=fps, loop=0)
            print("✅ 저장 완료!")
            if renderer is not None:
                renderer.close()