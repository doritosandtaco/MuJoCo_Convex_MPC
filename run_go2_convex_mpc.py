from __future__ import annotations

import argparse
import pathlib

import mujoco
import mujoco.viewer

from go2_convex_mpc import MPCConfig, ControlConfig, Go2ConvexMPCController
from go2_convex_mpc.controller import Command


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", type=str, default="./go2/scene.xml", help="Path to go2.xml")
    p.add_argument("--mode", type=str, default="trot", choices=["stand", "trot"])
    p.add_argument("--vx", type=float, default=0.5)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--yaw-rate", type=float, default=0.0)
    p.add_argument("--body-height", type=float, default=0.4)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--dt-mpc", type=float, default=0.02)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--mu", type=float, default=0.8)
    p.add_argument("--fz-max", type=float, default=660.0)
    p.add_argument("--push-force", type=float, default=0.0, help="밀기 힘의 최대치 (N). 0이면 끕니다.")
    p.add_argument("--push-interval", type=float, default=3.0, help="몇 초마다 밀 것인지 주기 (초)")
    p.add_argument("--push-duration", type=float, default=0.15, help="미는 힘이 지속되는 시간 (초)")
    p.add_argument("--record", type=str, default="", help="저장할 GIF 파일 이름 (예: trot.gif)")
    return p.parse_args()


def main():
    args = parse_args()
    xml_path = pathlib.Path(args.xml).expanduser().resolve()
    if not xml_path.exists():
        raise FileNotFoundError(xml_path)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    ctrl_cfg = ControlConfig()
    if model.opt.timestep > 0:
        ctrl_cfg.sim_dt = float(model.opt.timestep)
    ctrl_cfg.body_height = args.body_height

    mpc_cfg = MPCConfig(
        horizon=args.horizon,
        dt_mpc=args.dt_mpc,
        mu=args.mu,
        fz_max=args.fz_max,
    )

    cmd = Command(
        vx=args.vx,
        vy=args.vy,
        yaw_rate=args.yaw_rate,
        body_height=args.body_height,
    )

    controller = Go2ConvexMPCController(
        model=model,
        data=data,
        mujoco=mujoco,
        mpc_cfg=mpc_cfg,
        ctrl_cfg=ctrl_cfg,
        mode=args.mode,
        command=cmd,
    )
    controller.reset_pose()

    if args.headless or args.record:
        # 녹화 중일 때는 화면 렌더링 부하를 줄이기 위해 헤드리스(Headless) 모드로 강제 실행하는 것이 좋습니다.
        controller.run(
            viewer=None, 
            steps=args.steps, 
            push_force_max=args.push_force, 
            push_interval_sec=args.push_interval, 
            push_duration_sec=args.push_duration,
            record_path=args.record if args.record else None
        )
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            controller.run(
                viewer=viewer, 
                steps=args.steps, 
                push_force_max=args.push_force, 
                push_interval_sec=args.push_interval, 
                push_duration_sec=args.push_duration,
                record_path=None
            )

if __name__ == "__main__":
    main()
