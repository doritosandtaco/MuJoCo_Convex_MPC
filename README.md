# 🐕 Go2 Convex MPC in MuJoCo

A robust, real-time Convex Model Predictive Control (MPC) implementation for the Unitree Go2 quadruped robot, simulated in MuJoCo.

이 프로젝트는 Python과 MuJoCo 물리 엔진을 활용하여 Unitree Go2 4족 보행 로봇의 실시간 안정화 및 보행 제어를 구현한 리포지토리입니다. MIT Cheetah 3의 Convex MPC 구조를 기반으로 작성되었으며, OSQP 솔버를 통한 실시간 연산 최적화와 강력한 외란 회복(Push Recovery) 기능을 갖추고 있습니다.

## 🎥 Demonstrations

### 1. Forward Trotting
로봇이 안정적인 자세를 유지하며 전진하는 기본 Trot 보행 모드입니다.

![Trot Forward](trot_forward.gif)

### 2. Push Recovery (Disturbance Rejection)
보행 중 측면 및 전후방에서 강력한 무작위 외란(Push)이 발생했을 때, Raibert Heuristic과 MPC가 협력하여 발을 뻗어 중심을 되찾는 샌드백 테스트입니다.

![Push Recovery](trot_forward_with_push.gif)

<br>

## ✨ Key Features

* **Real-time Optimization (OSQP)**: 기존 범용 비선형 솔버 대신 고속 2차 계획법(QP) 솔버인 OSQP와 Euler 이산화 기법을 적용하여 MPC 연산을 수행합니다.
* **Robust Push Recovery**: 외란 발생 시 몸체의 실제 속도 오차를 반영하는 Raibert Heuristic 알고리즘을 통해 넘어지는 방향으로 스윙 발을 뻗어 밸런스를 스스로 회복합니다.
* **Hardware-Accurate Dynamics**: 로봇의 기하학적 중심이 아닌 실제 관성 중심(Center of Mass, `xipos`)을 기준으로 제어하여, 머리와 배터리로 인해 앞이 무거운 실제 Go2 하드웨어의 무게 배분을 완벽히 보상합니다.
* **Roll Divergence Prevention**: 강한 측면 충격 시 로봇의 다리가 찢어지며 제어력을 상실하는 현상(Kinematic Singularity)을 방지하기 위해 보폭 제한(Clipping) 및 스탠스 너비 강제 확장 로직이 적용되었습니다.

<br>

## ⚙️ Installation

Python 3.8 이상 환경을 권장합니다. 아래 명령어를 통해 필수 라이브러리를 설치해 주세요.

```bash
pip install numpy scipy osqp mujoco mujoco-viewer imageio
`````

## 🚀 Usage
기본적인 실행 파일은 run_go2_convex_mpc.py입니다. 다양한 Command Line 인자를 지원하여 시뮬레이션 환경을 쉽게 제어할 수 있습니다.

1. 기본 제자리걸음 (Trot in place)

```Bash
python run_go2_convex_mpc.py --mode trot
`````
2. 전진 보행 (Forward Trotting)
목표 속도(vx)를 0.4 m/s로 설정하여 전진합니다.

```Bash
python run_go2_convex_mpc.py --mode trot --vx 0.4
`````
3. 외란 회복 테스트 (Random Push Test)
3초 간격으로 무작위 방향에서 50N의 힘을 가하여 로봇의 밸런싱 능력을 테스트합니다.

```Bash
python run_go2_convex_mpc.py --mode trot --vx 0.4 --push-force 50 --push-interval 3.0 --push-duration 0.15
`````
