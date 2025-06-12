
# 영상 최대길이 4초, 초기값 0으로 설정, xml파일에서 lap_start키프레임 joint값 안설정되어있음.
# rollout안보임, 에피소드 제대로 종료 x, video이름, 파일 관리 안됨 (덮어쓰기)
# 문제 다 해결
# /home/minjun/rl_ws/src/mujoco_rl_env/env/franka_reach_env.py
import gymnasium as gym  # gym → gymnasium으로 변경!
from gymnasium import spaces
import numpy as np
import mujoco

class FrankaReachEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    """
    Gym environment for moving a Franka Panda arm in MuJoCo
    from an initial joint configuration to a goal pose, with a reward
    that encourages smooth, accurate motion.
    """
    def __init__(self, xml_path, init_qpos=None, goal_qpos=None, render_mode=None):
        super().__init__()
        # Load model and data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Progress reward 계수
        self.kappa = 5.0

        # reset() 시 세팅될 값들
        self.init_pos_err = None
        self.prev_pos_err = None

        # # --- lap_start, lap_end site ID 조회 ---
        # self.start_site_id = mujoco.mj_name2id(
        #     self.model, mujoco.mjtObj.mjOBJ_SITE, "lap_start"
        # )
        # self.end_site_id = mujoco.mj_name2id(
        #     self.model, mujoco.mjtObj.mjOBJ_SITE, "lap_end"
        # )

        # render_mode설정!
        self.render_mode = render_mode

        # Initial joint positions
        if init_qpos is None:
            # default to zeros if no keyframes
            self.init_qpos = np.zeros(self.model.nq)
        else:
            # 입력받은 init_qpos가 nq보다 작으면 0으로 패딩
            if len(init_qpos) < self.model.nq:
                self.init_qpos = np.zeros(self.model.nq)
                self.init_qpos[:len(init_qpos)] = init_qpos
            else:
                self.init_qpos = np.array(init_qpos[:self.model.nq])

        # Fixed goal pose [x, y, z, qx, qy, qz, qw]
        # goal pose -> goal jointstate로 변경
        # self.goal_pose = np.array(goal_pose) if goal_pose is not None else None
        # Identify end-effector site (must exist in XML as <site name="ee_site" ...>)
        # self.ee_site_id = mujoco.mj_name2id(
        #     self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        # )

        if goal_qpos is None:
            # sample once at init
            low = self.model.jnt_range[:, 0].astype(np.float32)
            high = self.model.jnt_range[:, 1].astype(np.float32)
            self.goal_qpos = np.random.uniform(low, high)
        else:
            arr_g = np.array(goal_qpos, dtype=np.float32)
            self.goal_qpos = arr_g[:self.model.nq] if arr_g.size >= self.model.nq else np.pad(arr_g, (0, self.model.nq - arr_g.size))


        # Observation: [qpos (nq), qvel (nv), ee_pose (7), init_ee_pose (7), goal_pose (7)]
        # obs_dim = self.model.nq + self.model.nv + 7 + 7 + 7
        obs_dim = self.model.nq + self.model.nv + self.model.nq + self.model.nq
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: direct control of joint torques or velocities (size = nu)
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.model.nu,), dtype=np.float32
        )

        # Reward weights
        # self.alpha = 1.0    # position error weight
        # self.beta = 0.01    # velocity penalty
        # self.gamma = 0.001  # jerk penalty

        # ★ 논문 기반 가중치 설정 ★
        # - Joint 목표: 강하게 유도 (α = 5 ~ 10)  
        # - 속도 패널티: 작게 (γ = 0.001 ~ 0.01)  
        # - jerk 패널티: 매우 작게 (δ = 1e-4 ~ 1e-3)

        self.alpha = 50.0    # position error weight
        self.beta = 0.000    # velocity penalty
        self.gamma = 0.0000  # jerk penalty

        # Episode limits
        self.max_steps = 200 # trajectory 길이,         100step까지만 했기때문에 - 3.3초
        self.step_count = 0
        self.last_action = np.zeros(self.model.nu)

        # 렌더러 초기화
        self.viewer = None
        self.renderer = None

    def render(self):  # gymnasium은 render()에 mode 파라미터가 없음
        mode = self.render_mode  # render_mode 사용

        if mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()  # Sync viewer with data

        # rgb_array 모드: 픽셀 배열(프레임) 리턴
        elif mode == "rgb_array":
            # MuJoCo 렌더러 초기화
            if self.renderer is None:
                from mujoco import Renderer
                self.renderer = Renderer(self.model, 480, 480)
            
            # 중요: 매번 scene 업데이트!
            self.renderer.update_scene(self.data)
            pixels = self.renderer.render()
            return pixels  # (H, W, 3) uint8
        
        return None

    def reset(self, *, seed=None, options=None):  # gymnasium 시그니처
        super().reset(seed=seed)
        
        # Reset joint positions and velocities
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        # 초기/이전 오차 세팅
        self.init_pos_err = np.linalg.norm(self.init_qpos - self.goal_qpos)
        self.prev_pos_err = self.init_pos_err

        # --- XML 정의된 site 로부터 초기 EE pose 추출 ---
        # lap_start 위치/자세 → self.init_ee_pose 에 저장
        # start_pos = self.data.site_xpos[self.start_site_id].copy()
        # quat_buf  = np.zeros(4, dtype=np.float64)
        # mujoco.mju_mat2Quat(quat_buf, self.data.site_xmat[self.start_site_id])
        # self.init_ee_pose = np.concatenate([start_pos, quat_buf])

        # # --- goal_pose 설정 ---
        # if self.goal_pose is None:
        #     # lap_end 위치/자세 → self.goal_pose 에 저장
        #     end_pos   = self.data.site_xpos[self.end_site_id].copy()
        #     quat_buf2 = np.zeros(4, dtype=np.float64)
        #     mujoco.mju_mat2Quat(quat_buf2, self.data.site_xmat[self.end_site_id])
        #     self.goal_pose = np.concatenate([end_pos, quat_buf2])

        # reset counters
        self.step_count = 0
        self.last_action[:] = 0
        
        info = {"goal_qpos": self.goal_qpos.copy()}
        return self._get_obs(), info  # gymnasium은 (obs, info) 튜플 반환

    def _get_obs(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        # ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        # # Convert rotation matrix to quaternion for observation
        # mat = self.data.site_xmat[self.ee_site_id]
        # quat_buf = np.zeros(4, dtype=np.float64)
        # mujoco.mju_mat2Quat(quat_buf, mat)
        # ee_quat = quat_buf.copy()
        # return np.concatenate([
        #     qpos, qvel, ee_pos, ee_quat, self.init_ee_pose, self.goal_pose
        # ])
        return np.concatenate([qpos, qvel, self.goal_qpos, self.init_qpos])

    def step(self, action):
        # Clip and apply action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action

        # Step the simulation
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()

        # 1) Joint-space error (주목표) -  Compute joint-space distance
        pos_err = np.linalg.norm(self.data.qpos - self.goal_qpos)
        # 2) 속도 및 제어 변화 패널티
        vel = np.linalg.norm(self.data.qvel)
        jerk = np.linalg.norm(action - self.last_action)

        # Shaped reward to encourage movement
        reward = (
            - self.alpha * pos_err #pos_err_reward
            - self.beta * vel**2
            - self.gamma * jerk**2
        )

        # → 여기에 progress 보상 추가
        progress = (self.prev_pos_err - pos_err) / (self.init_pos_err + 1e-6)
        reward  += self.kappa * progress

        # 목표 도달 성공시 보너스
        required_pose_err = 0.05 # 0.05 기준
        if pos_err < required_pose_err: 
            reward += 50.0 #원래 10.0에서 수정
            print(f"🎉 Goal reached! pos_error: {pos_err:.3f}")

        # 다음 스텝을 위해 이전 오차 업데이트
        self.prev_pos_err = pos_err

        reward = float(np.clip(reward, -1.0, 1.0)) #clip도 -1.0, 1.0은 하는게 나은듯

        self.last_action = action.copy()
        self.step_count += 1

        # gymnasium은 terminated와 truncated를 구분
        terminated = bool(pos_err < required_pose_err)  # 목표 도달
        truncated = bool(self.step_count >= self.max_steps)  # 시간 초과

        info = {
            "pos_error": float(pos_err),
            "success": terminated
        }
        return obs, reward, terminated, truncated, info  # 5개 반환!
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        
        if self.renderer is not None:
            del self.renderer
            self.renderer = None
