
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
        # - EE 위치: 보조 보상 (β = 1 ~ 3) -> 제거
        # - 속도 패널티: 작게 (γ = 0.001 ~ 0.01)  
        # - jerk 패널티: 매우 작게 (δ = 1e-4 ~ 1e-3)

        self.alpha = 10.0    # position error weight
        self.beta = 0.01    # velocity penalty
        self.gamma = 0.0001  # jerk penalty

        # Episode limits
        self.max_steps = 200 # trajectory 길이, 100step까지만 했기때문에 - 3.3초
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


        # ▶ pos_err: 전체 관절(qpos) 평균 L2 거리
        # ▶ vel_pen: 관절 속도 제곱합 (energy cost)
        # ▶ act_pen: 제어 입력 변화(jerk) 제곱합
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
        if pos_err < required_pose_err:  # pos_err가 0.001로 수정 (더 합리적인 값)
            reward += 50.0 #원래 10.0
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

#------------------------------------------------------------------------------------------------------------------



# /home/minjun/rl_ws/src/mujoco_rl_env/env/franka_reach_env.py
# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import mujoco

# class FrankaReachEnv(gym.Env):
#     metadata = {"render_modes": ["human", "rgb_array"]}
#     """
#     Gym environment for moving a Franka Panda arm in MuJoCo
#     from an initial joint configuration to a goal configuration.
#     """
#     def __init__(self, xml_path, init_qpos=None, goal_qpos=None, render_mode=None):
#         super().__init__()
#         # Load model and data
#         self.model = mujoco.MjModel.from_xml_path(xml_path)
#         self.data = mujoco.MjData(self.model)

#         self.render_mode = render_mode

#         # Initial joint positions (전체 qpos)
#         if init_qpos is None:
#             self.init_qpos = np.zeros(self.model.nq)
#         else:
#             # 입력받은 init_qpos가 nq보다 작으면 0으로 패딩
#             if len(init_qpos) < self.model.nq:
#                 self.init_qpos = np.zeros(self.model.nq)
#                 self.init_qpos[:len(init_qpos)] = init_qpos
#             else:
#                 self.init_qpos = np.array(init_qpos[:self.model.nq])

#         # Goal joint positions (7 DOF arm only)
#         if goal_qpos is None:
#             # 기본값: 초기 위치와 동일
#             self.goal_qpos = self.init_qpos[:7].copy()
#         else:
#             self.goal_qpos = np.array(goal_qpos[:7])  # 첫 7개 joint만 사용

#         # Get end-effector body (last link in the arm)
#         self.ee_body_id = self.model.nbody - 1
        
#         # Observation: [qpos (nq), qvel (nv), goal_qpos (7)]
#         obs_dim = self.model.nq + self.model.nv + 7
#         self.observation_space = spaces.Box(
#             -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
#         )

#         # Action: joint torques
#         self.action_space = spaces.Box(
#             -1.0, 1.0, shape=(self.model.nu,), dtype=np.float32
#         )

#         # Reward weights
#         self.alpha = 10.0    # joint position error weight
#         self.beta = 0.1     # velocity penalty
#         self.gamma = 0.01   # jerk penalty

#         # Episode limits
#         self.max_steps = 300
#         self.step_count = 0
#         self.last_action = np.zeros(self.model.nu)

#         # Renderers
#         self.viewer = None
#         self.renderer = None

#         # For tracking initial and goal end-effector positions
#         self.init_ee_pos = None
#         self.goal_ee_pos = None
        
#         # Debug print frequency
#         self.episode_count = 0

#     def _compute_ee_pos_from_qpos(self, qpos):
#         """주어진 joint configuration에서 end-effector 위치 계산"""
#         # 새로운 MjData 인스턴스 생성하여 현재 상태를 보존
#         temp_data = mujoco.MjData(self.model)
#         temp_data.qpos[:] = qpos
#         mujoco.mj_forward(self.model, temp_data)
        
#         # End-effector 위치 가져오기
#         ee_pos = temp_data.xpos[self.ee_body_id].copy()
#         return ee_pos

#     def render(self):
#         mode = self.render_mode

#         if mode == "human":
#             if self.viewer is None:
#                 from mujoco import viewer
#                 self.viewer = viewer.launch_passive(self.model, self.data)
#             self.viewer.sync()

#         elif mode == "rgb_array":
#             if self.renderer is None:
#                 from mujoco import Renderer
#                 self.renderer = Renderer(self.model, 480, 480)
            
#             self.renderer.update_scene(self.data)
#             pixels = self.renderer.render()
#             return pixels
        
#         return None

#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)
        
#         # Reset to initial joint configuration
#         self.data.qpos[:] = self.init_qpos
#         self.data.qvel[:] = 0
#         mujoco.mj_forward(self.model, self.data)

#         # Compute initial and goal end-effector positions
#         self.init_ee_pos = self.data.xpos[self.ee_body_id].copy()
        
#         # Goal qpos로 goal EE position 계산
#         goal_qpos_full = self.init_qpos.copy()
#         goal_qpos_full[:7] = self.goal_qpos
#         self.goal_ee_pos = self._compute_ee_pos_from_qpos(goal_qpos_full)

#         self.step_count = 0
#         self.last_action[:] = 0
#         self.episode_count += 1
        
#         # Print debug info only every 10 episodes
#         if self.episode_count % 10 == 1:
#             print(f"\n🎯 Reset environment (Episode {self.episode_count}):")
#             print(f"   Initial arm joints: {self.init_qpos[:7]}")
#             print(f"   Goal arm joints: {self.goal_qpos}")
#             print(f"   Initial EE pos: {self.init_ee_pos}")
#             print(f"   Goal EE pos: {self.goal_ee_pos}")
#             print(f"   EE distance to goal: {np.linalg.norm(self.init_ee_pos - self.goal_ee_pos):.3f}")
#             print(f"   Joint distance to goal: {np.linalg.norm(self.init_qpos[:7] - self.goal_qpos):.3f}")
        
#         info = {}
#         return self._get_obs(), info

#     def _get_obs(self):
#         qpos = self.data.qpos.copy()
#         qvel = self.data.qvel.copy()
        
#         return np.concatenate([
#             qpos, qvel, self.goal_qpos
#         ])

#     def step(self, action):
#         # Clip and scale action
#         action = np.clip(action, self.action_space.low, self.action_space.high)
#         self.data.ctrl[:] = action * 5.0

#         # Step simulation
#         mujoco.mj_step(self.model, self.data)

#         obs = self._get_obs()

#         # Compute rewards
#         # 1. Joint position error (main objective)
#         joint_error = np.linalg.norm(self.data.qpos[:7] - self.goal_qpos)
        
#         # 2. End-effector position error (auxiliary)
#         ee_pos = self.data.xpos[self.ee_body_id]
#         ee_error = np.linalg.norm(ee_pos - self.goal_ee_pos)
        
#         # 3. Velocity and jerk penalties
#         vel = np.linalg.norm(self.data.qvel)
#         jerk = np.linalg.norm(action - self.last_action)

#         # Shaped reward
#         joint_reward = np.exp(-joint_error * 2.0) * 5.0  # Joint 정확도
#         ee_reward = np.exp(-ee_error * 3.0) * 3.0       # EE 위치 보조
        
#         # Progress reward
#         init_joint_error = np.linalg.norm(self.init_qpos[:7] - self.goal_qpos)
#         joint_progress = (init_joint_error - joint_error) / (init_joint_error + 1e-6)
#         progress_reward = joint_progress * 5.0

#         # Total reward
#         reward = (
#             joint_reward 
#             + ee_reward
#             + progress_reward
#             - self.beta * vel**2
#             - self.gamma * jerk**2
#         )

#         # Success bonus
#         success = joint_error < 0.1 and ee_error < 0.05  # Joint과 EE 모두 근접
#         if success:
#             reward += 100.0
#             if self.episode_count % 10 == 1:  # 10 에피소드마다만 출력
#                 print(f"🎉 Goal reached! Joint error: {joint_error:.3f}, EE error: {ee_error:.3f}")

#         # Clip reward
#         reward = float(np.clip(reward, -10.0, 110.0))

#         self.last_action = action.copy()
#         self.step_count += 1

#         # Episode termination
#         terminated = bool(success)
#         truncated = bool(self.step_count >= self.max_steps)
        
#         info = {
#             "joint_error": float(joint_error),
#             "ee_error": float(ee_error),
#             "success": success
#         }
        
#         return obs, reward, terminated, truncated, info
    
#     def close(self):
#         if self.viewer is not None:
#             self.viewer.close()
#             self.viewer = None
        
#         if self.renderer is not None:
#             del self.renderer
#             self.renderer = None

#------------------------------------------------------------------------------------------------------------------
# qpos기반으로 ee pose inverse kinematic 계산 -> 수정
# 밑에꺼 코드 오류 초기 ee와 나중 ee같다고함 forward kinematic잘 작동 x
# /home/minjun/rl_ws/src/mujoco_rl_env/env/franka_reach_env.py
# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import mujoco

# class FrankaReachEnv(gym.Env):
#     metadata = {"render_modes": ["human", "rgb_array"]}
#     """
#     Gym environment for moving a Franka Panda arm in MuJoCo
#     from an initial joint configuration to a goal configuration.
#     """
#     def __init__(self, xml_path, init_qpos=None, goal_qpos=None, render_mode=None):
#         super().__init__()
#         # Load model and data
#         self.model = mujoco.MjModel.from_xml_path(xml_path)
#         self.data = mujoco.MjData(self.model)

#         self.render_mode = render_mode

#         # Initial joint positions (전체 qpos)
#         if init_qpos is None:
#             self.init_qpos = np.zeros(self.model.nq)
#         else:
#             # 입력받은 init_qpos가 nq보다 작으면 0으로 패딩
#             if len(init_qpos) < self.model.nq:
#                 self.init_qpos = np.zeros(self.model.nq)
#                 self.init_qpos[:len(init_qpos)] = init_qpos
#             else:
#                 self.init_qpos = np.array(init_qpos[:self.model.nq])

#         # Goal joint positions (7 DOF arm only)
#         if goal_qpos is None:
#             # 기본값: 초기 위치와 동일
#             self.goal_qpos = self.init_qpos[:7].copy()
#         else:
#             self.goal_qpos = np.array(goal_qpos[:7])  # 첫 7개 joint만 사용

#         # Get end-effector body (last link in the arm)
#         self.ee_body_id = self.model.nbody - 1
        
#         # Observation: [qpos (nq), qvel (nv), goal_qpos (7)]
#         obs_dim = self.model.nq + self.model.nv + 7
#         self.observation_space = spaces.Box(
#             -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
#         )

#         # Action: joint torques
#         self.action_space = spaces.Box(
#             -1.0, 1.0, shape=(self.model.nu,), dtype=np.float32
#         )

#         # Reward weights
#         self.alpha = 10.0    # joint position error weight
#         self.beta = 0.1     # velocity penalty
#         self.gamma = 0.01   # jerk penalty

#         # Episode limits
#         self.max_steps = 300
#         self.step_count = 0
#         self.last_action = np.zeros(self.model.nu)

#         # Renderers
#         self.viewer = None
#         self.renderer = None

#         # For tracking initial and goal end-effector positions
#         self.init_ee_pos = None
#         self.goal_ee_pos = None

#     def _compute_ee_pos_from_qpos(self, qpos):
#         """주어진 joint configuration에서 end-effector 위치 계산"""
#         # 임시로 qpos 설정하여 forward kinematics 계산
#         old_qpos = self.data.qpos.copy()
#         self.data.qpos[:] = qpos
#         mujoco.mj_forward(self.model, self.data)
#         ee_pos = self.data.xpos[self.ee_body_id].copy()
#         # 원래 상태로 복원
#         self.data.qpos[:] = old_qpos
#         mujoco.mj_forward(self.model, self.data)
#         return ee_pos

#     def render(self):
#         mode = self.render_mode

#         if mode == "human":
#             if self.viewer is None:
#                 from mujoco import viewer
#                 self.viewer = viewer.launch_passive(self.model, self.data)
#             self.viewer.sync()

#         elif mode == "rgb_array":
#             if self.renderer is None:
#                 from mujoco import Renderer
#                 self.renderer = Renderer(self.model, 480, 480)
            
#             self.renderer.update_scene(self.data)
#             pixels = self.renderer.render()
#             return pixels
        
#         return None

#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)
        
#         # Reset to initial joint configuration
#         self.data.qpos[:] = self.init_qpos
#         self.data.qvel[:] = 0
#         mujoco.mj_forward(self.model, self.data)

#         # Compute initial and goal end-effector positions
#         self.init_ee_pos = self.data.xpos[self.ee_body_id].copy()
        
#         # Goal qpos로 goal EE position 계산
#         goal_qpos_full = self.init_qpos.copy()
#         goal_qpos_full[:7] = self.goal_qpos
#         self.goal_ee_pos = self._compute_ee_pos_from_qpos(goal_qpos_full)

#         self.step_count = 0
#         self.last_action[:] = 0
        
#         # Print debug info
#         print(f"\n🎯 Reset environment:")
#         print(f"   Initial arm joints: {self.init_qpos[:7]}")
#         print(f"   Goal arm joints: {self.goal_qpos}")
#         print(f"   Initial EE pos: {self.init_ee_pos}")
#         print(f"   Goal EE pos: {self.goal_ee_pos}")
#         print(f"   EE distance to goal: {np.linalg.norm(self.init_ee_pos - self.goal_ee_pos):.3f}")
        
#         info = {}
#         return self._get_obs(), info

#     def _get_obs(self):
#         qpos = self.data.qpos.copy()
#         qvel = self.data.qvel.copy()
        
#         return np.concatenate([
#             qpos, qvel, self.goal_qpos
#         ])

#     def step(self, action):
#         # Clip and scale action
#         action = np.clip(action, self.action_space.low, self.action_space.high)
#         self.data.ctrl[:] = action * 5.0

#         # Step simulation
#         mujoco.mj_step(self.model, self.data)

#         obs = self._get_obs()

#         # Compute rewards
#         # 1. Joint position error (main objective)
#         joint_error = np.linalg.norm(self.data.qpos[:7] - self.goal_qpos)
        
#         # 2. End-effector position error (auxiliary)
#         ee_pos = self.data.xpos[self.ee_body_id]
#         ee_error = np.linalg.norm(ee_pos - self.goal_ee_pos)
        
#         # 3. Velocity and jerk penalties
#         vel = np.linalg.norm(self.data.qvel)
#         jerk = np.linalg.norm(action - self.last_action)

#         # Shaped reward
#         joint_reward = np.exp(-joint_error * 2.0) * 5.0  # Joint 정확도
#         ee_reward = np.exp(-ee_error * 3.0) * 3.0       # EE 위치 보조
        
#         # Progress reward
#         init_joint_error = np.linalg.norm(self.init_qpos[:7] - self.goal_qpos)
#         joint_progress = (init_joint_error - joint_error) / (init_joint_error + 1e-6)
#         progress_reward = joint_progress * 5.0

#         # Total reward
#         reward = (
#             joint_reward 
#             + ee_reward
#             + progress_reward
#             - self.beta * vel**2
#             - self.gamma * jerk**2
#         )

#         # Success bonus
#         success = joint_error < 0.1 and ee_error < 0.05  # Joint과 EE 모두 근접
#         if success:
#             reward += 100.0
#             print(f"🎉 Goal reached! Joint error: {joint_error:.3f}, EE error: {ee_error:.3f}")

#         # Clip reward
#         reward = float(np.clip(reward, -10.0, 110.0))

#         self.last_action = action.copy()
#         self.step_count += 1

#         # Episode termination
#         terminated = bool(success)
#         truncated = bool(self.step_count >= self.max_steps)
        
#         info = {
#             "joint_error": float(joint_error),
#             "ee_error": float(ee_error),
#             "success": success
#         }
        
#         return obs, reward, terminated, truncated, info
    
#     def close(self):
#         if self.viewer is not None:
#             self.viewer.close()
#             self.viewer = None
        
#         if self.renderer is not None:
#             del self.renderer
#             self.renderer = None


#------------------------------------------------------------------------------------------------------------------
# 20250608 1717 수정완료
# 일단 claude 식으로 변경 후, 초기 qpos찾자 - mujoco에서!
# # /home/minjun/rl_ws/src/mujoco_rl_env/env/franka_reach_env.py
# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import mujoco

# class FrankaReachEnv(gym.Env):
#     metadata = {"render_modes": ["human", "rgb_array"]}
#     """
#     Gym environment for moving a Franka Panda arm in MuJoCo
#     from lap_start to lap_end, with proper initialization.
#     """
#     def __init__(self, xml_path, init_qpos=None, goal_pose=None, render_mode=None):
#         super().__init__()
#         # Load model and data
#         self.model = mujoco.MjModel.from_xml_path(xml_path)
#         self.data = mujoco.MjData(self.model)

#         # Get site IDs
#         self.start_site_id = mujoco.mj_name2id(
#             self.model, mujoco.mjtObj.mjOBJ_SITE, "lap_start"
#         )
#         self.end_site_id = mujoco.mj_name2id(
#             self.model, mujoco.mjtObj.mjOBJ_SITE, "lap_end"
#         )

#         self.render_mode = render_mode

#         # Get keyframe qpos for lap_start if available and init_qpos not provided
#         if init_qpos is None:
#             # Try to get lap_start keyframe
#             try:
#                 keyframe_id = mujoco.mj_name2id(
#                     self.model, mujoco.mjtObj.mjOBJ_KEY, "lap_start"
#                 )
#                 if keyframe_id >= 0:
#                     # Use lap_start keyframe joint positions
#                     self.init_qpos = self.model.key_qpos[keyframe_id].copy()
#                     print(f"✅ Using lap_start keyframe for initial pose")
#                 else:
#                     self.init_qpos = np.zeros(self.model.nq)
#                     print(f"⚠️  No lap_start keyframe found, using zeros")
#             except:
#                 self.init_qpos = np.zeros(self.model.nq)
#                 print(f"⚠️  Could not find lap_start keyframe, using zeros")
#         else:
#             # Use provided init_qpos
#             if len(init_qpos) < self.model.nq:
#                 self.init_qpos = np.zeros(self.model.nq)
#                 self.init_qpos[:len(init_qpos)] = init_qpos
#             else:
#                 self.init_qpos = np.array(init_qpos[:self.model.nq])

#         # Goal pose
#         self.goal_pose = np.array(goal_pose) if goal_pose is not None else None

#         # For tracking end-effector, we'll use the last link instead of ee_site
#         # Assuming the last body is the end-effector
#         self.ee_body_id = self.model.nbody - 1  # Last body in the kinematic chain
        
#         # Observation: [qpos (nq), qvel (nv), ee_pose (7), init_ee_pose (7), goal_pose (7)]
#         obs_dim = self.model.nq + self.model.nv + 7 + 7 + 7
#         self.observation_space = spaces.Box(
#             -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
#         )

#         # Action: joint torques
#         self.action_space = spaces.Box(
#             -1.0, 1.0, shape=(self.model.nu,), dtype=np.float32
#         )

#         # Reward weights
#         self.alpha = 10.0    # position error weight (increased)
#         self.beta = 0.1     # velocity penalty (increased)
#         self.gamma = 0.01   # jerk penalty (increased)

#         # Episode limits
#         self.max_steps = 300  # Increased from 100 to 300 for longer episodes
#         self.step_count = 0
#         self.last_action = np.zeros(self.model.nu)

#         # Renderers
#         self.viewer = None
#         self.renderer = None

#     def _get_ee_pose(self):
#         """Get end-effector pose using body position instead of site"""
#         # Get body position
#         ee_pos = self.data.xpos[self.ee_body_id].copy()
        
#         # Get body orientation matrix and convert to quaternion
#         mat = self.data.xmat[self.ee_body_id].reshape(3, 3)
#         quat_buf = np.zeros(4, dtype=np.float64)
#         mujoco.mju_mat2Quat(quat_buf, mat)
        
#         return np.concatenate([ee_pos, quat_buf])

#     def render(self):
#         mode = self.render_mode

#         if mode == "human":
#             if self.viewer is None:
#                 from mujoco import viewer
#                 self.viewer = viewer.launch_passive(self.model, self.data)
#             self.viewer.sync()

#         elif mode == "rgb_array":
#             if self.renderer is None:
#                 from mujoco import Renderer
#                 self.renderer = Renderer(self.model, 480, 480)
            
#             self.renderer.update_scene(self.data)
#             pixels = self.renderer.render()
#             return pixels
        
#         return None

#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)
        
#         # Reset joint positions to lap_start configuration
#         self.data.qpos[:] = self.init_qpos
#         self.data.qvel[:] = 0
#         mujoco.mj_forward(self.model, self.data)

#         # Get initial EE pose from lap_start site
#         start_pos = self.data.site_xpos[self.start_site_id].copy()
#         quat_buf = np.zeros(4, dtype=np.float64)
#         mujoco.mju_mat2Quat(quat_buf, self.data.site_xmat[self.start_site_id])
#         self.init_ee_pose = np.concatenate([start_pos, quat_buf])

#         # Set goal pose from lap_end site
#         if self.goal_pose is None:
#             end_pos = self.data.site_xpos[self.end_site_id].copy()
#             quat_buf2 = np.zeros(4, dtype=np.float64)
#             mujoco.mju_mat2Quat(quat_buf2, self.data.site_xmat[self.end_site_id])
#             self.goal_pose = np.concatenate([end_pos, quat_buf2])

#         self.step_count = 0
#         self.last_action[:] = 0
        
#         # Print debug info
#         print(f"🎯 Initial EE pos: {self.init_ee_pose[:3]}")
#         print(f"🎯 Goal pos: {self.goal_pose[:3]}")
#         print(f"🎯 Distance to goal: {np.linalg.norm(self.init_ee_pose[:3] - self.goal_pose[:3]):.3f}")
        
#         info = {}
#         return self._get_obs(), info

#     def _get_obs(self):
#         qpos = self.data.qpos.copy()
#         qvel = self.data.qvel.copy()
#         ee_pose = self._get_ee_pose()
        
#         return np.concatenate([
#             qpos, qvel, ee_pose, self.init_ee_pose, self.goal_pose
#         ])

#     def step(self, action):
#         # Clip and scale action
#         action = np.clip(action, self.action_space.low, self.action_space.high)
#         # Scale up the action for more aggressive movement
#         self.data.ctrl[:] = action * 5.0  # Increased action scale

#         # Step simulation
#         mujoco.mj_step(self.model, self.data)

#         obs = self._get_obs()

#         # Compute reward
#         ee_pose = self._get_ee_pose()
#         ee_pos = ee_pose[:3]
#         goal_pos = self.goal_pose[:3]
        
#         # Distance to goal
#         dist = np.linalg.norm(ee_pos - goal_pos)
        
#         # Velocity and jerk penalties
#         vel = np.linalg.norm(self.data.qvel)
#         jerk = np.linalg.norm(action - self.last_action)

#         # Shaped reward to encourage movement
#         dist_reward = np.exp(-dist * 5.0)  # Exponential distance reward
        
#         # Progress reward - reward getting closer to goal
#         prev_dist = np.linalg.norm(self.init_ee_pose[:3] - goal_pos)
#         progress = (prev_dist - dist) / prev_dist
#         progress_reward = progress * 10.0

#         # Total reward
#         reward = (
#             dist_reward * self.alpha
#             + progress_reward
#             - self.beta * vel**2
#             - self.gamma * jerk**2
#         )

#         # Success bonus
#         if dist < 0.02:  # 2cm threshold
#             reward += 100.0
#             print(f"🎉 Goal reached! Distance: {dist:.3f}")

#         # Clip reward
#         reward = float(np.clip(reward, -10.0, 110.0))

#         self.last_action = action.copy()
#         self.step_count += 1

#         # Episode termination
#         terminated = bool(dist < 0.02)  # Success
#         truncated = bool(self.step_count >= self.max_steps)  # Timeout
        
#         info = {
#             "distance": float(dist),
#             "success": terminated,
#             "progress": float(progress)
#         }
        
#         return obs, reward, terminated, truncated, info
    
#     def close(self):
#         if self.viewer is not None:
#             self.viewer.close()
#             self.viewer = None
        
#         if self.renderer is not None:
#             del self.renderer
#             self.renderer = None





#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------
# 2025.06.08 13:01
# import gym
# from gym import spaces
# import numpy as np
# import mujoco

# class FrankaReachEnv(gym.Env):
#     metadata = {"render_modes": ["human", "rgb_array"]}  # ← 추가
#     """
#     Gym environment for moving a Franka Panda arm in MuJoCo
#     from an initial joint configuration to a goal pose, with a reward
#     that encourages smooth, accurate motion.
#     """
#     def __init__(self, xml_path, init_qpos=None, goal_pose=None, render_mode=None):
#         super().__init__()
#         # Load model and data
#         self.model = mujoco.MjModel.from_xml_path(xml_path)
#         self.data = mujoco.MjData(self.model)

#         # --- lap_start, lap_end site ID 조회 ---
#         #   XML 상 <site name="lap_start"/> 와 <site name="lap_end"/> 의 ID
#         self.start_site_id = mujoco.mj_name2id(
#             self.model, mujoco.mjtObj.mjOBJ_SITE, "lap_start"
#         )
#         self.end_site_id = mujoco.mj_name2id(
#             self.model, mujoco.mjtObj.mjOBJ_SITE, "lap_end"
#         )

#         # render_mode설정!
#         self.render_mode = render_mode

#         # Initial joint positions
#         if init_qpos is None:
#             # default to zeros if no keyframes
#             self.init_qpos = np.zeros(self.model.nq)
#         else:
#             self.init_qpos = np.array(init_qpos)

#         # Fixed goal pose [x, y, z, qx, qy, qz, qw]
#         self.goal_pose = np.array(goal_pose) if goal_pose is not None else None

#         # Identify end-effector site (must exist in XML as <site name="ee_site" ...>)
#         self.ee_site_id = mujoco.mj_name2id(
#             self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
#         )

#         # Observation: [qpos (nq), qvel (nv), ee_pose (7), goal_pose (7)]
#         obs_dim = self.model.nq + self.model.nv + 7 + 7 + 7
#         self.observation_space = spaces.Box(
#             -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
#         )

#         # Action: direct control of joint torques or velocities (size = nu)
#         self.action_space = spaces.Box(
#             -1.0, 1.0, shape=(self.model.nu,), dtype=np.float32
#         )

#         # Reward weights
#         # first reward: 발산
#         self.alpha = 1.0    # position error weight
#         self.beta = 0.01    # velocity penalty
#         self.gamma = 0.001  # jerk penalty

#         # Episode limits
#         self.max_steps = 100 #100에서 바꿈 -> 100했을 시 245iteation정도
#         self.step_count = 0
#         self.last_action = np.zeros(self.model.nu)

#         #렌더러 초기화
#         self.viewer = None
#         self.renderer = None

#     def render(self, mode=None):
#         # human 모드: 창 띄워서 실시간 보기
#         if mode is None:
#             mode = self.render_mode

#         if mode == "human":
#             if self.viewer is None:
#                 from mujoco import viewer
#                 self.viewer = viewer.launch_passive(self.model, self.data)
#             self.viewer.sync()  # Sync viewer with data
            
#             '''
#             if not hasattr(self, "viewer"):
#                 from mujoco.viewer import MjViewer
#                 self.viewer = MjViewer(self.model)
#             self.viewer.render()
#             '''

#         # rgb_array 모드: 픽셀 배열(프레임) 리턴
#         elif mode == "rgb_array":
#             # MuJoCo 오프스크린 렌더 컨텍스트 생성
#             if self.viewer is None:
#                 from mujoco import Renderer
#                 self.renderer = Renderer(self.model, 480, 480)
            
#             self.renderer.update_scene(self.data)
#             return self.renderer.render()  # (H, W, 3) uint8
#             '''
#                 self.viewer = viewer.launch_passive(self.model, self.data)
#             from mujoco.viewer import OffscreenRenderer
#             if not hasattr(self, "renderer"):
#                 # (model, width, height)
#                 self.renderer = OffscreenRenderer(self.model, 480, 480)
#             # 픽셀 버퍼 읽어오기 (H, W, 3), uint8
#             img = self.renderer.render(self.data)
#             # MuJoCo는 이미지가 뒤집혀 나옵니다. 위아래 뒤집어서 반환
#             return img
#             '''
#         else:
#             if mode is None:
#                 raise ValueError(f"Unknown render_mode: {mode}")


#     def reset(self):
#         # Reset joint positions and velocities
#         self.data.qpos[:] = self.init_qpos
#         self.data.qvel[:] = 0
#         mujoco.mj_forward(self.model, self.data)

#         # --- XML 정의된 site 로부터 초기 EE pose 추출 ---
#         # lap_start 위치/자세 → self.init_ee_pose 에 저장
#         start_pos = self.data.site_xpos[self.start_site_id].copy()
#         quat_buf  = np.zeros(4, dtype=np.float64)
#         mujoco.mju_mat2Quat(quat_buf, self.data.site_xmat[self.start_site_id])
#         self.init_ee_pose = np.concatenate([start_pos, quat_buf])

#         # --- XML 정의된 site 로부터 목표 pose 설정 ---
#         # lap_end 위치/자세 → self.goal_pose 에 저장
#         end_pos   = self.data.site_xpos[self.end_site_id].copy()
#         quat_buf2 = np.zeros(4, dtype=np.float64)
#         mujoco.mju_mat2Quat(quat_buf2, self.data.site_xmat[self.end_site_id])
#         self.goal_pose    = np.concatenate([end_pos, quat_buf2])

#         # # Sample a random goal if none provided
#         # if self.goal_pose is None:
#         #     # End-effector position
#         #     ee_pos = self.data.site_xpos[self.ee_site_id].copy()
#         #     # Convert rotation matrix to quaternion
#         #     mat = self.data.site_xmat[self.ee_site_id]
#         #     quat_buf = np.zeros(4, dtype=np.float64)
#         #     mujoco.mju_mat2Quat(quat_buf, mat)
#         #     ee_quat = quat_buf.copy()
#         #     pos = ee_pos + np.random.uniform(-0.1, 0.1, size=3)
#         #     quat = ee_quat
#         #     self.goal_pose = np.concatenate([pos, quat])

#         self.step_count = 0
#         self.last_action[:] = 0
#         return self._get_obs()

#     def _get_obs(self):
#         qpos = self.data.qpos.copy()
#         qvel = self.data.qvel.copy()
#         ee_pos = self.data.site_xpos[self.ee_site_id].copy()
#         # Convert rotation matrix to quaternion for observation
#         mat = self.data.site_xmat[self.ee_site_id]
#         quat_buf = np.zeros(4, dtype=np.float64)
#         mujoco.mju_mat2Quat(quat_buf, mat)
#         ee_quat = quat_buf.copy()
#         return np.concatenate([
#             qpos, qvel, ee_pos, ee_quat, self.init_ee_pose, self.goal_pose
#         ])

#     def step(self, action):
#         # Clip and apply action
#         action = np.clip(action, self.action_space.low, self.action_space.high)
#         self.data.ctrl[:] = action

#         # Step the simulation
#         mujoco.mj_step(self.model, self.data)

#         obs = self._get_obs()

#         # Compute reward
#         ee_pos = self.data.site_xpos[self.ee_site_id]
#         goal_pos = self.goal_pose[:3]
#         dist = np.linalg.norm(ee_pos - goal_pos)
#         vel = np.linalg.norm(self.data.qvel)
#         jerk = np.linalg.norm(action - self.last_action)
#         ''' 처음 reward -> RMSE
#         reward = - (
#             self.alpha * dist**2
#             + self.beta * vel**2
#             + self.gamma * jerk**2
#         )'''
#         # 2번째 reward -> clip reward
#         raw = - (
#             self.alpha * dist**2
#             + self.beta * vel**2
#             + self.gamma * jerk**2
#         )

#         #3번째 추가 - 목표 도달 성공시 +1보너스
#         if dist < 10.0:
#             raw += 100.0
#         # 1.0 보너스는 목표 도달시만 주고, 그 외엔 -1.0 ~ 1.0 사이로 보상
#         reward = float(np.clip(raw, -1.0, 1.0))
#         # clip reward to [-1, 1] range

#         self.last_action = action.copy()
#         self.step_count += 1

#         done = bool(dist < 0.01 or self.step_count >= self.max_steps)
#         info = {"distance": float(dist)}
#         return obs, reward, done, info
#         #return super().close()
    
#     def close(self):
#         if self.viewer is not None:
#             self.viewer.close()
#             self.viewer = None
        
#         if self.renderer is not None:
#             self.renderer = None




#------------------------------------------------------------------------------------------------------------------
''' 
import gym
from gym import spaces
import numpy as np
import mujoco


class FrankaReachEnv(gym.Env):
    """Reach a target pose with the Franka Panda arm in MuJoCo."""
    metadata = {"render_modes": ["human", "rgb_array"]}

    # ------------------------------------------------------------------ #
    # 초기화
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        xml_path: str,
        init_qpos: np.ndarray | None = None,
        goal_pose: np.ndarray | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()

        # MuJoCo 모델·데이터
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        # 초기 관절각 / 목표 포즈
        self.init_qpos = np.zeros(self.model.nq) if init_qpos is None else np.array(init_qpos)
        self.goal_pose = None if goal_pose is None else np.array(goal_pose)

        # EE site id
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

        # 관측/행동 공간
        obs_dim = self.model.nq + self.model.nv + 7 + 7      # qpos+qvel+ee(7)+goal(7)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)

        # 보상 계수
        self.alpha, self.beta, self.gamma = 1.0, 0.01, 0.001

        # 에피소드 설정
        self.max_steps   = 100
        self.step_count  = 0
        self.last_action = np.zeros(self.model.nu)

        # --- 렌더 관련 ---
        self.render_mode = render_mode          # 사용자가 넘긴 모드 저장
        self._viewer     = None                 # human 모드
        self._renderer   = None                 # rgb_array 모드(지연 초기화)

    # ------------------------------------------------------------------ #
    # 렌더링
    # ------------------------------------------------------------------ #
    def render(self, mode: str | None = None):
        mode = mode or self.render_mode or "rgb_array"   # default = rgb_array

        if mode == "human":
            if self._viewer is None:
                from mujoco.viewer import Viewer        # MuJoCo ≥ 3.1
                self._viewer = Viewer(self.model, self.data)
            self._viewer.render()
            return

        elif mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 480, 480)
            self._renderer.update_scene(self.data)
            img = self._renderer.render()               # (H,W,3) uint8
            return img

        else:
            raise ValueError(f"Unknown render_mode: {mode}")

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        # 무작위 목표
        if self.goal_pose is None:
            ee_pos = self.data.site_xpos[self.ee_site_id]
            quat   = np.zeros(4); mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.ee_site_id])
            pos    = ee_pos + self.np_random.uniform(-0.1, 0.1, size=3)
            self.goal_pose = np.concatenate([pos, quat])

        self.step_count  = 0
        self.last_action = np.zeros_like(self.last_action)

        return self._get_obs()

    def step(self, action):
        # 액션 적용
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        # 관측
        obs = self._get_obs()

        # 보상
        ee_pos   = self.data.site_xpos[self.ee_site_id]
        goal_pos = self.goal_pose[:3]
        dist  = np.linalg.norm(ee_pos - goal_pos)
        vel   = np.linalg.norm(self.data.qvel)
        jerk  = np.linalg.norm(action - self.last_action)

        raw_r = -(self.alpha*dist + self.beta*vel + self.gamma*jerk)
        if dist < 0.01:
            raw_r += 1.0                             # 목표 도달 보너스
        reward = float(np.clip(raw_r, -1.0, 1.0))

        # 상태 업데이트
        self.last_action = action.copy()
        self.step_count += 1
        done = dist < 0.01 or self.step_count >= self.max_steps
        info = {}
        return obs, reward, done, info

    # ------------------------------------------------------------------ #
    # 헬퍼
    # ------------------------------------------------------------------ #
    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel

        ee_pos = self.data.site_xpos[self.ee_site_id]
        quat   = np.zeros(4); mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.ee_site_id])

        return np.concatenate([qpos, qvel, ee_pos, quat, self.goal_pose])
'''