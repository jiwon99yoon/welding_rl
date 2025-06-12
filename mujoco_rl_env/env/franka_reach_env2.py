# /home/minjun/rl_ws/src/mujoco_rl_env/mujoco_rl_env/env/franka_reach_env2.py
import numpy as np
import mujoco
import mujoco.viewer
from gymnasium import spaces
from gymnasium.core import Env
from typing import Optional, Tuple, Union, List
import time

class FrankaReachEnv(Env):
    """
    Franka 로봇이 웨이포인트를 순차적으로 거쳐 목표 위치에 도달하는 강화학습 환경
    Domain Randomization 및 장애물 회피 기능 포함
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(
        self,
        xml_path: str,
        init_qpos: Optional[Union[List[float], np.ndarray]] = None,
        goal_qpos: Optional[Union[List[float], np.ndarray]] = None,
        render_mode: Optional[str] = None,
        enable_randomization: bool = True,  # Domain Randomization 활성화
        randomization_config: Optional[dict] = None  # 무작위화 설정
    ):
        super().__init__()
        
        # MuJoCo 모델 로드
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Domain Randomization 설정
        self.enable_randomization = enable_randomization
        self.randomization_config = randomization_config or self._get_default_randomization_config()
        
        # 로봇 관절 정보
        self.n_joints = 7  # Franka의 7개 관절
        self.n_fingers = 2  # 그리퍼 손가락
        
        # 웨이포인트 설정
        self.waypoint_names = ["lap_start", "lap_waypoint1", "lap_waypoint2", "lap_waypoint3", "lap_end"]
        self.current_waypoint_idx = 0
        self.waypoint_reached_threshold = 0.05  # 5cm
        
        # 초기 및 목표 위치 설정
        self._setup_initial_and_goal_positions(init_qpos, goal_qpos)
        
        # 렌더링 설정
        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None
        
        # 관찰 및 행동 공간 정의
        self._setup_spaces()
        
        # 무작위화를 위한 body/site ID 저장
        self._setup_randomization_ids()
        
        # 충돌 감지를 위한 geom ID 저장
        self._setup_collision_detection()
        
        # 에피소드 정보
        self.step_count = 0
        self.max_episode_steps = 500
        self.collision_penalty = 0
        
    def _get_default_randomization_config(self):
        """기본 무작위화 설정"""
        return {
            'table_position': {
                'x_range': (0.45, 0.55),
                'y_range': (-0.05, 0.05),
                'z_range': (0.43, 0.47)
            },
            'table_rotation': {
                'z_range': (-0.2, 0.2)  # radians
            },
            'lap_joint_angle': {
                'range': (-0.1, 0.1)  # radians
            },
            'obstacle_position': {
                'range': 0.05  # 장애물 위치 변동 범위
            },
            'worker_position': {
                'x_range': (-0.1, 0.1),
                'y_range': (-0.3, -0.2)
            }
        }
    
    def _setup_collision_detection(self):
        """충돌 감지를 위한 geom ID 설정"""
        self.robot_geom_ids = []
        self.obstacle_geom_ids = []
        
        # 로봇 geom ID 수집
        robot_bodies = ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 
                       'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7', 'panda_hand']
        for body_name in robot_bodies:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id != -1:
                # 해당 body의 모든 geom 찾기
                for i in range(self.model.ngeom):
                    if self.model.geom_bodyid[i] == body_id:
                        self.robot_geom_ids.append(i)
        
        # 장애물 geom ID 수집
        obstacle_names = ['obstacle_sphere', 'finger1', 'static_clamp1', 'clamp_arm1', 
                         'tool_obstacle', 'tool_tip', 'worker_torso', 'worker_head', 
                         'worker_left_arm', 'worker_right_arm']
        for geom_name in obstacle_names:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id != -1:
                self.obstacle_geom_ids.append(geom_id)
    
    def _setup_randomization_ids(self):
        """무작위화에 필요한 body/joint ID 저장"""
        try:
            # Body IDs
            self.table_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "table")
            self.lap_base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "lap_base")
            self.fillet_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "fillet_joint_base")
            self.curved_pipe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "curved_pipe")
            self.moving_obstacle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moving_obstacle")
            self.worker_torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "worker_torso")
            
            # Joint IDs for moving obstacle
            self.obstacle_joint_ids = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_x"),
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_y"),
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_z")
            ]
            
            # 원래 상대 위치 저장
            self.original_table_pos = self.model.body_pos[self.table_id].copy()
            self.relative_positions = {}
            for name, body_id in [
                ('lap_base', self.lap_base_id),
                ('fillet_joint', self.fillet_joint_id),
                ('curved_pipe', self.curved_pipe_id),
                ('moving_obstacle', self.moving_obstacle_id),
                ('worker_torso', self.worker_torso_id)
            ]:
                if body_id != -1:
                    self.relative_positions[name] = self.model.body_pos[body_id] - self.original_table_pos
                    
        except Exception as e:
            print(f"Warning: Some randomization bodies not found: {e}")
            self.enable_randomization = False
    
    def _setup_initial_and_goal_positions(self, init_qpos, goal_qpos):
        """초기 및 목표 위치 설정"""
        # 기본 joint 위치 정의 (알려진 값들)
        default_init_qpos = np.array([0.0555323, -0.519356, -0.522075, -2.239023, -0.030734, 2.252302, 0.329849])
        default_goal_qpos = np.array([-0.0214929, -0.7660162, -0.3266901, -2.4520896, -0.0856205, 2.2117625, 0.4996375])
        
        # 사용자 제공 위치 또는 기본값 사용
        self.init_qpos = init_qpos if init_qpos is not None else default_init_qpos
        self.goal_qpos = goal_qpos if goal_qpos is not None else default_goal_qpos
        
        # 배열로 변환 및 크기 확인
        self.init_qpos = np.array(self.init_qpos)
        self.goal_qpos = np.array(self.goal_qpos)
        
        # 7개 관절에 맞게 크기 조정
        if len(self.init_qpos) > self.n_joints:
            self.init_qpos = self.init_qpos[:self.n_joints]
        if len(self.goal_qpos) > self.n_joints:
            self.goal_qpos = self.goal_qpos[:self.n_joints]
        
    def _setup_spaces(self):
        """관찰 및 행동 공간 정의"""
        # 관찰 공간: [로봇 관절 위치(7), 로봇 관절 속도(7), EE 위치(3), 현재 목표 위치(3), 웨이포인트 진행도(1), 충돌 정보(1)]
        obs_dim = self.n_joints * 2 + 3 + 3 + 1 + 1  # 22차원
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # 행동 공간: 7개 관절의 속도 제어
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32
        )
    
    def _randomize_environment(self):
        """환경 무작위화 수행"""
        if not self.enable_randomization:
            return
            
        config = self.randomization_config
        
        # 1. 테이블 위치 무작위화
        if 'table_position' in config:
            table_config = config['table_position']
            new_table_pos = np.array([
                np.random.uniform(*table_config['x_range']),
                np.random.uniform(*table_config['y_range']),
                np.random.uniform(*table_config['z_range'])
            ])
            
            # 테이블과 관련 오브젝트 이동
            self.model.body_pos[self.table_id] = new_table_pos
            for name, rel_pos in self.relative_positions.items():
                body_id = getattr(self, f"{name}_id", -1)
                if body_id != -1:
                    self.model.body_pos[body_id] = new_table_pos + rel_pos
        
        # 2. 테이블 회전 무작위화
        if 'table_rotation' in config:
            z_angle = np.random.uniform(*config['table_rotation']['z_range'])
            self.model.body_quat[self.table_id] = self._euler_to_quat(0, 0, z_angle)
        
        # 3. Lap joint 각도 무작위화
        if 'lap_joint_angle' in config:
            angle = np.random.uniform(*config['lap_joint_angle']['range'])
            current_quat = self.model.body_quat[self.lap_base_id].copy()
            rotation_quat = self._euler_to_quat(0, 0, angle)
            self.model.body_quat[self.lap_base_id] = self._quat_multiply(current_quat, rotation_quat)
        
        # 4. 움직이는 장애물 위치 무작위화
        if 'obstacle_position' in config:
            for i, joint_id in enumerate(self.obstacle_joint_ids):
                if joint_id != -1 and joint_id < len(self.model.jnt_range):
                    range_min = self.model.jnt_range[joint_id][0]
                    range_max = self.model.jnt_range[joint_id][1]
                    joint_adr = self.model.jnt_qposadr[joint_id]
                    if joint_adr < len(self.data.qpos):
                        self.data.qpos[joint_adr] = np.random.uniform(range_min, range_max)
        
        # 5. 작업자 위치 무작위화
        if 'worker_position' in config and self.worker_torso_id != -1:
            worker_config = config['worker_position']
            current_pos = self.model.body_pos[self.worker_torso_id].copy()
            current_pos[0] += np.random.uniform(*worker_config['x_range'])
            current_pos[1] = np.random.uniform(*worker_config['y_range'])
            self.model.body_pos[self.worker_torso_id] = current_pos
    
    def reset(self, seed=None, options=None):
        """환경 리셋"""
        super().reset(seed=seed)
        
        # 모델 리셋
        mujoco.mj_resetData(self.model, self.data)
        
        # Domain Randomization 적용
        self._randomize_environment()
        
        # 로봇 초기 위치 설정
        self.data.qpos[:self.n_joints] = self.init_qpos
        self.data.qvel[:self.n_joints] = 0
        
        # 웨이포인트 초기화
        self.current_waypoint_idx = 0
        self.collision_penalty = 0
        
        # Forward kinematics 업데이트
        mujoco.mj_forward(self.model, self.data)
        
        # 에피소드 정보 초기화
        self.step_count = 0
        
        # 초기 관찰 반환
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        """환경 스텝"""
        # 행동 적용 (속도 제어)
        self.data.ctrl[:self.n_joints] = action * 2.0  # 스케일링
        
        # 시뮬레이션 진행
        mujoco.mj_step(self.model, self.data)
        
        # 움직이는 장애물 애니메이션
        if self.enable_randomization and hasattr(self, 'obstacle_joint_ids'):
            t = self.data.time
            if self.obstacle_joint_ids[0] != -1:
                joint_adr = self.model.jnt_qposadr[self.obstacle_joint_ids[0]]
                if joint_adr < len(self.data.qpos):
                    self.data.qpos[joint_adr] = 0.05 * np.sin(0.5 * t)
        
        self.step_count += 1
        
        # 웨이포인트 진행 체크
        self._check_waypoint_progress()
        
        # 관찰, 보상, 종료 조건 계산
        obs = self._get_observation()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_episode_steps
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _check_waypoint_progress(self):
        """웨이포인트 진행 상황 체크"""
        if self.current_waypoint_idx >= len(self.waypoint_names):
            return
        
        # 현재 목표 웨이포인트 위치
        current_target = self._get_current_waypoint_position()
        
        # End-effector 위치
        ee_pos = self._get_ee_position()
        
        # 거리 계산
        distance = np.linalg.norm(ee_pos - current_target)
        
        # 웨이포인트 도달 확인
        if distance < self.waypoint_reached_threshold:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx < len(self.waypoint_names):
                print(f"Waypoint {self.current_waypoint_idx-1} reached! Moving to {self.waypoint_names[self.current_waypoint_idx]}")
    
    def _get_observation(self):
        """현재 관찰 반환"""
        # 로봇 관절 정보
        robot_qpos = self.data.qpos[:self.n_joints]
        robot_qvel = self.data.qvel[:self.n_joints]
        
        # End-effector 위치
        ee_pos = self._get_ee_position()
        
        # 현재 목표 웨이포인트 위치
        current_target = self._get_current_waypoint_position()
        
        # 웨이포인트 진행도 (0~1)
        waypoint_progress = self.current_waypoint_idx / len(self.waypoint_names)
        
        # 충돌 정보 (0: 충돌 없음, 1: 충돌)
        collision_detected = self._detect_collision()
        
        obs = np.concatenate([
            robot_qpos,                # 7
            robot_qvel,                # 7  
            ee_pos,                    # 3
            current_target,            # 3
            [waypoint_progress],       # 1
            [collision_detected]       # 1
        ])
        
        return obs.astype(np.float32)
    
    def _get_ee_position(self):
        """End-effector 위치 반환"""
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        if ee_site_id != -1:
            return self.data.site_xpos[ee_site_id].copy()
        else:
            # ee_site가 없으면 panda_hand body 위치 사용
            hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
            if hand_body_id != -1:
                return self.data.xpos[hand_body_id].copy()
        return np.zeros(3)
    
    def _get_current_waypoint_position(self):
        """현재 목표 웨이포인트 위치 반환"""
        if self.current_waypoint_idx >= len(self.waypoint_names):
            # 모든 웨이포인트 완료 시 마지막 웨이포인트 반환
            waypoint_name = self.waypoint_names[-1]
        else:
            waypoint_name = self.waypoint_names[self.current_waypoint_idx]
        
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, waypoint_name)
        if site_id != -1:
            return self.data.site_xpos[site_id].copy()
        return np.zeros(3)
    
    def _detect_collision(self):
        """충돌 감지"""
        if self.data.ncon == 0:
            return 0.0
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            # 로봇과 장애물 간 충돌 확인
            robot_collision = (geom1 in self.robot_geom_ids and geom2 in self.obstacle_geom_ids) or \
                            (geom2 in self.robot_geom_ids and geom1 in self.obstacle_geom_ids)
            
            if robot_collision:
                return 1.0
        
        return 0.0
    
    def _compute_reward(self):
        """보상 계산"""
        # End-effector와 현재 목표 웨이포인트 간 거리
        ee_pos = self._get_ee_position()
        current_target = self._get_current_waypoint_position()
        ee_error = np.linalg.norm(ee_pos - current_target)
        
        # 기본 거리 기반 보상 (거리가 가까울수록 높은 보상)
        distance_reward = -ee_error
        
        # 웨이포인트 도달 보너스
        waypoint_bonus = 0.0
        if ee_error < self.waypoint_reached_threshold:
            waypoint_bonus = 5.0  # 웨이포인트 도달 시 보너스
            
        # 최종 목표 도달 시 큰 보너스
        final_bonus = 0.0
        if self.current_waypoint_idx >= len(self.waypoint_names) - 1 and ee_error < self.waypoint_reached_threshold:
            final_bonus = 20.0
        
        # 웨이포인트 진행 보상
        progress_reward = self.current_waypoint_idx * 2.0
        
        # 충돌 페널티
        collision_penalty = 0.0
        if self._detect_collision():
            collision_penalty = -5.0
            self.collision_penalty += 1
        
        # 관절 움직임 페널티 (너무 큰 움직임 방지)
        joint_velocity_penalty = -0.01 * np.sum(np.abs(self.data.qvel[:self.n_joints]))
        
        # 관절 한계 페널티
        joint_limit_penalty = 0.0
        for i in range(self.n_joints):
            joint_range = self.model.jnt_range[i]
            if joint_range[0] != joint_range[1]:  # limited joint
                joint_pos = self.data.qpos[i]
                if joint_pos < joint_range[0] + 0.1 or joint_pos > joint_range[1] - 0.1:
                    joint_limit_penalty -= 1.0
        
        # 총 보상
        total_reward = (distance_reward + waypoint_bonus + final_bonus + 
                       progress_reward + collision_penalty + 
                       joint_velocity_penalty + joint_limit_penalty)
        
        return total_reward
    
    def _is_terminated(self):
        """종료 조건 확인"""
        # 최종 목표 도달 확인
        if self.current_waypoint_idx >= len(self.waypoint_names):
            ee_pos = self._get_ee_position()
            final_target = self._get_current_waypoint_position()
            distance = np.linalg.norm(ee_pos - final_target)
            
            if distance < self.waypoint_reached_threshold:
                return True
        
        # 너무 많은 충돌 시 종료
        if self.collision_penalty > 10:
            return True
        
        return False
    
    def _get_info(self):
        """추가 정보 반환"""
        info = {
            'step_count': self.step_count,
            'time': self.data.time,
            'current_waypoint_idx': self.current_waypoint_idx,
            'total_waypoints': len(self.waypoint_names),
            'collision_count': self.collision_penalty,
        }
        
        # 현재 EE 위치와 목표 거리
        ee_pos = self._get_ee_position()
        current_target = self._get_current_waypoint_position()
        info['ee_distance'] = np.linalg.norm(ee_pos - current_target)
        
        # 각 웨이포인트까지의 거리
        waypoint_distances = []
        for waypoint_name in self.waypoint_names:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, waypoint_name)
            if site_id != -1:
                wp_pos = self.data.site_xpos[site_id]
                dist = np.linalg.norm(ee_pos - wp_pos)
                waypoint_distances.append(dist)
        info['waypoint_distances'] = waypoint_distances
            
        return info
    
    def render(self):
        """환경 렌더링"""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data)
            return self.renderer.render()
    
    def close(self):
        """리소스 정리"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer = None
    
    def _euler_to_quat(self, roll, pitch, yaw):
        """Euler angles to quaternion"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])
    
    def _quat_multiply(self, q1, q2):
        """Quaternion multiplication"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
