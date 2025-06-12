# /home/minjun/rl_ws/src/mujoco_rl_env/train2.py
import argparse
import os
import imageio
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from mujoco_rl_env.env.franka_reach_env2 import FrankaReachEnv

class VideoRecorderCallback(BaseCallback):
    """50 iteration마다 비디오를 저장하는 커스텀 콜백"""
    def __init__(self, eval_env_fn, video_folder="videos", video_length=500, record_freq=50, run_name="", verbose=1):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.video_folder = video_folder
        self.video_length = video_length
        self.record_freq = record_freq
        self.run_name = run_name
        self.iteration_count = 0
        os.makedirs(video_folder, exist_ok=True)
        
    def _on_step(self) -> bool:
        if self.n_calls % 2048 == 0:
            self.iteration_count += 1
            
            if self.iteration_count % self.record_freq == 0:
                self._record_video()
                
        return True
    
    def _record_video(self):
        """현재 정책으로 비디오 녹화"""
        if self.verbose > 0:
            print(f"\n🎬 Recording video at iteration {self.iteration_count}...")
        
        # 평가용 환경 생성 (무작위화 없이)
        env = self.eval_env_fn()
        
        if hasattr(self.model, 'get_vec_normalize_env'):
            vec_norm = self.model.get_vec_normalize_env()
            if vec_norm is not None:
                eval_env_norm = VecNormalize(DummyVecEnv([lambda: env]), training=False)
                eval_env_norm.obs_rms = vec_norm.obs_rms
                eval_env_norm.ret_rms = vec_norm.ret_rms
                obs = eval_env_norm.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                eval_env = eval_env_norm
            else:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
        else:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
        
        frames = []
        episode_reward = 0
        waypoints_reached = 0
        
        for step in range(self.video_length):
            if isinstance(eval_env, VecNormalize):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                frame = eval_env.venv.envs[0].render()
                current_wp_idx = info[0].get('current_waypoint_idx', 0)
            else:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                frame = env.render()
                current_wp_idx = info.get('current_waypoint_idx', 0)
            
            if isinstance(reward, np.ndarray):
                reward = reward.item()
            episode_reward += reward
            waypoints_reached = max(waypoints_reached, current_wp_idx)
            
            if frame is not None:
                frames.append(frame)
            
            if isinstance(done, np.ndarray):
                done = done.item()
            if done:
                print(f"   🎯 Goal reached at step {step}! Waypoints: {waypoints_reached}/{len(['lap_start', 'lap_waypoint1', 'lap_waypoint2', 'lap_waypoint3', 'lap_end'])}")
                for _ in range(min(30, self.video_length - step)):
                    frame = env.render() if not isinstance(eval_env, VecNormalize) else eval_env.venv.envs[0].render()
                    if frame is not None:
                        frames.append(frame)
                break
        
        if frames:
            filename = f"{self.video_folder}/{self.run_name}_iter_{self.iteration_count:04d}.mp4"
            imageio.mimsave(filename, frames, fps=30)
            
            if self.verbose > 0:
                print(f"✅ Video saved: {filename}")
                reward_value = float(episode_reward) if isinstance(episode_reward, (int, float)) else float(episode_reward.item())
                print(f"   Episode reward: {reward_value:.2f}")
                print(f"   Waypoints reached: {waypoints_reached}")
                print(f"   Episode length: {len(frames)} frames ({len(frames)/30:.1f} seconds)")
        
        if isinstance(eval_env, VecNormalize):
            eval_env.close()
        else:
            env.close()

def get_randomization_config(level='medium'):
    """난이도별 무작위화 설정 반환"""
    if level == 'none':
        return None
    elif level == 'easy':
        return {
            'table_position': {
                'x_range': (0.48, 0.52),
                'y_range': (-0.02, 0.02),
                'z_range': (0.44, 0.46)
            },
            'lap_joint_angle': {
                'range': (-0.05, 0.05)
            }
        }
    elif level == 'medium':
        return {
            'table_position': {
                'x_range': (0.45, 0.55),
                'y_range': (-0.05, 0.05),
                'z_range': (0.43, 0.47)
            },
            'table_rotation': {
                'z_range': (-0.1, 0.1)
            },
            'lap_joint_angle': {
                'range': (-0.1, 0.1)
            },
            'obstacle_position': {
                'range': 0.05
            }
        }
    elif level == 'hard':
        return {
            'table_position': {
                'x_range': (0.4, 0.6),
                'y_range': (-0.1, 0.1),
                'z_range': (0.4, 0.5)
            },
            'table_rotation': {
                'z_range': (-0.2, 0.2)
            },
            'lap_joint_angle': {
                'range': (-0.15, 0.15)
            },
            'obstacle_position': {
                'range': 0.1
            },
            'worker_position': {
                'x_range': (-0.1, 0.1),
                'y_range': (-0.35, -0.2)
            }
        }
    else:
        raise ValueError(f"Unknown randomization level: {level}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO on the Franka waypoint navigation environment with Domain Randomization"
    )
    parser.add_argument(
        "--xml", default="models/fr3_rl_reach.xml",
        help="Path to your MuJoCo XML model"
    )
    parser.add_argument(
        "--video-length", type=int, default=500,
        help="Maximum video length in steps (default: 500)"
    )
    parser.add_argument(
        "--record-freq", type=int, default=50,
        help="Record video every N iterations (default: 50)"
    )
    parser.add_argument(
        "--randomization", type=str, default="medium",
        choices=['none', 'easy', 'medium', 'hard'],
        help="Domain randomization level (default: medium)"
    )
    parser.add_argument(
        "--eval-randomization", type=str, default="none",
        choices=['none', 'easy', 'medium', 'hard'],
        help="Randomization level for evaluation (default: none)"
    )
    args = parser.parse_args()

    # 실행 시간 기반 고유 ID 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_franka_waypoints_DR_{args.randomization}_{timestamp}"

    # 디렉토리 생성
    os.makedirs("videos", exist_ok=True)
    os.makedirs(f"runs/{run_name}/best_model", exist_ok=True)
    os.makedirs(f"runs/{run_name}/logs", exist_ok=True)

    # 무작위화 설정
    train_randomization_config = get_randomization_config(args.randomization)
    eval_randomization_config = get_randomization_config(args.eval_randomization)

    # 1) 학습용 환경 생성 (하드코딩된 joint 값 사용)
    def make_train_env():
        env = FrankaReachEnv(
            xml_path=args.xml,
            init_qpos=[0.0555323, -0.519356, -0.522075, -2.239023, -0.030734, 2.252302, 0.329849],  # lap_start 위치
            goal_qpos=[-0.0214929, -0.7660162, -0.3266901, -2.4520896, -0.0856205, 2.2117625, 0.4996375],  # lap_end 위치
            render_mode=None,
            enable_randomization=(args.randomization != 'none'),
            randomization_config=train_randomization_config
        )
        env = Monitor(env)
        return env
    
    # 2) 평가용 환경 생성 함수
    def make_eval_env():
        env = FrankaReachEnv(
            xml_path=args.xml,
            init_qpos=[0.0555323, -0.519356, -0.522075, -2.239023, -0.030734, 2.252302, 0.329849],  # lap_start 위치
            goal_qpos=[-0.0214929, -0.7660162, -0.3266901, -2.4520896, -0.0856205, 2.2117625, 0.4996375],  # lap_end 위치
            render_mode="rgb_array",
            enable_randomization=(args.eval_randomization != 'none'),
            randomization_config=eval_randomization_config
        )
        env = Monitor(env)
        return env

    # Vectorized 환경 생성 (병렬 학습을 위해 여러 환경 사용)
    n_envs = 4  # 병렬 환경 수
    train_env = DummyVecEnv([make_train_env for _ in range(n_envs)])
    
    # 정규화 래퍼 추가
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    # 평가 환경 설정
    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        training=False,
        clip_obs=10.0
    )
    
    # 3) 콜백 설정
    # EvalCallback - TensorBoard에 eval 메트릭 기록
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"runs/{run_name}/best_model",
        log_path=f"runs/{run_name}/logs",
        eval_freq=args.record_freq * 2048 // n_envs,  # 병렬 환경 고려
        n_eval_episodes=5,
        deterministic=True,
        verbose=1
    )
    
    # VideoRecorderCallback - 비디오 녹화
    video_callback = VideoRecorderCallback(
        eval_env_fn=make_eval_env,
        video_folder=f"videos/{run_name}",
        video_length=args.video_length,
        record_freq=args.record_freq,
        run_name=run_name,
        verbose=1
    )
    
    # 콜백 리스트로 묶기
    callbacks = CallbackList([eval_callback, video_callback])
    
    # 4) PPO 모델 생성 및 학습
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048 // n_envs,  # 병렬 환경 고려
        batch_size=128,
        n_epochs=20,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=f"runs/{run_name}/tensorboard"
    )
    
    # 환경 정보 출력
    test_env = make_train_env()
    print("="*80)
    print(f"PPO Waypoint Navigation Training: {run_name}")
    print(f"- XML model: {args.xml}")
    print(f"- Waypoints: {test_env.unwrapped.waypoint_names}")
    print(f"- Initial pose: [0.0555323, -0.519356, -0.522075, -2.239023, -0.030734, 2.252302, 0.329849]")
    print(f"- Goal pose: [-0.0214929, -0.7660162, -0.3266901, -2.4520896, -0.0856205, 2.2117625, 0.4996375]")
    print(f"- Training randomization: {args.randomization}")
    print(f"- Evaluation randomization: {args.eval_randomization}")
    print(f"- Parallel environments: {n_envs}")
    print(f"- Max episode steps: {test_env.unwrapped.max_episode_steps}")
    print(f"- Recording video every {args.record_freq} iterations")
    print(f"- Video length: {args.video_length} steps ({args.video_length/30:.1f} seconds)")
    print(f"- Videos: videos/{run_name}/")
    print(f"- TensorBoard: runs/{run_name}/tensorboard")
    print("="*80)
    test_env.close()
    
    # 정규화 통계 동기화
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
    
    try:
        model.learn(
            total_timesteps=5_000_000,  # 웨이포인트 네비게이션을 위해 더 많은 스텝
            callback=callbacks,
            log_interval=10,
            tb_log_name=run_name
        )
        
        # 모델 및 정규화 통계 저장
        model.save(f"runs/{run_name}/final_model")
        train_env.save(f"runs/{run_name}/vec_normalize.pkl")
        
        print(f"\n✅ Training completed successfully!")
        print(f"🔸 Run name: {run_name}")
        print(f"🔸 Final model: runs/{run_name}/final_model.zip")
        print(f"🔸 Best model: runs/{run_name}/best_model/best_model.zip")
        print(f"🔸 Normalization stats: runs/{run_name}/vec_normalize.pkl")
        
        # Domain Randomization 설정 저장
        import json
        dr_config = {
            'training_randomization': args.randomization,
            'evaluation_randomization': args.eval_randomization,
            'config': train_randomization_config,
            'waypoints': ["lap_start", "lap_waypoint1", "lap_waypoint2", "lap_waypoint3", "lap_end"]
        }
        with open(f"runs/{run_name}/dr_config.json", 'w') as f:
            json.dump(dr_config, f, indent=2)
        print(f"🔸 DR config saved: runs/{run_name}/dr_config.json")
        
        # 저장된 비디오 목록 출력
        videos = sorted([f for f in os.listdir(f"videos/{run_name}") if f.endswith('.mp4')])
        print(f"\n📹 Saved videos ({len(videos)} total):")
        for video in videos[-5:]:
            print(f"   - {video}")
        if len(videos) > 5:
            print(f"   ... and {len(videos)-5} more")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        model.save(f"runs/{run_name}/interrupted_model")
        train_env.save(f"runs/{run_name}/vec_normalize.pkl")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise
        
    finally:
        train_env.close()
        eval_env.close()
        print(f"\n💡 To monitor training progress, run:")
        print(f"   tensorboard --logdir runs/{run_name}/tensorboard")

