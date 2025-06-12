#------------------------------------------------------------------------------------------------------------------
# video length조절, env.py바뀜에 따라 똑같이 바꿔줌 - 20250608 1430\
# train2.py로 바꿈
# /home/minjun/rl_ws/src/mujoco_rl_env/train.py
import argparse
import os
import imageio
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from mujoco_rl_env.env.franka_reach_env import FrankaReachEnv

class VideoRecorderCallback(BaseCallback):
    """50 iteration마다 비디오를 저장하는 커스텀 콜백"""
    # video length 300으로 변경, record freq 50(iteration) , 100일때 4초쯤
    def __init__(self, eval_env_fn, video_folder="videos", video_length=300, record_freq=50, run_name="", verbose=1):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.video_folder = video_folder
        self.video_length = video_length # 300 steps = 10 seconds @ 30fps
        self.record_freq = record_freq
        self.run_name = run_name
        self.iteration_count = 0
        os.makedirs(video_folder, exist_ok=True)
        
    def _on_step(self) -> bool:
        # 매 2048 스텝(1 iteration)마다 카운트
        if self.n_calls % 2048 == 0:
            self.iteration_count += 1
            
            # 50 iteration마다 비디오 녹화 
            if self.iteration_count % self.record_freq == 0:
                self._record_video()
                
        return True
    
    def _record_video(self):
        """현재 정책으로 비디오 녹화"""
        if self.verbose > 0:
            print(f"\n🎬 Recording video at iteration {self.iteration_count}...")
        
        # 평가용 환경 생성 (정규화 없이)
        env = self.eval_env_fn()
        
        # VecNormalize 래퍼가 있다면 정규화 통계 동기화
        if hasattr(self.model, 'get_vec_normalize_env'):
            vec_norm = self.model.get_vec_normalize_env()
            if vec_norm is not None:
                eval_env_norm = VecNormalize(DummyVecEnv([lambda: env]), training=False)
                eval_env_norm.obs_rms = vec_norm.obs_rms
                eval_env_norm.ret_rms = vec_norm.ret_rms
                obs = eval_env_norm.reset()
                eval_env = eval_env_norm
            else:
                obs = env.reset()
        else:
            obs = env.reset()
        
        frames = []
        episode_reward = 0
        
        for step in range(self.video_length):
            # 현재 정책으로 행동 선택
            if isinstance(eval_env, VecNormalize):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                # 실제 환경에서 프레임 가져오기
                frame = eval_env.venv.envs[0].render()
            else:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                frame = env.render()
            
            # reward가 배열인 경우 스칼라로 변환
            if isinstance(reward, np.ndarray):
                reward = reward.item()
            episode_reward += reward
            
            if frame is not None:
                frames.append(frame)
            
            # done도 배열일 수 있음
            if isinstance(done, np.ndarray):
                done = done.item()
            if done:
                print(f"   🎯 Goal reached at step {step}!")
                # 목표 도달 후 추가 프레임 기록 (성공 확인용)
                for _ in range(min(30, self.video_length - step)):  # 1초 더
                    frame = env.render() if not isinstance(eval_env, VecNormalize) else eval_env.venv.envs[0].render()
                    if frame is not None:
                        frames.append(frame)
                break
        
        # 비디오 저장
        if frames:
            # 파일명에 run_name 포함
            filename = f"{self.video_folder}/{self.run_name}_iter_{self.iteration_count:04d}.mp4"
            
            # 파일명 동일 - 겹침 이슈
            # start_iter = self.iteration_count - 49
            # end_iter = self.iteration_count
            # filename = f"{self.video_folder}/iteration_{start_iter:03d}-{end_iter:03d}.mp4"
            
            imageio.mimsave(filename, frames, fps=30)
            
            if self.verbose > 0:
                print(f"✅ Video saved: {filename}")
                # episode_reward가 배열일 수 있으므로 float로 변환
                reward_value = float(episode_reward) if isinstance(episode_reward, (int, float)) else float(episode_reward.item())
                print(f"   Episode reward: {reward_value:.2f}")
                print(f"   Episode length: {len(frames)} frames ({len(frames)/30:.1f} seconds)")
        
        # 환경 정리
        if isinstance(eval_env, VecNormalize):
            eval_env.close()
        else:
            env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO on the Franka reach environment"
    )
    parser.add_argument(
        "--xml", default="models/fr3_reach.xml",
        help="Path to your MuJoCo XML model"
    )
    parser.add_argument(
        "--init", nargs='*', type=float,
        default=None,  # None으로 기본값 설정
        help="Initial joint positions (optional)"
    )
    parser.add_argument(
        "--goal", nargs='*', type=float,
        default=None,  # None으로 기본값 설정
        help="Goal joint positions (optional, length = model.nq)"
    )
    parser.add_argument(
        "--video-length", type=int, default=300,
        help="Maximum video length in steps (default: 300 = 10 seconds)"
    )
    parser.add_argument(
        "--record-freq", type=int, default=50,
        help="Record video every N iterations (default: 50)"
    )
    args = parser.parse_args()

    # 실행 시간 기반 고유 ID 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_franka_{timestamp}"

    # 디렉토리 생성
    os.makedirs("videos", exist_ok=True)
    os.makedirs("runs/best_model", exist_ok=True)
    os.makedirs("runs/franka_reach", exist_ok=True)

    # init 인자 처리
    if args.init is not None and len(args.init) > 0:
        init_qpos = args.init
    else:
        init_qpos = None  # 환경에서 기본값 사용
        print("ℹ️  No initial pose provided, using lap_start from XML")

    # goal 인자 처리
    if args.goal is not None and len(args.goal)>0:
        goal_qpos = args.goal
        print(f"✅ Goal joint positions: {goal_qpos}")
    else:
        goal_qpos = None  # 환경에서 lap_end 사용
        print("ℹ️  No goal_qpos provided, sampling or using default init as goal")

    # 1) 학습용 환경 생성 (Monitor로 감싸기)
    def make_train_env():
        env = FrankaReachEnv(
            xml_path=args.xml,
            init_qpos=init_qpos,
            goal_qpos=goal_qpos,
            render_mode=None
        )
        # Monitor로 감싸서 에피소드 통계 기록
        env = Monitor(env)
        return env
    
    # 2) 평가용 환경 생성 함수
    def make_eval_env():
        env = FrankaReachEnv(
            xml_path=args.xml,
            init_qpos=init_qpos,
            goal_qpos=goal_qpos,
            render_mode="rgb_array"
        )
        # 평가 환경도 Monitor로 감싸기
        env = Monitor(env)
        return env
    
    # 평가 환경 설정 (정규화 포함)
    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        training=False,  # 평가시에는 통계 업데이트 안함
        clip_obs=10.0
    )
    
    # 3) 콜백 설정
    # EvalCallback - TensorBoard에 eval 메트릭 기록
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"runs/{run_name}/best_model",
        log_path=f"runs/{run_name}/logs",
        eval_freq=args.record_freq * 2048,  # 50 iteration마다
        n_eval_episodes=5,     # 5 에피소드 평균
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
        learning_rate=2e-4, #기존 3e-4
        n_steps=2048,
        batch_size=64,
        n_epochs=20,        #기존 10
        gamma=0.995,        #기존 0.99 
        gae_lambda=0.95,
        clip_range=0.2,     #기존 0.2
        ent_coef=0.05,      #기존 0.1
        vf_coef=0.1,        #기존 0.05
        max_grad_norm=0.5, #기존 ent0.5
        verbose=1,
        tensorboard_log=f"runs/{run_name}/tensorboard"
    )
    
    print("="*60)
    print("PPO Training on Franka Reach Environment: {run_name}")
    print(f"- Initial pose: {init_qpos if init_qpos else 'lap_start (from XML)'}")
    print(f"- Goal pose: {goal_qpos if goal_qpos else 'randomly sampled or init'}")
    print(f"- Recording video every {args.record_freq} iterations")
    print(f"- Evaluating every 50 iterations (5 episodes)")
    print(f"- Video length: {args.video_length} steps ({args.video_length/30:.1f} seconds)")
    print(f"- Videos will be saved to: videos/{run_name}/")
    print(f"- TensorBoard logs: runs/{run_name}/tensorboard")
    print("="*60)
    
    # 정규화 통계 동기화
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
    
    try:
        model.learn(
            total_timesteps=3_000_000, #timestep 1000000까지
            callback=callbacks,
            log_interval=10,
            tb_log_name=run_name # TensorBoard 로그 이름
        )
        
        # 모델 및 정규화 통계 저장
        model.save(f"runs/{run_name}/final_model")
        train_env.save(f"runs/{run_name}/vec_normalize.pkl")
        
        print(f"\n✅ Training completed successfully!")
        print(f"🔸 Run name: {run_name}")
        print(f"🔸 Final model: runs/{run_name}/final_model.zip")
        print(f"🔸 Best model: runs/{run_name}/best_model/best_model.zip")
        print(f"🔸 Normalization stats: runs/{run_name}/vec_normalize.pkl")
        
        # 저장된 비디오 목록 출력
        videos = sorted([f for f in os.listdir(f"videos/{run_name}") if f.endswith('.mp4')])
        print(f"\n📹 Saved videos ({len(videos)} total):")
        for video in videos[-5:]:  # 마지막 5개만 표시
            print(f"   - {video}")
        if len(videos) > 5:
            print(f"   ... and {len(videos)-5} more")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        model.save(f"runs/{run_name}/interrupted_model")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise
        
    finally:
        train_env.close()
        eval_env.close()
        print(f"\n💡 To monitor training progress, run:")
        print(f"   tensorboard --logdir runs/{run_name}/tensorboard")

