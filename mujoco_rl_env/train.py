#------------------------------------------------------------------------------------------------------------------
# video length조절, env.py바뀜에 따라 똑같이 바꿔줌 - 20250608 1430
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
    
    # 학습 환경 설정
    train_env = DummyVecEnv([make_train_env])
    train_env = VecNormalize(
        train_env, 
        norm_obs=True, 
        norm_reward=False,
        clip_obs=10.0
    )
    
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
        learning_rate=3e-4, #기존 3e-4
        n_steps=2048,
        batch_size=64,
        n_epochs=20, #기존 10
        gamma=0.995,# 기존 0.99 
        gae_lambda=0.95,
        clip_range=0.3, #기존 0.2
        ent_coef=0.05,
        vf_coef=0.3, # 기존 0.5
        max_grad_norm=0.7, #기존 0.5
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
            total_timesteps=1_000_000, #timestep 1000000까지
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



# ------------------------------------------------------------------------------
# /home/minjun/rl_ws/src/mujoco_rl_env/train.py
# import argparse
# import os
# import imageio
# import numpy as np
# from datetime import datetime
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines3.common.monitor import Monitor
# from mujoco_rl_env.env.franka_reach_env import FrankaReachEnv

# class VideoRecorderCallback(BaseCallback):
#     """비디오를 저장하는 커스텀 콜백"""
#     def __init__(self, eval_env_fn, video_folder="videos", video_length=300, 
#                  record_freq=50, run_name="", verbose=1):
#         super().__init__(verbose)
#         self.eval_env_fn = eval_env_fn
#         self.video_folder = video_folder
#         self.video_length = video_length  # 300 steps = 10 seconds @ 30fps
#         self.record_freq = record_freq
#         self.run_name = run_name
#         self.iteration_count = 0
#         os.makedirs(video_folder, exist_ok=True)
        
#     def _on_step(self) -> bool:
#         # 매 2048 스텝(1 iteration)마다 카운트
#         if self.n_calls % 2048 == 0:
#             self.iteration_count += 1
            
#             # record_freq iteration마다 비디오 녹화
#             if self.iteration_count % self.record_freq == 0:
#                 self._record_video()
                
#         return True
    
#     def _record_video(self):
#         """현재 정책으로 비디오 녹화"""
#         if self.verbose > 0:
#             print(f"\n🎬 Recording video at iteration {self.iteration_count}...")
        
#         # 평가용 환경 생성
#         env = self.eval_env_fn()
        
#         # VecNormalize 래퍼가 있다면 정규화 통계 동기화
#         if hasattr(self.model, 'get_vec_normalize_env'):
#             vec_norm = self.model.get_vec_normalize_env()
#             if vec_norm is not None:
#                 eval_env_norm = VecNormalize(DummyVecEnv([lambda: env]), training=False)
#                 eval_env_norm.obs_rms = vec_norm.obs_rms
#                 eval_env_norm.ret_rms = vec_norm.ret_rms
#                 obs = eval_env_norm.reset()
#                 eval_env = eval_env_norm
#             else:
#                 obs = env.reset()
#         else:
#             obs = env.reset()
        
#         frames = []
#         episode_reward = 0
        
#         for step in range(self.video_length):
#             # 현재 정책으로 행동 선택
#             if isinstance(eval_env, VecNormalize):
#                 action, _ = self.model.predict(obs, deterministic=True)
#                 obs, reward, done, info = eval_env.step(action)
#                 frame = eval_env.venv.envs[0].render()
#             else:
#                 action, _ = self.model.predict(obs, deterministic=True)
#                 obs, reward, done, info = env.step(action)
#                 frame = env.render()
            
#             # reward가 배열인 경우 스칼라로 변환
#             if isinstance(reward, np.ndarray):
#                 reward = reward.item()
#             episode_reward += reward
            
#             if frame is not None:
#                 frames.append(frame)
            
#             # done도 배열일 수 있음
#             if isinstance(done, np.ndarray):
#                 done = done.item()
#             if done:
#                 print(f"   🎯 Goal reached at step {step}!")
#                 # 목표 도달 후 추가 프레임 기록 (성공 확인용)
#                 for _ in range(min(30, self.video_length - step)):  # 1초 더
#                     frame = env.render() if not isinstance(eval_env, VecNormalize) else eval_env.venv.envs[0].render()
#                     if frame is not None:
#                         frames.append(frame)
#                 break
        
#         # 비디오 저장
#         if frames:
#             # 파일명에 run_name 포함
#             filename = f"{self.video_folder}/{self.run_name}_iter_{self.iteration_count:04d}.mp4"
            
#             imageio.mimsave(filename, frames, fps=30)
            
#             if self.verbose > 0:
#                 print(f"✅ Video saved: {filename}")
#                 reward_value = float(episode_reward) if isinstance(episode_reward, (int, float)) else float(episode_reward.item())
#                 print(f"   Episode reward: {reward_value:.2f}")
#                 print(f"   Episode length: {len(frames)} frames ({len(frames)/30:.1f} seconds)")
        
#         # 환경 정리
#         if isinstance(eval_env, VecNormalize):
#             eval_env.close()
#         else:
#             env.close()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Train PPO on the Franka reach environment"
#     )
#     parser.add_argument(
#         "--xml", default="models/fr3_reach.xml",
#         help="Path to your MuJoCo XML model"
#     )
#     parser.add_argument(
#         "--init", nargs='*', type=float,
#         default=None,
#         help="Initial joint positions (optional)"
#     )
#     parser.add_argument(
#         "--goal", nargs=7, type=float,
#         default=None,
#         help="Goal joint positions for 7 DOF arm"
#     )
#     parser.add_argument(
#         "--video-length", type=int, default=300,
#         help="Maximum video length in steps (default: 300 = 10 seconds)"
#     )
#     parser.add_argument(
#         "--record-freq", type=int, default=50,
#         help="Record video every N iterations (default: 50)"
#     )
#     args = parser.parse_args()

#     # 실행 시간 기반 고유 ID 생성
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     run_name = f"ppo_franka_{timestamp}"
    
#     # 디렉토리 생성
#     os.makedirs(f"videos/{run_name}", exist_ok=True)
#     os.makedirs(f"runs/{run_name}/best_model", exist_ok=True)
#     os.makedirs(f"runs/{run_name}/logs", exist_ok=True)

#     # init 인자 처리
#     if args.init is not None and len(args.init) > 0:
#         init_qpos = args.init
#     else:
#         init_qpos = None
#         print("ℹ️  No initial pose provided, will try to use lap_start keyframe from XML")

#     # goal 인자 처리 (joint positions)
#     if args.goal is not None:
#         goal_qpos = args.goal
#         print(f"✅ Goal joint positions: {goal_qpos}")
#     else:
#         goal_qpos = None
#         print("ℹ️  No goal joint positions provided")

#     # 1) 학습용 환경 생성
#     def make_train_env():
#         env = FrankaReachEnv(
#             xml_path=args.xml,
#             init_qpos=init_qpos,
#             goal_qpos=goal_qpos,
#             render_mode=None
#         )
#         env = Monitor(env, f"runs/{run_name}/logs")
#         return env
    
#     # 2) 평가용 환경 생성 함수
#     def make_eval_env():
#         env = FrankaReachEnv(
#             xml_path=args.xml,
#             init_qpos=init_qpos,
#             goal_qpos=goal_qpos,
#             render_mode="rgb_array"
#         )
#         env = Monitor(env)
#         return env
    
#     # 학습 환경 설정
#     train_env = DummyVecEnv([make_train_env])
#     train_env = VecNormalize(
#         train_env, 
#         norm_obs=True, 
#         norm_reward=True,  # reward normalization 활성화
#         clip_obs=10.0,
#         clip_reward=10.0
#     )
    
#     # 평가 환경 설정
#     eval_env = DummyVecEnv([make_eval_env])
#     eval_env = VecNormalize(
#         eval_env,
#         norm_obs=True,
#         norm_reward=False,
#         training=False,
#         clip_obs=10.0
#     )
    
#     # 3) 콜백 설정
#     # EvalCallback
#     eval_callback = EvalCallback(
#         eval_env,
#         best_model_save_path=f"runs/{run_name}/best_model",
#         log_path=f"runs/{run_name}/logs",
#         eval_freq=args.record_freq * 2048,  # record_freq iterations마다
#         n_eval_episodes=5,
#         deterministic=True,
#         verbose=1
#     )
    
#     # VideoRecorderCallback
#     video_callback = VideoRecorderCallback(
#         eval_env_fn=make_eval_env,
#         video_folder=f"videos/{run_name}",
#         video_length=args.video_length,
#         record_freq=args.record_freq,
#         run_name=run_name,
#         verbose=1
#     )
    
#     callbacks = CallbackList([eval_callback, video_callback])
    
#     # 4) PPO 모델 생성
#     model = PPO(
#         "MlpPolicy",
#         train_env,
#         learning_rate=3e-4,
#         n_steps=2048,
#         batch_size=64,
#         n_epochs=10,
#         gamma=0.99,
#         gae_lambda=0.95,
#         clip_range=0.2,
#         ent_coef=0.01,
#         vf_coef=0.5,
#         max_grad_norm=0.5,
#         verbose=1,
#         tensorboard_log=f"runs/{run_name}/tensorboard"
#     )
    
#     print("="*60)
#     print(f"PPO Training: {run_name}")
#     print(f"- Initial pose: {init_qpos if init_qpos else 'zeros'}")
#     print(f"- Goal joints: {goal_qpos if goal_qpos else 'same as initial'}")
#     print(f"- Recording video every {args.record_freq} iterations")
#     print(f"- Video length: {args.video_length} steps ({args.video_length/30:.1f} seconds)")
#     print(f"- Videos will be saved to: videos/{run_name}/")
#     print(f"- TensorBoard logs: runs/{run_name}/tensorboard")
#     print("="*60)
    
#     # 정규화 통계 동기화
#     eval_env.obs_rms = train_env.obs_rms
#     eval_env.ret_rms = train_env.ret_rms
    
#     try:
#         model.learn(
#             total_timesteps=500_000,
#             callback=callbacks,
#             log_interval=10,
#             tb_log_name=run_name
#         )
        
#         # 모델 및 정규화 통계 저장
#         model.save(f"runs/{run_name}/final_model")
#         train_env.save(f"runs/{run_name}/vec_normalize.pkl")
        
#         print(f"\n✅ Training completed successfully!")
#         print(f"🔸 Run name: {run_name}")
#         print(f"🔸 Final model: runs/{run_name}/final_model.zip")
#         print(f"🔸 Best model: runs/{run_name}/best_model/best_model.zip")
#         print(f"🔸 Normalization stats: runs/{run_name}/vec_normalize.pkl")
        
#         # 저장된 비디오 목록 출력
#         videos = sorted([f for f in os.listdir(f"videos/{run_name}") if f.endswith('.mp4')])
#         print(f"\n📹 Saved videos ({len(videos)} total):")
#         for video in videos[-5:]:  # 마지막 5개만 표시
#             print(f"   - {video}")
#         if len(videos) > 5:
#             print(f"   ... and {len(videos)-5} more")
        
#     except KeyboardInterrupt:
#         print("\n⚠️  Training interrupted by user")
#         model.save(f"runs/{run_name}/interrupted_model")
        
#     except Exception as e:
#         print(f"\n❌ Error during training: {e}")
#         raise
        
#     finally:
#         train_env.close()
#         eval_env.close()
#         print(f"\n💡 To monitor training progress, run:")
#         print(f"   tensorboard --logdir runs/{run_name}/tensorboard")




#------------------------------------------------------------------------------------------------------------------
# 20250608 1717 수정
# /home/minjun/rl_ws/src/mujoco_rl_env/train.py
# import argparse
# import os
# import imageio
# import numpy as np
# from datetime import datetime
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines3.common.monitor import Monitor
# from mujoco_rl_env.env.franka_reach_env import FrankaReachEnv

# class VideoRecorderCallback(BaseCallback):
#     """비디오를 저장하는 커스텀 콜백"""
#     def __init__(self, eval_env_fn, video_folder="videos", video_length=300, 
#                  record_freq=50, run_name="", verbose=1):
#         super().__init__(verbose)
#         self.eval_env_fn = eval_env_fn
#         self.video_folder = video_folder
#         self.video_length = video_length  # 300 steps = 10 seconds @ 30fps
#         self.record_freq = record_freq
#         self.run_name = run_name
#         self.iteration_count = 0
#         os.makedirs(video_folder, exist_ok=True)
        
#     def _on_step(self) -> bool:
#         # 매 2048 스텝(1 iteration)마다 카운트
#         if self.n_calls % 2048 == 0:
#             self.iteration_count += 1
            
#             # record_freq iteration마다 비디오 녹화
#             if self.iteration_count % self.record_freq == 0:
#                 self._record_video()
                
#         return True
    
#     def _record_video(self):
#         """현재 정책으로 비디오 녹화"""
#         if self.verbose > 0:
#             print(f"\n🎬 Recording video at iteration {self.iteration_count}...")
        
#         # 평가용 환경 생성
#         env = self.eval_env_fn()
        
#         # VecNormalize 래퍼가 있다면 정규화 통계 동기화
#         if hasattr(self.model, 'get_vec_normalize_env'):
#             vec_norm = self.model.get_vec_normalize_env()
#             if vec_norm is not None:
#                 eval_env_norm = VecNormalize(DummyVecEnv([lambda: env]), training=False)
#                 eval_env_norm.obs_rms = vec_norm.obs_rms
#                 eval_env_norm.ret_rms = vec_norm.ret_rms
#                 obs = eval_env_norm.reset()
#                 eval_env = eval_env_norm
#             else:
#                 obs = env.reset()
#         else:
#             obs = env.reset()
        
#         frames = []
#         episode_reward = 0
        
#         for step in range(self.video_length):
#             # 현재 정책으로 행동 선택
#             if isinstance(eval_env, VecNormalize):
#                 action, _ = self.model.predict(obs, deterministic=True)
#                 obs, reward, done, info = eval_env.step(action)
#                 frame = eval_env.venv.envs[0].render()
#             else:
#                 action, _ = self.model.predict(obs, deterministic=True)
#                 obs, reward, done, info = env.step(action)
#                 frame = env.render()
            
#             # reward가 배열인 경우 스칼라로 변환
#             if isinstance(reward, np.ndarray):
#                 reward = reward.item()
#             episode_reward += reward
            
#             if frame is not None:
#                 frames.append(frame)
            
#             # done도 배열일 수 있음
#             if isinstance(done, np.ndarray):
#                 done = done.item()
#             if done:
#                 print(f"   🎯 Goal reached at step {step}!")
#                 # 목표 도달 후 추가 프레임 기록 (성공 확인용)
#                 for _ in range(min(30, self.video_length - step)):  # 1초 더
#                     frame = env.render() if not isinstance(eval_env, VecNormalize) else eval_env.venv.envs[0].render()
#                     if frame is not None:
#                         frames.append(frame)
#                 break
        
#         # 비디오 저장
#         if frames:
#             # 파일명에 run_name 포함
#             filename = f"{self.video_folder}/{self.run_name}_iter_{self.iteration_count:04d}.mp4"
            
#             imageio.mimsave(filename, frames, fps=30)
            
#             if self.verbose > 0:
#                 print(f"✅ Video saved: {filename}")
#                 reward_value = float(episode_reward) if isinstance(episode_reward, (int, float)) else float(episode_reward.item())
#                 print(f"   Episode reward: {reward_value:.2f}")
#                 print(f"   Episode length: {len(frames)} frames ({len(frames)/30:.1f} seconds)")
        
#         # 환경 정리
#         if isinstance(eval_env, VecNormalize):
#             eval_env.close()
#         else:
#             env.close()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Train PPO on the Franka reach environment"
#     )
#     parser.add_argument(
#         "--xml", default="models/fr3_reach.xml",
#         help="Path to your MuJoCo XML model"
#     )
#     parser.add_argument(
#         "--init", nargs='*', type=float,
#         default=None,
#         help="Initial joint positions (optional)"
#     )
#     parser.add_argument(
#         "--goal", nargs=7, type=float,
#         default=None,
#         help="Goal pose: x y z qx qy qz qw"
#     )
#     parser.add_argument(
#         "--video-length", type=int, default=300,
#         help="Maximum video length in steps (default: 300 = 10 seconds)"
#     )
#     parser.add_argument(
#         "--record-freq", type=int, default=50,
#         help="Record video every N iterations (default: 50)"
#     )
#     args = parser.parse_args()

#     # 실행 시간 기반 고유 ID 생성
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     run_name = f"ppo_franka_{timestamp}"
    
#     # 디렉토리 생성
#     os.makedirs(f"videos/{run_name}", exist_ok=True)
#     os.makedirs(f"runs/{run_name}/best_model", exist_ok=True)
#     os.makedirs(f"runs/{run_name}/logs", exist_ok=True)

#     # init 인자 처리
#     if args.init is not None and len(args.init) > 0:
#         init_qpos = args.init
#     else:
#         init_qpos = None
#         print("ℹ️  No initial pose provided, will try to use lap_start keyframe from XML")

#     # goal 인자 처리
#     if args.goal is not None:
#         goal_pose = args.goal
#     else:
#         goal_pose = None
#         print("ℹ️  No goal pose provided, using lap_end from XML")

#     # 1) 학습용 환경 생성
#     def make_train_env():
#         env = FrankaReachEnv(
#             xml_path=args.xml,
#             init_qpos=init_qpos,
#             goal_pose=goal_pose,
#             render_mode=None
#         )
#         env = Monitor(env, f"runs/{run_name}/logs")
#         return env
    
#     # 2) 평가용 환경 생성 함수
#     def make_eval_env():
#         env = FrankaReachEnv(
#             xml_path=args.xml,
#             init_qpos=init_qpos,
#             goal_pose=goal_pose,
#             render_mode="rgb_array"
#         )
#         env = Monitor(env)
#         return env
    
#     # 학습 환경 설정
#     train_env = DummyVecEnv([make_train_env])
#     train_env = VecNormalize(
#         train_env, 
#         norm_obs=True, 
#         norm_reward=True,  # reward normalization 활성화
#         clip_obs=10.0,
#         clip_reward=10.0
#     )
    
#     # 평가 환경 설정
#     eval_env = DummyVecEnv([make_eval_env])
#     eval_env = VecNormalize(
#         eval_env,
#         norm_obs=True,
#         norm_reward=False,
#         training=False,
#         clip_obs=10.0
#     )
    
#     # 3) 콜백 설정
#     # EvalCallback
#     eval_callback = EvalCallback(
#         eval_env,
#         best_model_save_path=f"runs/{run_name}/best_model",
#         log_path=f"runs/{run_name}/logs",
#         eval_freq=args.record_freq * 2048,  # record_freq iterations마다
#         n_eval_episodes=5,
#         deterministic=True,
#         verbose=1
#     )
    
#     # VideoRecorderCallback
#     video_callback = VideoRecorderCallback(
#         eval_env_fn=make_eval_env,
#         video_folder=f"videos/{run_name}",
#         video_length=args.video_length,
#         record_freq=args.record_freq,
#         run_name=run_name,
#         verbose=1
#     )
    
#     callbacks = CallbackList([eval_callback, video_callback])
    
#     # 4) PPO 모델 생성
#     model = PPO(
#         "MlpPolicy",
#         train_env,
#         learning_rate=3e-4,
#         n_steps=2048,
#         batch_size=64,
#         n_epochs=10,
#         gamma=0.99,
#         gae_lambda=0.95,
#         clip_range=0.2,
#         ent_coef=0.01,
#         vf_coef=0.5,
#         max_grad_norm=0.5,
#         verbose=1,
#         tensorboard_log=f"runs/{run_name}/tensorboard"
#     )
    
#     print("="*60)
#     print(f"PPO Training: {run_name}")
#     print(f"- Initial pose: {init_qpos if init_qpos else 'lap_start keyframe from XML'}")
#     print(f"- Goal pose: {goal_pose if goal_pose else 'lap_end site from XML'}")
#     print(f"- Recording video every {args.record_freq} iterations")
#     print(f"- Video length: {args.video_length} steps ({args.video_length/30:.1f} seconds)")
#     print(f"- Videos will be saved to: videos/{run_name}/")
#     print(f"- TensorBoard logs: runs/{run_name}/tensorboard")
#     print("="*60)
    
#     # 정규화 통계 동기화
#     eval_env.obs_rms = train_env.obs_rms
#     eval_env.ret_rms = train_env.ret_rms
    
#     try:
#         model.learn(
#             total_timesteps=500_000,
#             callback=callbacks,
#             log_interval=10,
#             tb_log_name=run_name
#         )
        
#         # 모델 및 정규화 통계 저장
#         model.save(f"runs/{run_name}/final_model")
#         train_env.save(f"runs/{run_name}/vec_normalize.pkl")
        
#         print(f"\n✅ Training completed successfully!")
#         print(f"🔸 Run name: {run_name}")
#         print(f"🔸 Final model: runs/{run_name}/final_model.zip")
#         print(f"🔸 Best model: runs/{run_name}/best_model/best_model.zip")
#         print(f"🔸 Normalization stats: runs/{run_name}/vec_normalize.pkl")
        
#         # 저장된 비디오 목록 출력
#         videos = sorted([f for f in os.listdir(f"videos/{run_name}") if f.endswith('.mp4')])
#         print(f"\n📹 Saved videos ({len(videos)} total):")
#         for video in videos[-5:]:  # 마지막 5개만 표시
#             print(f"   - {video}")
#         if len(videos) > 5:
#             print(f"   ... and {len(videos)-5} more")
        
#     except KeyboardInterrupt:
#         print("\n⚠️  Training interrupted by user")
#         model.save(f"runs/{run_name}/interrupted_model")
        
#     except Exception as e:
#         print(f"\n❌ Error during training: {e}")
#         raise
        
#     finally:
#         train_env.close()
#         eval_env.close()
#         print(f"\n💡 To monitor training progress, run:")
#         print(f"   tensorboard --logdir runs/{run_name}/tensorboard")




#------------------------------------------------------------------------------------------------------------------
# 2025.06.08 13:13 영상 생성까지 다됨. 근데 50에서 멈춤!
# import argparse
# import os
# import imageio
# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines3.common.monitor import Monitor
# from mujoco_rl_env.env.franka_reach_env import FrankaReachEnv

# class VideoRecorderCallback(BaseCallback):
#     """50 iteration마다 비디오를 저장하는 커스텀 콜백"""
#     def __init__(self, eval_env_fn, video_folder="videos", video_length=100, verbose=1):
#         super().__init__(verbose)
#         self.eval_env_fn = eval_env_fn
#         self.video_folder = video_folder
#         self.video_length = video_length
#         self.iteration_count = 0
#         os.makedirs(video_folder, exist_ok=True)
        
#     def _on_step(self) -> bool:
#         # 매 2048 스텝(1 iteration)마다 카운트
#         if self.n_calls % 2048 == 0:
#             self.iteration_count += 1
            
#             # 50 iteration마다 비디오 녹화
#             if self.iteration_count % 50 == 0:
#                 self._record_video()
                
#         return True
    
#     def _record_video(self):
#         """현재 정책으로 비디오 녹화"""
#         if self.verbose > 0:
#             print(f"\n🎬 Recording video at iteration {self.iteration_count}...")
        
#         # 평가용 환경 생성 (정규화 없이)
#         env = self.eval_env_fn()
        
#         # VecNormalize 래퍼가 있다면 정규화 통계 동기화
#         if hasattr(self.model, 'get_vec_normalize_env'):
#             vec_norm = self.model.get_vec_normalize_env()
#             if vec_norm is not None:
#                 eval_env_norm = VecNormalize(DummyVecEnv([lambda: env]), training=False)
#                 eval_env_norm.obs_rms = vec_norm.obs_rms
#                 eval_env_norm.ret_rms = vec_norm.ret_rms
#                 obs = eval_env_norm.reset()
#                 eval_env = eval_env_norm
#             else:
#                 obs = env.reset()
#         else:
#             obs = env.reset()
        
#         frames = []
#         episode_reward = 0
        
#         for step in range(self.video_length):
#             # 현재 정책으로 행동 선택
#             if isinstance(eval_env, VecNormalize):
#                 action, _ = self.model.predict(obs, deterministic=True)
#                 obs, reward, done, info = eval_env.step(action)
#                 # 실제 환경에서 프레임 가져오기
#                 frame = eval_env.venv.envs[0].render()
#             else:
#                 action, _ = self.model.predict(obs, deterministic=True)
#                 obs, reward, done, info = env.step(action)
#                 frame = env.render()
            
#             episode_reward += reward
            
#             if frame is not None:
#                 frames.append(frame)
            
#             if done:
#                 break
        
#         # 비디오 저장
#         if frames:
#             start_iter = self.iteration_count - 49
#             end_iter = self.iteration_count
#             filename = f"{self.video_folder}/iteration_{start_iter:03d}-{end_iter:03d}.mp4"
            
#             imageio.mimsave(filename, frames, fps=30)
            
#             if self.verbose > 0:
#                 print(f"✅ Video saved: {filename}")
#                 print(f"   Episode reward: {episode_reward:.2f}")
#                 print(f"   Episode length: {len(frames)}")
        
#         # 환경 정리
#         if isinstance(eval_env, VecNormalize):
#             eval_env.close()
#         else:
#             env.close()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Train PPO on the Franka reach environment"
#     )
#     parser.add_argument(
#         "--xml", default="models/fr3_reach.xml",
#         help="Path to your MuJoCo XML model"
#     )
#     parser.add_argument(
#         "--init", nargs='*', type=float,
#         default=None,  # None으로 기본값 설정
#         help="Initial joint positions (optional)"
#     )
#     parser.add_argument(
#         "--goal", nargs=7, type=float,
#         default=None,  # None으로 기본값 설정
#         help="Goal pose: x y z qx qy qz qw"
#     )
#     args = parser.parse_args()

#     # 디렉토리 생성
#     os.makedirs("videos", exist_ok=True)
#     os.makedirs("runs/best_model", exist_ok=True)
#     os.makedirs("runs/franka_reach", exist_ok=True)

#     # init 인자 처리
#     if args.init is not None and len(args.init) > 0:
#         init_qpos = args.init
#     else:
#         init_qpos = None  # 환경에서 기본값 사용
#         print("ℹ️  No initial pose provided, using lap_start from XML")

#     # goal 인자 처리
#     if args.goal is not None:
#         goal_pose = args.goal
#     else:
#         goal_pose = None  # 환경에서 lap_end 사용
#         print("ℹ️  No goal pose provided, using lap_end from XML")

#     # 1) 학습용 환경 생성 (Monitor로 감싸기)
#     def make_train_env():
#         env = FrankaReachEnv(
#             xml_path=args.xml,
#             init_qpos=init_qpos,
#             goal_pose=goal_pose,
#             render_mode=None
#         )
#         # Monitor로 감싸서 에피소드 통계 기록
#         env = Monitor(env)
#         return env
    
#     # 2) 평가용 환경 생성 함수
#     def make_eval_env():
#         env = FrankaReachEnv(
#             xml_path=args.xml,
#             init_qpos=init_qpos,
#             goal_pose=goal_pose,
#             render_mode="rgb_array"
#         )
#         # 평가 환경도 Monitor로 감싸기
#         env = Monitor(env)
#         return env
    
#     # 학습 환경 설정
#     train_env = DummyVecEnv([make_train_env])
#     train_env = VecNormalize(
#         train_env, 
#         norm_obs=True, 
#         norm_reward=False,
#         clip_obs=10.0
#     )
    
#     # 평가 환경 설정 (정규화 포함)
#     eval_env = DummyVecEnv([make_eval_env])
#     eval_env = VecNormalize(
#         eval_env,
#         norm_obs=True,
#         norm_reward=False,
#         training=False,  # 평가시에는 통계 업데이트 안함
#         clip_obs=10.0
#     )
    
#     # 3) 콜백 설정
#     # EvalCallback - TensorBoard에 eval 메트릭 기록
#     eval_callback = EvalCallback(
#         eval_env,
#         best_model_save_path="runs/best_model",
#         log_path="runs/franka_reach",
#         eval_freq=50 * 2048,  # 50 iteration마다
#         n_eval_episodes=5,     # 5 에피소드 평균
#         deterministic=True,
#         verbose=1
#     )
    
#     # VideoRecorderCallback - 비디오 녹화
#     video_callback = VideoRecorderCallback(
#         eval_env_fn=make_eval_env,
#         video_folder="videos",
#         video_length=100,
#         verbose=1
#     )
    
#     # 콜백 리스트로 묶기
#     callbacks = CallbackList([eval_callback, video_callback])
    
#     # 4) PPO 모델 생성 및 학습
#     model = PPO(
#         "MlpPolicy",
#         train_env,
#         learning_rate=3e-4,
#         n_steps=2048,
#         batch_size=64,
#         n_epochs=10,
#         gamma=0.99,
#         gae_lambda=0.95,
#         clip_range=0.2,
#         ent_coef=0.01,
#         vf_coef=0.5,
#         max_grad_norm=0.5,
#         verbose=1,
#         tensorboard_log="runs/franka_reach"
#     )
    
#     print("="*60)
#     print("PPO Training on Franka Reach Environment")
#     print(f"- Initial pose: {init_qpos if init_qpos else 'lap_start (from XML)'}")
#     print(f"- Goal pose: {goal_pose if goal_pose else 'lap_end (from XML)'}")
#     print(f"- Recording video every 50 iterations")
#     print(f"- Evaluating every 50 iterations (5 episodes)")
#     print(f"- Videos will be saved to: videos/")
#     print(f"- TensorBoard logs: runs/franka_reach")
#     print("="*60)
    
#     # 정규화 통계 동기화
#     eval_env.obs_rms = train_env.obs_rms
#     eval_env.ret_rms = train_env.ret_rms
    
#     try:
#         model.learn(
#             total_timesteps=500_000,
#             callback=callbacks,
#             log_interval=10,
#             tb_log_name="PPO"  # TensorBoard 로그 이름
#         )
        
#         # 모델 및 정규화 통계 저장
#         model.save("ppo_franka_reach")
#         train_env.save("runs/vec_normalize.pkl")
        
#         print("\n✅ Training completed successfully!")
#         print(f"- Model saved: ppo_franka_reach.zip")
#         print(f"- Normalization stats: runs/vec_normalize.pkl")
        
#         # 저장된 비디오 목록 출력
#         videos = sorted([f for f in os.listdir("videos") if f.endswith('.mp4')])
#         print(f"\n📹 Saved videos ({len(videos)} total):")
#         for video in videos:
#             print(f"   - {video}")
        
#     except KeyboardInterrupt:
#         print("\n⚠️  Training interrupted by user")
#         model.save("ppo_franka_reach_interrupted")
        
#     except Exception as e:
#         print(f"\n❌ Error during training: {e}")
#         raise
        
#     finally:
#         train_env.close()
#         eval_env.close()


#2025.06.08 13:02 - 영상 픽셀 이상
# import argparse
# import os
# import imageio
# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines3.common.monitor import Monitor
# from mujoco_rl_env.env.franka_reach_env import FrankaReachEnv

# class VideoRecorderCallback(BaseCallback):
#     """50 iteration마다 비디오를 저장하는 커스텀 콜백"""
#     def __init__(self, eval_env_fn, video_folder="videos", video_length=100, verbose=1):
#         super().__init__(verbose)
#         self.eval_env_fn = eval_env_fn
#         self.video_folder = video_folder
#         self.video_length = video_length
#         self.iteration_count = 0
#         os.makedirs(video_folder, exist_ok=True)
        
#     def _on_step(self) -> bool:
#         # 매 2048 스텝(1 iteration)마다 카운트
#         if self.n_calls % 2048 == 0:
#             self.iteration_count += 1
            
#             # 50 iteration마다 비디오 녹화
#             if self.iteration_count % 50 == 0:
#                 self._record_video()
                
#         return True
    
#     def _record_video(self):
#         """현재 정책으로 비디오 녹화"""
#         if self.verbose > 0:
#             print(f"\n🎬 Recording video at iteration {self.iteration_count}...")
        
#         # 평가용 환경 생성 (정규화 없이)
#         env = self.eval_env_fn()
        
#         # VecNormalize 래퍼가 있다면 정규화 통계 동기화
#         if hasattr(self.model, 'get_vec_normalize_env'):
#             vec_norm = self.model.get_vec_normalize_env()
#             if vec_norm is not None:
#                 eval_env_norm = VecNormalize(DummyVecEnv([lambda: env]), training=False)
#                 eval_env_norm.obs_rms = vec_norm.obs_rms
#                 eval_env_norm.ret_rms = vec_norm.ret_rms
#                 obs = eval_env_norm.reset()
#                 eval_env = eval_env_norm
#             else:
#                 obs = env.reset()
#         else:
#             obs = env.reset()
        
#         frames = []
#         episode_reward = 0
        
#         for step in range(self.video_length):
#             # 현재 정책으로 행동 선택
#             if isinstance(eval_env, VecNormalize):
#                 action, _ = self.model.predict(obs, deterministic=True)
#                 obs, reward, done, info = eval_env.step(action)
#                 # 실제 환경에서 프레임 가져오기
#                 frame = eval_env.venv.envs[0].render(mode="rgb_array")
#             else:
#                 action, _ = self.model.predict(obs, deterministic=True)
#                 obs, reward, done, info = env.step(action)
#                 frame = env.render(mode="rgb_array")
            
#             episode_reward += reward
            
#             if frame is not None:
#                 frames.append(frame)
            
#             if done:
#                 break
        
#         # 비디오 저장
#         if frames:
#             start_iter = self.iteration_count - 49
#             end_iter = self.iteration_count
#             filename = f"{self.video_folder}/iteration_{start_iter:03d}-{end_iter:03d}.mp4"
            
#             imageio.mimsave(filename, frames, fps=30)
            
#             if self.verbose > 0:
#                 print(f"✅ Video saved: {filename}")
#                 print(f"   Episode reward: {episode_reward:.2f}")
#                 print(f"   Episode length: {len(frames)}")
        
#         # 환경 정리
#         if isinstance(eval_env, VecNormalize):
#             eval_env.close()
#         else:
#             env.close()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Train PPO on the Franka reach environment"
#     )
#     parser.add_argument(
#         "--xml", default="models/fr3_reach.xml",
#         help="Path to your MuJoCo XML model"
#     )
#     parser.add_argument(
#         "--init", nargs='*', type=float,
#         default=None,  # None으로 기본값 설정
#         help="Initial joint positions (optional)"
#     )
#     parser.add_argument(
#         "--goal", nargs=7, type=float,
#         default=None,  # None으로 기본값 설정
#         help="Goal pose: x y z qx qy qz qw"
#     )
#     args = parser.parse_args()

#     # 디렉토리 생성
#     os.makedirs("videos", exist_ok=True)
#     os.makedirs("runs/best_model", exist_ok=True)
#     os.makedirs("runs/franka_reach", exist_ok=True)

#     # init 인자 처리
#     if args.init is not None and len(args.init) > 0:
#         init_qpos = args.init
#     else:
#         init_qpos = None  # 환경에서 기본값 사용
#         print("ℹ️  No initial pose provided, using lap_start from XML")

#     # goal 인자 처리
#     if args.goal is not None:
#         goal_pose = args.goal
#     else:
#         goal_pose = None  # 환경에서 lap_end 사용
#         print("ℹ️  No goal pose provided, using lap_end from XML")

#     # 1) 학습용 환경 생성 (Monitor로 감싸기)
#     def make_train_env():
#         env = FrankaReachEnv(
#             xml_path=args.xml,
#             init_qpos=init_qpos,
#             goal_pose=goal_pose,
#             render_mode=None
#         )
#         # Monitor로 감싸서 에피소드 통계 기록
#         env = Monitor(env)
#         return env
    
#     # 2) 평가용 환경 생성 함수
#     def make_eval_env():
#         env = FrankaReachEnv(
#             xml_path=args.xml,
#             init_qpos=init_qpos,
#             goal_pose=goal_pose,
#             render_mode="rgb_array"
#         )
#         # 평가 환경도 Monitor로 감싸기
#         env = Monitor(env)
#         return env
    
#     # 학습 환경 설정
#     train_env = DummyVecEnv([make_train_env])
#     train_env = VecNormalize(
#         train_env, 
#         norm_obs=True, 
#         norm_reward=False,
#         clip_obs=10.0
#     )
    
#     # 평가 환경 설정 (정규화 포함)
#     eval_env = DummyVecEnv([make_eval_env])
#     eval_env = VecNormalize(
#         eval_env,
#         norm_obs=True,
#         norm_reward=False,
#         training=False,  # 평가시에는 통계 업데이트 안함
#         clip_obs=10.0
#     )
    
#     # 3) 콜백 설정
#     # EvalCallback - TensorBoard에 eval 메트릭 기록
#     eval_callback = EvalCallback(
#         eval_env,
#         best_model_save_path="runs/best_model",
#         log_path="runs/franka_reach",
#         eval_freq=50 * 2048,  # 50 iteration마다
#         n_eval_episodes=5,     # 5 에피소드 평균
#         deterministic=True,
#         verbose=1
#     )
    
#     # VideoRecorderCallback - 비디오 녹화
#     video_callback = VideoRecorderCallback(
#         eval_env_fn=make_eval_env,
#         video_folder="videos",
#         video_length=100,
#         verbose=1
#     )
    
#     # 콜백 리스트로 묶기
#     callbacks = CallbackList([eval_callback, video_callback])
    
#     # 4) PPO 모델 생성 및 학습
#     model = PPO(
#         "MlpPolicy",
#         train_env,
#         learning_rate=3e-4,
#         n_steps=2048,
#         batch_size=64,
#         n_epochs=10,
#         gamma=0.99,
#         gae_lambda=0.95,
#         clip_range=0.2,
#         ent_coef=0.01,
#         vf_coef=0.5,
#         max_grad_norm=0.5,
#         verbose=1,
#         tensorboard_log="runs/franka_reach"
#     )
    
#     print("="*60)
#     print("PPO Training on Franka Reach Environment")
#     print(f"- Initial pose: {init_qpos if init_qpos else 'lap_start (from XML)'}")
#     print(f"- Goal pose: {goal_pose if goal_pose else 'lap_end (from XML)'}")
#     print(f"- Recording video every 50 iterations")
#     print(f"- Evaluating every 50 iterations (5 episodes)")
#     print(f"- Videos will be saved to: videos/")
#     print(f"- TensorBoard logs: runs/franka_reach")
#     print("="*60)
    
#     # 정규화 통계 동기화
#     eval_env.obs_rms = train_env.obs_rms
#     eval_env.ret_rms = train_env.ret_rms
    
#     try:
#         model.learn(
#             total_timesteps=500_000,
#             callback=callbacks,
#             log_interval=10,
#             tb_log_name="PPO"  # TensorBoard 로그 이름
#         )
        
#         # 모델 및 정규화 통계 저장
#         model.save("ppo_franka_reach")
#         train_env.save("runs/vec_normalize.pkl")
        
#         print("\n✅ Training completed successfully!")
#         print(f"- Model saved: ppo_franka_reach.zip")
#         print(f"- Normalization stats: runs/vec_normalize.pkl")
        
#         # 저장된 비디오 목록 출력
#         videos = sorted([f for f in os.listdir("videos") if f.endswith('.mp4')])
#         print(f"\n📹 Saved videos ({len(videos)} total):")
#         for video in videos:
#             print(f"   - {video}")
        
#     except KeyboardInterrupt:
#         print("\n⚠️  Training interrupted by user")
#         model.save("ppo_franka_reach_interrupted")
        
#     except Exception as e:
#         print(f"\n❌ Error during training: {e}")
#         raise
        
#     finally:
#         train_env.close()
#         eval_env.close()



# 2025.06.08 12:42 - 비디오 깨짐 현상 
# import argparse
# import os
# import imageio
# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from mujoco_rl_env.env.franka_reach_env import FrankaReachEnv

# class VideoRecorderCallback(BaseCallback):
#     """50 iteration마다 비디오를 저장하는 커스텀 콜백"""
#     def __init__(self, eval_env_fn, video_folder="videos", video_length=100, verbose=1):
#         super().__init__(verbose)
#         self.eval_env_fn = eval_env_fn
#         self.video_folder = video_folder
#         self.video_length = video_length
#         self.iteration_count = 0
#         os.makedirs(video_folder, exist_ok=True)
        
#     def _on_step(self) -> bool:
#         # 매 2048 스텝(1 iteration)마다 카운트
#         if self.n_calls % 2048 == 0:
#             self.iteration_count += 1
            
#             # 50 iteration마다 비디오 녹화
#             if self.iteration_count % 50 == 0:
#                 self._record_video()
                
#         return True
    
#     def _record_video(self):
#         """현재 정책으로 비디오 녹화"""
#         if self.verbose > 0:
#             print(f"\n🎬 Recording video at iteration {self.iteration_count}...")
        
#         # 평가용 환경 생성
#         env = self.eval_env_fn()
#         obs = env.reset()
        
#         frames = []
#         episode_reward = 0
        
#         for step in range(self.video_length):
#             # 현재 정책으로 행동 선택
#             action, _ = self.model.predict(obs, deterministic=True)
#             obs, reward, done, info = env.step(action)
#             episode_reward += reward
            
#             # 프레임 캡처
#             frame = env.render(mode="rgb_array")
#             if frame is not None:
#                 frames.append(frame)
            
#             if done:
#                 break
        
#         # 비디오 저장
#         if frames:
#             start_iter = self.iteration_count - 49
#             end_iter = self.iteration_count
#             filename = f"{self.video_folder}/iteration_{start_iter:03d}-{end_iter:03d}.mp4"
            
#             imageio.mimsave(filename, frames, fps=30)
            
#             if self.verbose > 0:
#                 print(f"✅ Video saved: {filename}")
#                 print(f"   Episode reward: {episode_reward:.2f}")
#                 print(f"   Episode length: {len(frames)}")
        
#         env.close()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Train PPO on the Franka reach environment"
#     )
#     parser.add_argument(
#         "--xml", default="models/fr3_reach.xml",
#         help="Path to your MuJoCo XML model"
#     )
#     parser.add_argument(
#         "--init", nargs='*', type=float,
#         default=None,  # None으로 기본값 설정
#         help="Initial joint positions (optional)"
#     )
#     parser.add_argument(
#         "--goal", nargs=7, type=float,
#         help="Goal pose: x y z qx qy qz qw"
#     )
#     args = parser.parse_args()

#     # 디렉토리 생성
#     os.makedirs("videos", exist_ok=True)
#     os.makedirs("runs/best_model", exist_ok=True)
#     os.makedirs("runs/franka_reach", exist_ok=True)

#     # init 인자 처리
#     if args.init is not None and len(args.init) > 0:
#         init_qpos = args.init
#     else:
#         init_qpos = None  # 환경에서 기본값 사용
#         print("ℹ️  No initial pose provided, using environment defaults")

#     # 1) 학습용 환경 생성
#     def make_train_env():
#         return FrankaReachEnv(
#             xml_path=args.xml,
#             init_qpos=init_qpos,
#             goal_pose=args.goal,
#             render_mode=None
#         )
    
#     # 2) 평가용 환경 생성 함수
#     def make_eval_env():
#         env = FrankaReachEnv(
#             xml_path=args.xml,
#             init_qpos=init_qpos,
#             goal_pose=args.goal,
#             render_mode="rgb_array"
#         )
#         return env
    
#     # 학습 환경 설정
#     train_env = DummyVecEnv([make_train_env])
#     train_env = VecNormalize(
#         train_env, 
#         norm_obs=True, 
#         norm_reward=False,
#         clip_obs=10.0
#     )
    
#     # 3) 비디오 녹화 콜백
#     video_callback = VideoRecorderCallback(
#         eval_env_fn=make_eval_env,
#         video_folder="videos",
#         video_length=100,
#         verbose=1
#     )
    
#     # 4) PPO 모델 생성 및 학습
#     model = PPO(
#         "MlpPolicy",
#         train_env,
#         learning_rate=3e-4,
#         n_steps=2048,
#         batch_size=64,
#         n_epochs=10,
#         gamma=0.99,
#         gae_lambda=0.95,
#         clip_range=0.2,
#         ent_coef=0.01,
#         vf_coef=0.5,
#         max_grad_norm=0.5,
#         verbose=1,
#         tensorboard_log="runs/franka_reach"
#     )
    
#     print("="*60)
#     print("PPO Training on Franka Reach Environment")
#     print(f"- Initial pose: {init_qpos if init_qpos else 'Default (zeros)'}")
#     print(f"- Goal pose: {args.goal}")
#     print(f"- Recording video every 50 iterations")
#     print(f"- Videos will be saved to: videos/")
#     print("="*60)
    
#     try:
#         model.learn(
#             total_timesteps=500_000,
#             callback=video_callback,
#             log_interval=10
#         )
        
#         # 모델 및 정규화 통계 저장
#         model.save("ppo_franka_reach")
#         train_env.save("runs/vec_normalize.pkl")
        
#         print("\n✅ Training completed successfully!")
#         print(f"- Model saved: ppo_franka_reach.zip")
#         print(f"- Normalization stats: runs/vec_normalize.pkl")
        
#         # 저장된 비디오 목록 출력
#         videos = sorted([f for f in os.listdir("videos") if f.endswith('.mp4')])
#         print(f"\n📹 Saved videos ({len(videos)} total):")
#         for video in videos:
#             print(f"   - {video}")
        
#     except KeyboardInterrupt:
#         print("\n⚠️  Training interrupted by user")
#         model.save("ppo_franka_reach_interrupted")
        
#     except Exception as e:
#         print(f"\n❌ Error during training: {e}")
#         raise
        
#     finally:
#         train_env.close()

'''2025.06.07 23:49 이전 작업 - 클로드 전
#/home/minjun/rl_ws/src/mujoco_rl_env/train.py
import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize #normalized
from stable_baselines3.common.vec_env import VecVideoRecorder #video저장
from mujoco_rl_env.env.franka_reach_env import FrankaReachEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO on the Franka reach environment"
    )
    parser.add_argument(
        "--xml", default="models/fr3_reach.xml",
        help="Path to your MuJoCo XML model"
    )
    parser.add_argument(
        "--init", nargs=14, type=float,
        default=[0]*14,
        help="Initial joint positions (14 values)"
    )
    parser.add_argument(
        "--goal", nargs=7, type=float,
        help="Goal pose: x y z qx qy qz qw"
    )
    args = parser.parse_args()

    # 디렉토리 생성
    os.makedirs("videos", exist_ok=True)
    os.makedirs("runs/best_model", exist_ok=True)
    os.makedirs("runs/franka_reach", exist_ok=True)

    # 1) 학습용 환경 생성 (render_mode 없음)
    def make_train_env():
        return FrankaReachEnv(
            xml_path=args.xml,
            init_qpos=args.init,  # 전체 14개 값 모두 전달
            goal_pose=args.goal,
            render_mode=None  # 학습 중에는 렌더링 불필요
        )
    
    train_env = DummyVecEnv([make_train_env])
    
    # 2) 평가/녹화용 환경 생성 - 가장 중요한 부분!
    def make_eval_env():
        env = FrankaReachEnv(
            xml_path=args.xml,
            init_qpos=args.init,  # 전체 14개 값 모두 전달
            goal_pose=args.goal,
            render_mode="rgb_array"  # 비디오 녹화를 위해 필수
        )
        # 환경이 render_mode 속성을 갖도록 명시적으로 설정
        env.render_mode = "rgb_array"
        return env
    
    eval_env = DummyVecEnv([make_eval_env])
    
    # 핵심 수정: DummyVecEnv에 render_mode 속성 수동 추가!!!
    eval_env.render_mode = "rgb_array"
    
    # 3) 비디오 녹화 래퍼 적용
    record_freq = 50 * 2048  # 50 iteration마다
    video_eval_env = VecVideoRecorder(
        eval_env,
        video_folder="videos/",
        record_video_trigger=lambda x: x % record_freq == 0,
        video_length=100,  # 에피소드 길이
        name_prefix="franka_reach"
    )
    
    # 4) 정규화 래퍼 (학습 환경만)
    train_env = VecNormalize(
        train_env, 
        norm_obs=True, 
        norm_reward=False,
        clip_obs=10.0
    )
    
    # 평가 환경도 정규화 (training=False)
    eval_env_normalized = VecNormalize(
        video_eval_env,  # 비디오 래퍼 위에 정규화 래퍼
        norm_obs=True,
        norm_reward=False,
        training=False,
        clip_obs=10.0
    )
    
    # 5) 평가 콜백 설정
    eval_callback = EvalCallback(
        eval_env_normalized,
        best_model_save_path="runs/best_model",
        log_path="runs/franka_reach",
        eval_freq=record_freq,  # 50 iteration마다
        n_eval_episodes=1,      # 평가 시 1 에피소드만
        deterministic=True,
        verbose=1
    )
    
    # 6) PPO 모델 생성 및 학습
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="runs/franka_reach"
    )
    
    print("="*60)
    print("PPO Training on Franka Reach Environment")
    print(f"- Training with {train_env.num_envs} environment(s)")
    print(f"- Recording video every {record_freq} steps (50 iterations)")
    print(f"- Videos will be saved to: videos/")
    print(f"- Tensorboard logs: runs/franka_reach")
    print("="*60)
    
    try:
        model.learn(
            total_timesteps=500_000,
            callback=eval_callback,
            log_interval=10
        )
        
        # 모델 및 정규화 통계 저장
        model.save("ppo_franka_reach")
        train_env.save("runs/vec_normalize.pkl")
        
        print("\n✅ Training completed successfully!")
        print(f"- Model saved: ppo_franka_reach.zip")
        print(f"- Normalization stats: runs/vec_normalize.pkl")
        print(f"- Videos saved in: videos/")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        model.save("ppo_franka_reach_interrupted")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise
        
    finally:
        # 환경 정리
        train_env.close()
        eval_env_normalized.close()
'''
# # shapr 7인지 14인지에 대한 문제
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Train PPO on the Franka reach environment"
#     )
#     parser.add_argument(
#         "--xml", default="models/fr3_reach.xml",
#         help="Path to your MuJoCo XML model"
#     )
#     parser.add_argument(
#         "--init", nargs=14, type=float,
#         default=[0]*14,  # 14개로 수정
#         help="Initial joint positions (14 values)"
#     )
#     parser.add_argument(
#         "--goal", nargs=7, type=float,
#         help="Goal pose: x y z qx qy qz qw"
#     )
#     args = parser.parse_args()

#     # 디렉토리 생성
#     os.makedirs("videos", exist_ok=True)
#     os.makedirs("runs/best_model", exist_ok=True)
#     os.makedirs("runs/franka_reach", exist_ok=True)

#     # 1) 학습용 환경 생성
#     def make_train_env():
#         return FrankaReachEnv(
#             xml_path=args.xml,
#             init_qpos=args.init[:7],  # 처음 7개만 사용
#             goal_pose=args.goal,
#             render_mode=None
#         )
    
#     # 2) 평가용 환경 생성 (비디오 녹화용)
#     def make_eval_env():
#         return FrankaReachEnv(
#             xml_path=args.xml,
#             init_qpos=args.init[:7],  # 처음 7개만 사용
#             goal_pose=args.goal,
#             render_mode="rgb_array"
#         )
    
#     # 환경 래핑
#     train_env = DummyVecEnv([make_train_env])
#     eval_env = DummyVecEnv([make_eval_env])
    
#     # 중요: DummyVecEnv에 render_mode 속성 추가
#     eval_env.render_mode = "rgb_array"
    
#     # 3) 비디오 녹화 래퍼
#     record_freq = 50 * 2048  # 50 iteration마다
#     video_eval_env = VecVideoRecorder(
#         eval_env,
#         video_folder="videos/",
#         record_video_trigger=lambda x: x % record_freq == 0,
#         video_length=100,  # 최대 100 스텝
#         name_prefix="franka_reach"
#     )
    
#     # 4) 정규화 래퍼
#     train_env = VecNormalize(
#         train_env, 
#         norm_obs=True, 
#         norm_reward=False,
#         clip_obs=10.0
#     )
    
#     eval_env_normalized = VecNormalize(
#         video_eval_env,
#         norm_obs=True,
#         norm_reward=False,
#         training=False,
#         clip_obs=10.0
#     )
    
#     # 5) 평가 콜백
#     eval_callback = EvalCallback(
#         eval_env_normalized,
#         best_model_save_path="runs/best_model",
#         log_path="runs/franka_reach",
#         eval_freq=record_freq,
#         n_eval_episodes=1,
#         deterministic=True,
#         verbose=1
#     )
    
#     # 6) PPO 모델 생성
#     model = PPO(
#         "MlpPolicy",
#         train_env,
#         learning_rate=3e-4,
#         n_steps=2048,
#         batch_size=64,
#         n_epochs=10,
#         gamma=0.99,
#         gae_lambda=0.95,
#         clip_range=0.2,
#         ent_coef=0.01,
#         vf_coef=0.5,
#         max_grad_norm=0.5,
#         verbose=1,
#         tensorboard_log="runs/franka_reach",
#         device="auto"  # GPU 사용 가능시 자동 선택
#     )
    
#     print("="*50)
#     print("Training PPO on Franka Reach Environment")
#     print(f"Recording videos every {record_freq} steps (50 iterations)")
#     print(f"Videos will be saved to: videos/")
#     print("="*50)
    
#     try:
#         # 학습 시작
#         model.learn(
#             total_timesteps=500_000,
#             callback=eval_callback,
#             log_interval=10
#         )
        
#         # 모델 저장
#         model.save("ppo_franka_reach")
#         train_env.save("runs/vec_normalize.pkl")
        
#         print("\nTraining completed successfully!")
#         print(f"Model saved as: ppo_franka_reach.zip")
#         print(f"Normalization stats saved as: runs/vec_normalize.pkl")
        
#     except Exception as e:
#         print(f"\nError during training: {e}")
#         raise
    
#     finally:
#         # 환경 정리
#         train_env.close()
#         eval_env_normalized.close()



# class VideoRecordCallback(EvalCallback):
#     """ 50 iteration 마다 비디오 녹화하는 커스텀 콜백 """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.iteration_count = 0
        
#     def _on_step(self) -> bool:
#         result = super()._on_step()

#         if self.n_calls % self.model.n_steps == 0:
#             self.iteration_count += 1
        
#         return result

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Train PPO on the Franka reach environment"
#     )
#     parser.add_argument(
#         "--xml", default="models/fr3_reach.xml",
#         help="Path to your MuJoCo XML model"
#     )
#     parser.add_argument(
#         "--init", nargs=14, type=float,
#         default=[0]*7,
#         help="Initial joint positions (7 values)"
#     )
#     parser.add_argument(
#         "--goal", nargs=7, type=float,
#         help="Goal pose: x y z qx qy qz qw"
#     )
#     args = parser.parse_args()

#     os.makedirs("videos", exist_ok=True)
#     os.makedirs("runs/best_model", exist_ok=True)
#     os.makedirs("runs/franka_reach", exist_ok=True)
    
#     # env = FrankaReachEnv(
#     #     xml_path=args.xml,
#     #     init_qpos=args.init,
#     #     goal_pose=args.goal
#     # )
#     # 기존엔 model = PPO~부터만 있었음
#     # 1) 학습용/평가용 env 래핑
#     '''
#     # DummyVecEnv: 벡터화된 환경 래퍼 -> VecNormalize로 관측/보상 정규화 -> VecVideoRecorder로 비디오 녹화
#     train_env = DummyVecEnv([lambda: env])
#     eval_env  = DummyVecEnv([lambda: FrankaReachEnv(
#         xml_path=args.xml,
#         init_qpos=args.init,
#         goal_pose=args.goal,  # 고정 goal으로 평가
#         render_mode="rgb_array"   # ← 이걸 반드시 지정해야 비디오 녹화 가능
#     )])

#     #2) 관측만 정규화 (학습 보상은 그대로)
#     train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)
#     eval_env  = VecNormalize(eval_env,  norm_obs=True, norm_reward=False,
#                               training=False)
#     # 2.1) 비디오 녹화: 50 iteration 마다 한 번씩, 한 에피소드 길이 만큼 저장
#     #   step: 전체 timesteps, so record when step % (2048*50) == 0
#     record_freq = 2048 * 50
#     video_folder = "videos/"
#     eval_env = VecVideoRecorder(
#         eval_env,
#         video_folder=video_folder,
#         record_video_trigger=lambda step: step % record_freq == 0,
#         video_length=env.max_steps,   # 한 에피소드 길이(100 steps)만큼 녹화
#         name_prefix="franka_reach"
#     )    
#     '''
#     # video 녹화 먼저
#     # → 1) DummyVecEnv → 2) VecVideoRecorder → 3) VecNormalize

#     # 학습용 환경 (렌더링 없음)
#     train_env = DummyVecEnv([lambda: FrankaReachEnv(
#         xml_path=args.xml,
#         init_qpos=args.init,
#         goal_pose=args.goal,
#         render_mode=None  # 학습용 env는 렌더링 안함
#     )])

#     # 평가용 환경 (렌더링 있음)
#     eval_env = DummyVecEnv([lambda: FrankaReachEnv(
#         xml_path=args.xml,
#         init_qpos=args.init,
#         goal_pose=args.goal,  # 고정 goal으로 평가
#         render_mode="rgb_array"  # 비디오 녹화를 위해 렌더링 모드 지정
#     )])


#     # raw_eval_env = DummyVecEnv([lambda: FrankaReachEnv(
#     #     xml_path=args.xml,
#     #     init_qpos=args.init,
#     #     goal_pose=args.goal,
#     #     render_mode="rgb_array"       # ← 여기서 미리 지정
#     # )])

#     # DummyVecEnv(…) 자체에도 render_mode 부착!
#     #raw_eval_env.render_mode = "rgb_array"
#     # (2) 비디오 녹화 래퍼 먼저
#     # 2.1) 비디오 녹화: 50 iteration 마다 한 번씩, 한 에피소드 길이 만큼 저장
#     #   step: 전체 timesteps, so record when step % (2048*50) == 0
#     record_freq = 2048 * 50
#     video_folder = "videos/"
#     video_eval_env = VecVideoRecorder(
#         eval_env,
#         video_folder=video_folder,
#         record_video_trigger=lambda step: step % record_freq == 0,
#         video_length=env.max_steps, #100
#         name_prefix="franka_reach"
#     )
#     # (3) 그리고 나서 정규화 래퍼 (학습환경만)
#     train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)
    
#     #평가 환경도 정규화 (학습하지 않음)
#     eval_env_normalized  = VecNormalize(
#         video_eval_env,
#         norm_obs=True,
#         norm_reward=False,
#         training=False
#     )

    
#     # 3) EvalCallback: 50 iters(≈50*2048 steps)마다 평가 기록
#     eval_callback = VideoRecordCallback(
#         eval_env_normalized,
#         best_model_save_path="runs/best_model",
#         log_path="runs/franka_reach",
#         eval_freq=record_freq,
#         n_eval_episodes=1,
#         deterministic=True,
#         verbose = 1
#     )

#     # eval_callback = EvalCallback(
#     #     eval_env,
#     #     best_model_save_path="runs/best_model",
#     #     log_path="runs/franka_reach",
#     #     eval_freq=2048*50,
#     #     n_eval_episodes=5,
#     #     deterministic=True,
#     # )
#     # ==== 변경된 부분 끝 ====
#     # 기존엔 train_env대신 env만 사용했음
#     # 이후 model.learn(total_timesteps=500_000) 부분까지 동일 , callback은없었음
#     # 그냥 model = PPO("MlpPolicy", env, verbose=1)로만 사용했으면 -> hyperparameter 기본 세팅
#     # learning_rate	3×10⁻⁴ n_steps	2048 batch_size	64 _epochs	10 gamma	0.99 gae_lambda	0.95 clip_range	0.2 ent_coef	0.0 vf_coef	0.5 max_grad_norm	0.5
#     '''
#     model = PPO(
#         "MlpPolicy", train_env,
#         verbose=1,
#         tensorboard_log="runs/franka_reach"
#     )'''

#     # 4) PPO 모델 생성
#     model = PPO(
#         "MlpPolicy", train_env,``
#         learning_rate=3e-5,
#         n_steps=2048,
#         batch_size=128,
#         n_epochs=15,
#         gamma=0.99,
#         gae_lambda=0.95,
#         clip_range=0.2,
#         ent_coef=0.001,
#         vf_coef=0.5,
#         max_grad_norm=0.5,
#         verbose=1,
#         tensorboard_log="runs/franka_reach"
#     )

#     print("Starting training with video recording every 50 iterations...")
#     model.learn(
#         total_timesteps=500_000,
#         callback=eval_callback
#     )

#     # 모델 저장
#     model.save("ppo_franka_reach")
    
#     # 정규화 통계 저장
#     train_env.save("runs/vec_normalize.pkl")
    
#     print("Training completed!")
#     print("Videos saved in: videos/")
#     print("Model saved as: ppo_franka_reach.zip")