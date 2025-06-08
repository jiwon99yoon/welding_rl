# #!/usr/bin/env python3
# """
# 각 PPO 실행 디렉토리(runs/ppo_franka_*/)를 찾아,
#  - best_model.zip 또는 final_model.zip 을 로드하고
#  - vec_normalize.pkl(정규화 통계)이 있으면 함께 로드
#  - 한 에피소드 동안 첫 goal 도달 시점까지 시뮬레이션
#  - 프레임을 수집하여 mp4로 저장

# 출력 디렉토리: ~/rl_ws/goal_reached_step_mp4
# 파일명 형식: <run_name>_goal_reached.mp4
# """
# import os
# import argparse
# import imageio
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines3.common.monitor import Monitor
# from mujoco_rl_env.env.franka_reach_env import FrankaReachEnv

# def make_eval_env(xml_path, init_qpos, goal_qpos, render_mode="rgb_array"):
#     """평가용 단일 환경 팩토리 함수."""
#     def _init():
#         env = FrankaReachEnv(
#             xml_path=xml_path,
#             init_qpos=init_qpos,
#             goal_qpos=goal_qpos,
#             render_mode=render_mode
#         )
#         return Monitor(env)
#     return _init

# def find_model_and_stats(run_dir):
#     """
#     run_dir 하위에서
#      - best_model/best_model.zip 또는 final_model.zip
#      - vec_normalize.pkl
#     의 경로를 찾아 반환.
#     """
#     best = os.path.join(run_dir, "best_model", "best_model.zip")
#     final = os.path.join(run_dir, "final_model.zip")
#     vec   = os.path.join(run_dir, "vec_normalize.pkl")
#     model_path = best if os.path.isfile(best) else (final if os.path.isfile(final) else None)
#     stats_path = vec if os.path.isfile(vec) else None
#     return model_path, stats_path

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--runs-dir", default=os.path.expanduser("~/rl_ws/runs"),
#                    help="학습된 PPO 모델들이 들어있는 runs/ 디렉토리")
#     p.add_argument("--out-dir",  default=os.path.expanduser("~/rl_ws/goal_reached_step_mp4"),
#                    help="goal 도달 동영상을 저장할 출력 디렉토리")
#     p.add_argument("--xml",      default="models/fr3_reach.xml",
#                    help="MuJoCo XML 모델 경로")
#     p.add_argument("--init", nargs="*", type=float, default=None,
#                    help="초기 관절 위치 (옵션)")
#     p.add_argument("--goal", nargs="*", type=float, default=None,
#                    help="목표 관절 위치 (옵션)")
#     p.add_argument("--max-steps", type=int, default=300,
#                    help="한 에피소드 최대 시뮬레이션 스텝 수")
#     p.add_argument("--fps",       type=int, default=30,
#                    help="저장할 동영상 FPS")
#     args = p.parse_args()

#     os.makedirs(args.out_dir, exist_ok=True)

#     # runs/ 디렉토리 내 'ppo_franka_' 로 시작하는 모든 폴더 순회
#     for run_name in sorted(os.listdir(args.runs_dir)):
#         if not run_name.startswith("ppo_franka_"):
#             continue
#         run_path = os.path.join(args.runs_dir, run_name)
#         if not os.path.isdir(run_path):
#             continue

#         # 모델과 정규화 통계 경로 찾기
#         model_file, stats_file = find_model_and_stats(run_path)
#         if model_file is None:
#             print(f"[SKIP] {run_name} 에서 모델을 찾을 수 없습니다.")
#             continue

#         # 평가 환경 생성
#         eval_env = DummyVecEnv([ make_eval_env(
#             xml_path=args.xml,
#             init_qpos=args.init,
#             goal_qpos=args.goal,
#             render_mode="rgb_array"
#         )])
#         # 정규화 로드 또는 끄기
#         if stats_file:
#             eval_env = VecNormalize.load(stats_file, eval_env)
#         else:
#             eval_env = VecNormalize(eval_env, training=False)

#         # 정책 로드
#         model = PPO.load(model_file, env=eval_env)

#         # 첫 goal 도달까지만 롤아웃
#         obs = eval_env.reset()
#         frames = []
#         for step in range(args.max_steps):
#             action, _ = model.predict(obs, deterministic=True)
#             #obs, _, terminated, truncated, info = eval_env.step(action)
#             obs, reward, done, info = eval_env.step(action)
#             # 실제 env의 화면을 얻어옴
#             frame = eval_env.venv.envs[0].render()
#             frames.append(frame)
#             # success 정보가 있거나 terminated=True 이면 중단
#             if done:
#                 break
#             # info['success'] 는 step() 안에서 goal error < threshold 시 True 로 설정해둔 값
#             if info.get("success", False):
#                 print(f"✅ 진짜 goal reached at step {step}")
#                 break
#         # end for

#         # goal reached 한 영상이 아니라면 저장하지 않거나, 성공한 에피소드가 나올 때까지 다시 뽑기
#         if info.get("success", False):
#             out_path = os.path.join(args.out_dir, f"{run_name}_goal_reached.mp4")
#             imageio.mimsave(out_path, frames, fps=30)
#             print(f"Saved goal-reached video to {out_path}")
#         else:
#             print("⚠️ goal에 실제로 도달하지 못해 영상 저장 안 함")        
        
#         if not frames:
#             print(f"[{run_name}] 녹화할 프레임이 없습니다.")
#             continue

#         # 동영상 저장
#         out_path = os.path.join(args.out_dir, f"{run_name}_goal_reached.mp4")
#         imageio.mimsave(out_path, frames, fps=args.fps)
#         print(f"[저장됨] {out_path}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# """
# 각 PPO 실행 디렉토리(runs/ppo_franka_*/)를 찾아,
#  - best_model.zip 또는 final_model.zip 을 로드하고
#  - vec_normalize.pkl(정규화 통계)이 있으면 함께 로드
#  - 한 에피소드 동안 첫 goal 도달 시점까지 시뮬레이션
#  - 프레임을 수집하여 mp4로 저장

# 출력 디렉토리: ~/rl_ws/goal_reached_step_mp4
# 파일명 형식: <run_name>_goal_reached.mp4 (기존 파일이 있으면 생성 안 함)
# """
# import os
# import argparse
# import imageio
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines3.common.monitor import Monitor
# from mujoco_rl_env.env.franka_reach_env import FrankaReachEnv

# def make_eval_env(xml_path, init_qpos, goal_qpos, render_mode="rgb_array"):
#     def _init():
#         env = FrankaReachEnv(
#             xml_path=xml_path,
#             init_qpos=init_qpos,
#             goal_qpos=goal_qpos,
#             render_mode=render_mode
#         )
#         return Monitor(env)
#     return _init

# def find_model_and_stats(run_dir):
#     best  = os.path.join(run_dir, "best_model", "best_model.zip")
#     final = os.path.join(run_dir, "final_model.zip")
#     vec   = os.path.join(run_dir, "vec_normalize.pkl")
#     model_path = best if os.path.isfile(best) else final if os.path.isfile(final) else None
#     stats_path = vec if os.path.isfile(vec) else None
#     return model_path, stats_path

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--runs-dir", default=os.path.expanduser("~/rl_ws/runs"))
#     p.add_argument("--out-dir",  default=os.path.expanduser("~/rl_ws/goal_reached_step_mp4"))
#     p.add_argument("--xml",      default="models/fr3_reach.xml")
#     p.add_argument("--init", nargs="*", type=float, default=None)
#     p.add_argument("--goal", nargs="*", type=float, default=None)
#     p.add_argument("--max-steps", type=int, default=300)
#     p.add_argument("--fps",       type=int, default=30)
#     args = p.parse_args()

#     os.makedirs(args.out_dir, exist_ok=True)

#     for run_name in sorted(os.listdir(args.runs_dir)):
#         if not run_name.startswith("ppo_franka_"):
#             continue
#         run_path = os.path.join(args.runs_dir, run_name)
#         if not os.path.isdir(run_path):
#             continue

#         out_path = os.path.join(args.out_dir, f"{run_name}_goal_reached.mp4")
#         if os.path.isfile(out_path):
#             print(f"[SKIP] {run_name}: 이미 영상이 존재합니다 → {out_path}")
#             continue

#         model_file, stats_file = find_model_and_stats(run_path)
#         if model_file is None:
#             print(f"[SKIP] {run_name}: 모델을 찾을 수 없습니다.")
#             continue

#         # 평가 환경 준비
#         eval_env = DummyVecEnv([ make_eval_env(args.xml, args.init, args.goal) ])
#         if stats_file:
#             eval_env = VecNormalize.load(stats_file, eval_env)
#         else:
#             eval_env = VecNormalize(eval_env, training=False)

#         model = PPO.load(model_file, env=eval_env)

#         # 롤아웃 및 프레임 수집
#         obs = eval_env.reset()
#         frames = []
#         success = False
#         for step in range(args.max_steps):
#             action, _ = model.predict(obs, deterministic=True)
#             obs, reward, done, info = eval_env.step(action)
#             frames.append(eval_env.venv.envs[0].render())
#             if info.get("success", False):
#                 print(f"[{run_name}] goal reached at step {step}")
#                 success = True
#                 break
#             if done:
#                 break

#         if not frames:
#             print(f"[{run_name}] 녹화할 프레임이 없습니다.")
#             continue

#         if success:
#             imageio.mimsave(out_path, frames, fps=args.fps)
#             print(f"[저장됨] {out_path}")
#         else:
#             print(f"[경고] {run_name}: goal에 도달하지 못해 영상 저장 안 함")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# """
# 각 PPO 실행 디렉토리(runs/ppo_franka_*/)를 찾아,
#  - best_model.zip 또는 final_model.zip 을 로드하고
#  - vec_normalize.pkl(정규화 통계)이 있으면 함께 로드
#  - 한 에피소드 동안 첫 goal 도달 시점까지 시뮬레이션
#  - 프레임을 수집하여 mp4로 저장

# 출력 디렉토리: ~/rl_ws/goal_reached_step_mp4
# 파일명 형식: <run_name>_goal_reached.mp4
# """
# import os
# import argparse
# import imageio
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines3.common.monitor import Monitor
# from mujoco_rl_env.env.franka_reach_env import FrankaReachEnv

# def make_eval_env(xml_path, init_qpos, goal_qpos, render_mode="rgb_array"):
#     """평가용 단일 환경 팩토리 함수."""
#     def _init():
#         env = FrankaReachEnv(
#             xml_path=xml_path,
#             init_qpos=init_qpos,
#             goal_qpos=goal_qpos,
#             render_mode=render_mode
#         )
#         return Monitor(env)
#     return _init

# def find_model_and_stats(run_dir):
#     """
#     run_dir 하위에서
#      - best_model/best_model.zip 또는 final_model.zip
#      - vec_normalize.pkl
#     의 경로를 찾아 반환.
#     """
#     best = os.path.join(run_dir, "best_model", "best_model.zip")
#     final = os.path.join(run_dir, "final_model.zip")
#     vec   = os.path.join(run_dir, "vec_normalize.pkl")
#     model_path = best if os.path.isfile(best) else (final if os.path.isfile(final) else None)
#     stats_path = vec if os.path.isfile(vec) else None
#     return model_path, stats_path

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--runs-dir", default=os.path.expanduser("~/rl_ws/runs"),
#                    help="학습된 PPO 모델들이 들어있는 runs/ 디렉토리")
#     p.add_argument("--out-dir",  default=os.path.expanduser("~/rl_ws/goal_reached_step_mp4"),
#                    help="goal 도달 동영상을 저장할 출력 디렉토리")
#     p.add_argument("--xml",      default="models/fr3_reach.xml",
#                    help="MuJoCo XML 모델 경로")
#     p.add_argument("--init", nargs="*", type=float, default=None,
#                    help="초기 관절 위치 (옵션)")
#     p.add_argument("--goal", nargs="*", type=float, default=None,
#                    help="목표 관절 위치 (옵션)")
#     p.add_argument("--max-steps", type=int, default=300,
#                    help="한 에피소드 최대 시뮬레이션 스텝 수")
#     p.add_argument("--fps",       type=int, default=30,
#                    help="저장할 동영상 FPS")
#     args = p.parse_args()

#     os.makedirs(args.out_dir, exist_ok=True)

#     # runs/ 디렉토리 내 'ppo_franka_' 로 시작하는 모든 폴더 순회
#     for run_name in sorted(os.listdir(args.runs_dir)):
#         if not run_name.startswith("ppo_franka_"):
#             continue
#         run_path = os.path.join(args.runs_dir, run_name)
#         if not os.path.isdir(run_path):
#             continue

#         # 모델과 정규화 통계 경로 찾기
#         model_file, stats_file = find_model_and_stats(run_path)
#         if model_file is None:
#             print(f"[SKIP] {run_name} 에서 모델을 찾을 수 없습니다.")
#             continue

#         # 평가 환경 생성
#         eval_env = DummyVecEnv([ make_eval_env(
#             xml_path=args.xml,
#             init_qpos=args.init,
#             goal_qpos=args.goal,
#             render_mode="rgb_array"
#         )])
#         # 정규화 로드 또는 끄기
#         if stats_file:
#             eval_env = VecNormalize.load(stats_file, eval_env)
#         else:
#             eval_env = VecNormalize(eval_env, training=False)

#         # 정책 로드
#         model = PPO.load(model_file, env=eval_env)

#         # 한 에피소드 동안 첫 goal 도달시까지 롤아웃
#         obs = eval_env.reset()
#         frames = []
#         success = False
#         for step in range(args.max_steps):
#             action, _ = model.predict(obs, deterministic=True)
#             obs, reward, done, info = eval_env.step(action)
#             frame = eval_env.venv.envs[0].render()
#             frames.append(frame)
#             # 'success' flag나 done이 True면 중단
#             if info.get("success", False) or done:
#                 success = info.get("success", False)
#                 print(f"[{run_name}] Goal reached at step {step}: success={success}")
#                 break

#         # goal에 실제로 도달한 경우에만 저장
#         if success and frames:
#             out_path = os.path.join(args.out_dir, f"{run_name}_goal_reached.mp4")
#             # 기존 파일이 있으면 삭제하고 덮어쓰기
#             if os.path.exists(out_path):
#                 os.remove(out_path)
#             imageio.mimsave(out_path, frames, fps=args.fps)
#             print(f"[저장됨] {out_path}")
#         else:
#             print(f"[SKIP] {run_name} 은 goal에 도달하지 못해 영상 저장 안 함")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
각 PPO 실행 디렉토리(runs/ppo_franka_*/)를 찾아,
 - best_model.zip 또는 final_model.zip 을 로드하고
 - vec_normalize.pkl(정규화 통계)이 있으면 함께 로드
 - 한 에피소드 동안 첫 goal 도달 시점까지 시뮬레이션
 - 프레임을 수집하여 mp4로 저장

출력 디렉토리: ~/rl_ws/goal_reached_step_mp4
파일명 형식: <run_name>_goal_reached.mp4
"""
import os
import argparse
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from mujoco_rl_env.env.franka_reach_env import FrankaReachEnv

def make_eval_env(xml_path, init_qpos, goal_qpos, render_mode="rgb_array"):
    """평가용 단일 환경 팩토리 함수."""
    def _init():
        env = FrankaReachEnv(
            xml_path=xml_path,
            init_qpos=init_qpos,
            goal_qpos=goal_qpos,
            render_mode=render_mode
        )
        return Monitor(env)
    return _init

def find_model_and_stats(run_dir):
    """
    run_dir 하위에서
     - best_model/best_model.zip 또는 final_model.zip
     - vec_normalize.pkl
    의 경로를 찾아 반환.
    """
    best = os.path.join(run_dir, "best_model", "best_model.zip")
    final = os.path.join(run_dir, "final_model.zip")
    vec   = os.path.join(run_dir, "vec_normalize.pkl")
    model_path = best if os.path.isfile(best) else (final if os.path.isfile(final) else None)
    stats_path = vec if os.path.isfile(vec) else None
    return model_path, stats_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default=os.path.expanduser("~/rl_ws/runs"),
                   help="학습된 PPO 모델들이 들어있는 runs/ 디렉토리")
    p.add_argument("--out-dir",  default=os.path.expanduser("~/rl_ws/goal_reached_step_mp4"),
                   help="goal 도달 동영상을 저장할 출력 디렉토리")
    p.add_argument("--xml",      default="models/fr3_reach.xml",
                   help="MuJoCo XML 모델 경로")
    p.add_argument("--init", nargs="*", type=float, default=None,
                   help="초기 관절 위치 (옵션)")
    p.add_argument("--goal", nargs="*", type=float, default=None,
                   help="목표 관절 위치 (옵션)")
    p.add_argument("--max-steps", type=int, default=300,
                   help="한 에피소드 최대 시뮬레이션 스텝 수")
    p.add_argument("--fps",       type=int, default=30,
                   help="저장할 동영상 FPS")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for run_name in sorted(os.listdir(args.runs_dir)):
        if not run_name.startswith("ppo_franka_"):
            continue
        run_path = os.path.join(args.runs_dir, run_name)
        if not os.path.isdir(run_path):
            continue

        model_file, stats_file = find_model_and_stats(run_path)
        if model_file is None:
            print(f"[SKIP] {run_name} 에서 모델을 찾을 수 없습니다.")
            continue

        eval_env = DummyVecEnv([ make_eval_env(
            xml_path=args.xml,
            init_qpos=args.init,
            goal_qpos=args.goal,
            render_mode="rgb_array"
        )])
        if stats_file:
            eval_env = VecNormalize.load(stats_file, eval_env)
        else:
            eval_env = VecNormalize(eval_env, training=False)

        model = PPO.load(model_file, env=eval_env)

        obs = eval_env.reset()
        frames = []
        success = False

        for step in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            # VecEnv에서는 step이 네 개를 반환: obs, reward, done, infos(list)
            obs, reward, done, infos = eval_env.step(action)
            info = infos[0]  # 첫 번째 환경의 info dict
            frame = eval_env.venv.envs[0].render()
            frames.append(frame)

            if info.get("success", False):
                print(f"[{run_name}] Goal reached at step {step}: success={success}")
                success = True
                break
            if done:
                print(f"[{run_name}] ⏱ max_steps({args.max_steps}) reached at step {step}, success=False")
                success = False
                break  

        if success:
            out_path = os.path.join(args.out_dir, f"{run_name}_goal_reached.mp4")
            if os.path.exists(out_path):
                os.remove(out_path)
            imageio.mimsave(out_path, frames, fps=args.fps)
            print(f"[저장됨] {out_path}")
        else:
            print(f"[SKIP] {run_name} 은 goal에 도달하지 못해 영상 저장 안 함")

if __name__ == "__main__":
    main()
