# /home/minjun/rl_ws/src/mujoco_rl_env/test_model.py
import argparse
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mujoco_rl_env.env.franka_reach_env import FrankaReachEnv

def test_model(model_path, vec_normalize_path, xml_path, init_qpos=None, goal_pose=None, n_episodes=5):
    """학습된 모델을 테스트하고 시각화"""
    
    # 환경 생성
    def make_env():
        return FrankaReachEnv(
            xml_path=xml_path,
            init_qpos=init_qpos,
            goal_pose=goal_pose,
            render_mode="human"  # 시각화를 위해 human 모드
        )
    
    # 벡터화된 환경 생성
    env = DummyVecEnv([make_env])
    
    # 정규화 통계 로드
    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False  # 평가 모드
    
    # 모델 로드
    model = PPO.load(model_path)
    
    print(f"Testing model: {model_path}")
    print(f"Running {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\nEpisode {episode + 1}:")
        
        while not done:
            # 학습된 정책으로 행동 선택
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            episode_length += 1
            
            # 렌더링 (human 모드이므로 자동으로 창에 표시됨)
            env.render()
            
            # 속도 조절 (너무 빠르지 않게)
            time.sleep(0.01)
        
        print(f"  - Reward: {episode_reward:.2f}")
        print(f"  - Length: {episode_length}")
        if 'distance' in info[0]:
            print(f"  - Final distance: {info[0]['distance']:.4f}")
    
    env.close()
    print("\nTest completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained PPO model")
    parser.add_argument("--model", default="ppo_franka_reach", help="Model path (without .zip)")
    parser.add_argument("--vec-norm", default="runs/vec_normalize.pkl", help="VecNormalize stats path")
    parser.add_argument("--xml", default="models/fr3_reach.xml", help="MuJoCo XML path")
    parser.add_argument("--init", nargs='*', type=float, help="Initial joint positions")
    parser.add_argument("--goal", nargs=7, type=float, help="Goal pose: x y z qx qy qz qw")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    
    args = parser.parse_args()
    
    test_model(
        model_path=args.model,
        vec_normalize_path=args.vec_norm,
        xml_path=args.xml,
        init_qpos=args.init,
        goal_pose=args.goal,
        n_episodes=args.episodes
    )
