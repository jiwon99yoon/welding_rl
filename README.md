# 학습실행
cd ~/rl_ws
export PYTHONPATH=$PWD/src
python3 -m mujoco_rl_env.train2 --xml /home/minjun/rl_ws/src/mujoco_rl_env/models/fr3_rl_reach.xml
