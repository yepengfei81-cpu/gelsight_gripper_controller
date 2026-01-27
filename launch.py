import os
import sys
import multiprocessing
import time
import subprocess

def run_gelsight():
    gelsight_script = "/home/ypf/mujoco_learn/gelsight_manager.py"  # 替换为实际路径
    
    # 创建全新的环境变量
    env = os.environ.copy()
    env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    env["KIVY_NO_ARGS"] = "1"
    
    # 使用新进程运行
    subprocess.run(
        [sys.executable, gelsight_script],
        env=env,
        check=True
    )

def run_simulation():
    simulation_script = "/home/ypf/mujoco_learn/test.py"  # 替换为实际路径
    
    # 创建专门的环境变量
    env = os.environ.copy()
    env["QT_QPA_PLATFORM_PLUGIN_PATH"] = '/home/ypf/conda_install/envs/mujoco/lib/python3.8/site-packages/PyQt5/Qt5/plugins/platforms'
    
    # 使用新进程运行
    subprocess.run(
        [sys.executable, simulation_script],
        env=env,
        check=True
    )

if __name__ == "__main__":
    # 创建两个进程
    gelsight_process = multiprocessing.Process(target=run_gelsight)
    simulation_process = multiprocessing.Process(target=run_simulation)

    try:
        # 启动进程
        print("Starting GelSight Mini Manager...")
        gelsight_process.start()
        time.sleep(3)  # 给GelSight启动留出更多时间
        
        print("Starting MuJoCo Simulation...")
        simulation_process.start()

        # 等待两个进程结束
        gelsight_process.join()
        simulation_process.join()
        
    except KeyboardInterrupt:
        print("\nTerminating processes...")
        gelsight_process.terminate()
        simulation_process.terminate()
        gelsight_process.join()
        simulation_process.join()
        print("All processes terminated")