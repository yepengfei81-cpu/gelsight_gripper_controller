import os
import math
import time
import threading
import queue
import numpy as np
import glfw
from mujoco_py import load_model_from_path, MjSim, MjViewer
from math import pi, atan2, sqrt, sin, cos, tan, radians, degrees, asin, copysign
from transforms3d.euler import mat2euler, euler2mat
from PyQt5.QtWidgets import QApplication
from cube_controller import CubeController
from servo_interpolator import ServoInterpolator, InterpolatorPresets

# 设置Qt环境
envpath = '/home/ypf/conda_install/envs/mujoco/lib/python3.8/site-packages/PyQt5/Qt5/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

# ==================== 配置 ====================
ENABLE_SERVO_THREADS = True
ENABLE_INTERPOLATION = True  # 启用插值控制
MODEL_PATH = '/home/ypf/five_link_gripper_v6/xml/five_link_gripper_v6.xml'

# ==================== DH变换矩阵 ====================
def dh_transform_matrix(alpha, a, d, theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

rot_y_180 = np.array([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, -1]
])

# ==================== 舵机控制器类（无阻塞） ====================
class ServoController:
    def __init__(self, name="servo"):
        self.name = name
        self.command_queue = queue.Queue(maxsize=100)
        self.running = True
        self.thread = None
        self.last_positions = {}
        self.min_change_threshold = 1
        
    def start(self):
        self.thread = threading.Thread(target=self._servo_loop, daemon=True)
        self.thread.start()
        print(f"[{self.name}] 舵机控制线程已启动")
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        print(f"[{self.name}] 舵机控制线程已停止")
        
    def send_command(self, servo_id, position, move_time=1):
        if servo_id in self.last_positions:
            if abs(position - self.last_positions[servo_id]) < self.min_change_threshold:
                return
        try:
            self.command_queue.put_nowait((servo_id, position, move_time))
            self.last_positions[servo_id] = position
        except queue.Full:
            try:
                self.command_queue.get_nowait()
                self.command_queue.put_nowait((servo_id, position, move_time))
            except queue.Empty:
                pass
                
    def send_batch_commands(self, commands):
        for cmd in commands:
            self.send_command(*cmd)
            
    def _servo_loop(self):
        import BusServoCmd
        while self.running:
            try:
                commands_to_send = []
                while True:
                    try:
                        cmd = self.command_queue.get_nowait()
                        commands_to_send.append(cmd)
                    except queue.Empty:
                        break
                
                if commands_to_send:
                    latest_commands = {}
                    for servo_id, position, move_time in commands_to_send:
                        latest_commands[servo_id] = (position, move_time)
                    
                    for servo_id, (position, move_time) in latest_commands.items():
                        try:
                            print(f"[{self.name}] 发送: 电机{servo_id} = {position}, 时间={move_time}ms")
                            BusServoCmd.setBusServoPos(servo_id, position, move_time)
                        except Exception as e:
                            print(f"[{self.name}] 发送命令失败 servo_id={servo_id}: {e}")
                    
                time.sleep(0.002)
            except Exception as e:
                print(f"[{self.name}] 舵机循环异常: {e}")
                time.sleep(0.01)


# ==================== 逆运动学求解器 ====================
class ArmIK:
    def __init__(self, arm_side="left_arm"):
        self.arm_side = arm_side
        params = {
            "left_arm": {
                "a1": -0.042183, "d1": 0.041, "d2": 0.020255,  # d2: 0.02825 → 0.020255
                "q5_offset": 0, "angle_sign": -1, "q3_sign": -1,
                "q2_transform": lambda q: radians(90 - q),
                "q3_transform": lambda q: radians(q),
                "q4_formula": lambda q2, q3, q234: (pi/2 - q234) - (q2 + q3)
            },
            "right_arm": {
                "a1": -0.042183, "d1": -0.041, "d2": -0.020255,  # d2: -0.02825 → -0.020245
                "q5_offset": 0, "angle_sign": 1, "q3_sign": 1,
                "q2_transform": lambda q: radians(q - 90),
                "q3_transform": lambda q: radians(-q),
                "q4_formula": lambda q2, q3, q234: (q234 - pi/2) - (q2 + q3)
            }
        }
        self.params = params[arm_side]
        self.a2 = 0.090
        self.a3 = 0.0938
        self.a4 = 0.0904

    def is_valid_candidate(self, q2_val, q3_val, q4_val):
        q2_transform = self.params["q2_transform"]
        q3_transform = self.params["q3_transform"]
        q2_rad = q2_transform(q2_val)
        q3_rad = q3_transform(q3_val)
        return (
            q4_val < radians(-100.0) or q4_val > radians(100.0) or
            q2_rad < radians(-20.0) or q2_rad > radians(180.0) or
            q3_rad < radians(-150.0) or q3_rad > radians(120.0)
        )

    def solve(self, target_pos, target_rot_euler):
        target_rot_mat = euler2mat(
            radians(target_rot_euler[0]),
            radians(target_rot_euler[1]),
            radians(target_rot_euler[2]),
            axes='sxyz'
        ) @ rot_y_180
        return self.try_analytical_solution(target_pos, target_rot_mat)

    def try_analytical_solution(self, target_pos, target_rot_mat):
        try:
            p = self.params
            a1, d1, d2 = p["a1"], p["d1"], p["d2"]
            q5_offset, angle_sign, q3_sign = p["q5_offset"], p["angle_sign"], p["q3_sign"]
            q2_transform, q3_transform, q4_formula = p["q2_transform"], p["q3_transform"], p["q4_formula"]
            
            px, py, pz = target_pos
            px -= a1
            pz -= (d1 + d2)
            rot1, rot2, rot3 = target_rot_mat[:, 0], target_rot_mat[:, 1], target_rot_mat[:, 2]
            
            q1 = -(atan2(py, -px))
            
            r = sqrt(px**2 + py**2)
            h = pz
            
            q234 = atan2((rot3[0] * cos(q1) + rot3[1] * sin(q1)), -(rot3[2]))
            q5 = atan2(
                (rot1[1] * cos(q1) - rot1[0] * sin(q1)),
                (rot2[1] * cos(q1) - rot2[0] * sin(q1))
            ) + q5_offset
            
            angle_offset = atan2(0.0085, 0.0893)
            m = self.a4 * sin(q234 + angle_sign * angle_offset) - r
            n = self.a4 * cos(q234 + angle_sign * angle_offset) - h
            
            c1 = (self.a3**2 - self.a2**2 - m**2 - n**2) / (2 * self.a2)
            if (m**2 + n**2 - c1**2 < 0):
                return None, False

            q2_1 = (atan2(m, n) + atan2(sqrt(m**2 + n**2 - c1**2), c1)) * (180 / pi)
            q2_2 = (atan2(m, n) - atan2(sqrt(m**2 + n**2 - c1**2), c1)) * (180 / pi)
            if q2_1 > 90.0: q2_1 -= 360.0
            if q2_1 < -90.0: q2_1 += 360.0
            if q2_2 > 90.0: q2_2 -= 360.0
            if q2_2 < -90.0: q2_2 += 360.0

            c2 = (self.a2**2 - self.a3**2 - m**2 - n**2) / (2 * self.a3)
            if (m**2 + n**2 - c2**2 < 0):
                return None, False

            q23_1 = (atan2(m, n) + atan2(sqrt(m**2 + n**2 - c2**2), c2)) * (180 / pi)
            q23_2 = (atan2(m, n) - atan2(sqrt(m**2 + n**2 - c2**2), c2)) * (180 / pi)
            if q23_1 > 90.0: q23_1 -= 360.0
            if q23_1 < -90.0: q23_1 += 360.0
            if q23_2 > 90.0: q23_2 -= 360.0
            if q23_2 < -90.0: q23_2 += 360.0

            sorted_q = sorted([q2_1, q2_2, q23_1, q23_2], reverse=True)
            q3_offset = degrees(atan2(0.014926, 0.093814)) * q3_sign

            q2_1_val, q3_1_val = sorted_q[0], sorted_q[0] - sorted_q[2] + q3_offset
            q4_1_val = q4_formula((q2_transform(q2_1_val)), (q3_transform(q3_1_val)), q234)
            q2_2_val, q3_2_val = sorted_q[3], sorted_q[3] - sorted_q[1] + q3_offset
            q4_2_val = q4_formula((q2_transform(q2_2_val)), (q3_transform(q3_2_val)), q234)

            candidate1, candidate2 = (q2_1_val, q3_1_val, q4_1_val), (q2_2_val, q3_2_val, q4_2_val)

            if self.is_valid_candidate(*candidate2):
                q2, q3 = (q2_1_val, q3_1_val) if self.arm_side == "right_arm" else (q2_2_val, q3_2_val)
            elif self.is_valid_candidate(*candidate1):
                q2, q3 = q2_2_val, q3_2_val
            else:
                q2, q3 = q2_1_val, q3_1_val

            q2_rad, q3_rad = q2_transform(q2), q3_transform(q3)

            if self.arm_side == "right_arm":
                q5_temp = q5 + (2 * pi if q5 < 0 else 0) - pi
                q234_temp = q234
            elif self.arm_side == "left_arm":
                q5_temp = -q5
                q234_temp = q234 % (2 * pi)
            q4_rad = q4_formula(q2_rad, q3_rad, q234_temp)

            return [degrees(q1), degrees(q2_rad), degrees(q3_rad), degrees(q4_rad), degrees(q5_temp)], True
        except Exception as e:
            print(f"IK error: {str(e)}")
            return None, False


# ==================== 正运动学 ====================
def compute_forward_kinematics(arm_side, sim):
    joint_prefix = "left" if arm_side == "left_arm" else "right"
    q1 = sim.data.get_joint_qpos(f"{joint_prefix}_joint1")
    q2 = sim.data.get_joint_qpos(f"{joint_prefix}_joint2")
    q3 = sim.data.get_joint_qpos(f"{joint_prefix}_joint3")
    q4 = sim.data.get_joint_qpos(f"{joint_prefix}_joint4")
    q5 = sim.data.get_joint_qpos(f"{joint_prefix}_joint5")

    if arm_side == "left_arm":
        T_base_to_joint1 = dh_transform_matrix(0, -0.042183, 0.041, 0)
        T_joint1_to_joint2 = dh_transform_matrix(-math.pi/2, 0, 0.020255, q1)
        T_joint2_to_joint3 = dh_transform_matrix(0, -0.090, 0.0, q2 + 0.0)   # 0.062 → 0
        T_joint3_to_joint4 = dh_transform_matrix(0, -0.0938, 0, q3 + 0.158)
        T_joint4_to_joint5 = dh_transform_matrix(math.pi/2, -0.0085, 0, q4 + 1.413)
        T_joint5_to_end = dh_transform_matrix(0, 0, -0.0904, q5)
    else:
        T_base_to_joint1 = dh_transform_matrix(0, -0.042183, -0.041, 0)
        T_joint1_to_joint2 = dh_transform_matrix(math.pi/2, 0, -0.020255, q1)
        T_joint2_to_joint3 = dh_transform_matrix(0, -0.090, 0.0, q2 + 0.0)   # -0.062 → 0
        T_joint3_to_joint4 = dh_transform_matrix(0, -0.0938, 0, q3 + 0.158)
        T_joint4_to_joint5 = dh_transform_matrix(math.pi/2, -0.0085, 0, q4 + 1.413)
        T_joint5_to_end = dh_transform_matrix(0, 0, -0.0904, q5)

    T = T_base_to_joint1 @ T_joint1_to_joint2 @ T_joint2_to_joint3 @ T_joint3_to_joint4 @ T_joint4_to_joint5 @ T_joint5_to_end
    position = T[:3, 3]
    rotation_matrix = T[:3, :3]
    roll, pitch, yaw = mat2euler(rotation_matrix, 'sxyz')
    return position, (roll, pitch, yaw), rotation_matrix


# ==================== 方块位姿估计 ====================
def compute_cube_pose(left_pos, left_rot_mat, right_pos, right_rot_mat, cube_half_width):
    cube_pos = (left_pos + right_pos) / 2.0
    z_axis = left_pos - right_pos
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    y_axis_left = left_rot_mat[:, 1]
    y_axis_right = right_rot_mat[:, 1]
    if np.dot(y_axis_left, y_axis_right) < 0:
        y_axis_right = -y_axis_right
    
    y_axis = (y_axis_left + y_axis_right) / 2.0
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    cube_rot_mat = np.column_stack((x_axis, y_axis, z_axis))
    roll, pitch, yaw = mat2euler(cube_rot_mat, 'sxyz')
    return cube_pos, (roll, pitch, yaw)


# ==================== 主程序 ====================
def run_simulation():
    # 加载模型
    model = load_model_from_path(MODEL_PATH)
    sim = MjSim(model)
    viewer = MjViewer(sim)

    # 启动Qt界面
    ui_app = QApplication.instance()
    if not ui_app:
        ui_app = QApplication([])
    cube_controller = CubeController()
    cube_controller.show()
    cube_controller.raise_()

    # 线程锁
    pose_data_lock = threading.Lock()
    sim_lock = threading.Lock()  # 仿真数据锁

    def process_ui_events():
        if QApplication.instance():
            QApplication.processEvents()

    # 关节名称映射
    joint_names = [
        "left_joint1", "left_joint2", "left_joint3", "left_joint4", "left_joint5",
        "right_joint1", "right_joint2", "right_joint3", "right_joint4", "right_joint5"
    ]

    # 关节控制器
    joint_controllers = {}
    for joint_name in model.joint_names:
        try:
            joint_id = model.joint_name2id(joint_name)
            joint_controllers[joint_name] = {'joint_id': joint_id, 'target_position': 0.0}
        except KeyError:
            pass

    # 初始关节角度
    initial_angles_deg = {
        "left_joint1": 0, "left_joint2": 59.373207084913474,
        "left_joint3": -77.85859630095572, "left_joint4": -71.50461078395774,
        "left_joint5": 0,
        "right_joint1": 0, "right_joint2": 59.3732070849135,
        "right_joint3": -77.85859630095572, "right_joint4": -71.50461078395774,
        "right_joint5": 0
    }

    # 设置初始位置
    for joint_name, controller in joint_controllers.items():
        if joint_name in initial_angles_deg:
            controller['target_position'] = math.radians(initial_angles_deg[joint_name])
        else:
            controller['target_position'] = 0.0

    # 初始化仿真
    for joint_name, controller in joint_controllers.items():
        sim.data.set_joint_qpos(joint_name, controller['target_position'])
    sim.forward()

    # 创建舵机控制器
    left_servo_controller = ServoController(name="左臂")
    right_servo_controller = ServoController(name="右臂")

    # 创建插值控制器
    interpolator = None
    if ENABLE_INTERPOLATION:
        # 使用平滑跟踪预设 - 适合UI拖动控制
        preset = InterpolatorPresets.smooth_tracking()
        interpolator = ServoInterpolator(num_joints=10, **preset)
        
        # 设置初始位置
        initial_positions = {}
        for i, name in enumerate(joint_names):
            if name in initial_angles_deg:
                initial_positions[i] = math.radians(initial_angles_deg[name])
        interpolator.set_initial_positions(initial_positions)
        
        # 仿真更新回调 - 在插值线程中更新仿真关节位置
        def simulation_update_callback(positions):
            with sim_lock:
                for i, name in enumerate(joint_names):
                    if i in positions and name in joint_controllers:
                        joint_controllers[name]['target_position'] = positions[i]
        
        # 舵机更新回调
        def servo_update_callback(left_commands, right_commands):
            if ENABLE_SERVO_THREADS:
                left_servo_controller.send_batch_commands(left_commands)
                right_servo_controller.send_batch_commands(right_commands)
        
        interpolator.set_simulation_callback(simulation_update_callback)
        interpolator.set_servo_callback(servo_update_callback)

    # IK求解器
    left_ik_solver = ArmIK(arm_side="left_arm")
    right_ik_solver = ArmIK(arm_side="right_arm")

    # 初始末端位姿
    init_left_arm_rot = np.array([180.00, -0.01, -180.00])
    init_right_arm_rot = np.array([0.00, 0.01, -180.00])

    # 启动线程
    if ENABLE_SERVO_THREADS:
        left_servo_controller.start()
        right_servo_controller.start()
    
    if ENABLE_INTERPOLATION and interpolator:
        interpolator.start()

    print("=== 仿真启动 ===")
    print("使用Qt界面控制方块位姿")
    print(f"插值控制: {'启用' if ENABLE_INTERPOLATION else '禁用'}")
    print(f"舵机控制: {'启用' if ENABLE_SERVO_THREADS else '禁用'}")
    print("ESC=退出")

    start_time = time.time()
    last_fk_time = start_time

    # 初始化变量
    left_ee_pos = np.zeros(3)
    right_ee_pos = np.zeros(3)
    left_rot_mat = np.eye(3)
    right_rot_mat = np.eye(3)
    estimate_cube_pos = np.zeros(3)
    estimate_cube_orient = np.zeros(3)

    # 主循环
    try:
        while not glfw.window_should_close(viewer.window):
            current_time = time.time()
            elapsed = current_time - start_time

            # 处理Qt事件
            process_ui_events()

            # 从Qt界面获取目标位姿
            cube_dh_pos = cube_controller.get_position()
            cube_dh_ori = cube_controller.get_orientation()
            cube_half_width = cube_controller.get_gripper_half_distance()  # 动态获取夹爪距离

            # 计算目标旋转矩阵
            cube_rot_mat = euler2mat(radians(cube_dh_ori[0]), radians(cube_dh_ori[1]), radians(cube_dh_ori[2]), axes='sxyz')
            init_left_rot_mat = euler2mat(radians(init_left_arm_rot[0]), radians(init_left_arm_rot[1]), radians(init_left_arm_rot[2]), axes='sxyz')
            init_right_rot_mat = euler2mat(radians(init_right_arm_rot[0]), radians(init_right_arm_rot[1]), radians(init_right_arm_rot[2]), axes='sxyz')

            # 计算左右臂目标位姿
            left_point_in_cube = np.array([0, 0, cube_half_width])
            right_point_in_cube = np.array([0, 0, -cube_half_width])
            target_pos_left = cube_rot_mat.dot(left_point_in_cube) + cube_dh_pos
            target_pos_right = cube_rot_mat.dot(right_point_in_cube) + cube_dh_pos
            target_rot_mat_left = cube_rot_mat.dot(init_left_rot_mat)
            target_rot_mat_right = cube_rot_mat.dot(init_right_rot_mat)
            left_euler = mat2euler(target_rot_mat_left, axes='sxyz')
            right_euler = mat2euler(target_rot_mat_right, axes='sxyz')
            target_rot_left = [degrees(left_euler[0]), degrees(left_euler[1]), degrees(left_euler[2])]
            target_rot_right = [degrees(right_euler[0]), degrees(right_euler[1]), degrees(right_euler[2])]

            # 求解IK
            q_left, ok_left = left_ik_solver.solve(target_pos_left, target_rot_left)
            q_right, ok_right = right_ik_solver.solve(target_pos_right, target_rot_right)

            if ok_left and ok_right:
                if ENABLE_INTERPOLATION and interpolator:
                    # 使用插值控制器 - 设置目标位置（弧度）
                    left_angles_rad = [radians(q) for q in q_left]
                    right_angles_rad = [radians(q) for q in q_right]
                    interpolator.set_arm_targets(left_angles_rad, right_angles_rad)
                else:
                    # 直接控制（原来的方式）
                    for i in range(5):
                        joint_controllers[f"left_joint{i+1}"]['target_position'] = radians(q_left[i])
                        joint_controllers[f"right_joint{i+1}"]['target_position'] = radians(q_right[i])

            # 更新仿真（从插值器获取当前位置或直接使用目标位置）
            with sim_lock:
                for joint_name, controller in joint_controllers.items():
                    sim.data.set_joint_qpos(joint_name, controller['target_position'])
            sim.forward()

            # 定期计算正运动学并打印
            if current_time - last_fk_time >= 0.1:
                left_ee_pos, _, left_rot_mat = compute_forward_kinematics("left_arm", sim)
                right_ee_pos, _, right_rot_mat = compute_forward_kinematics("right_arm", sim)
                left_rot_mat = left_rot_mat @ rot_y_180
                right_rot_mat = right_rot_mat @ rot_y_180
                left_ee_orient = mat2euler(left_rot_mat, axes='sxyz')
                right_ee_orient = mat2euler(right_rot_mat, axes='sxyz')

                estimate_cube_pos, estimate_cube_orient = compute_cube_pose(
                    left_ee_pos, left_rot_mat, right_ee_pos, right_rot_mat, cube_half_width
                )
                last_fk_time = current_time

                with pose_data_lock:
                    current_target_pos = cube_dh_pos.copy()
                    current_target_ori = cube_dh_ori.copy()
                    current_estimate_pos = estimate_cube_pos.copy()
                    current_estimate_ori = [degrees(estimate_cube_orient[0]), degrees(estimate_cube_orient[1]), degrees(estimate_cube_orient[2])]
                
                cube_controller.update_pose_comparison(current_target_pos, current_target_ori, current_estimate_pos, current_estimate_ori)

                print(f"\n--- Time: {elapsed:.1f}s ---")
                print(f"Cube position: [{estimate_cube_pos[0]:.4f}, {estimate_cube_pos[1]:.4f}, {estimate_cube_pos[2]:.4f}]")
                print(f"Cube orientation: [{degrees(estimate_cube_orient[0]):.2f}, {degrees(estimate_cube_orient[1]):.2f}, {degrees(estimate_cube_orient[2]):.2f}]")
                print(f"Left EE position: [{left_ee_pos[0]:.4f}, {left_ee_pos[1]:.4f}, {left_ee_pos[2]:.4f}]")
                print(f"Left EE orientation: [{degrees(left_ee_orient[0]):.2f}, {degrees(left_ee_orient[1]):.2f}, {degrees(left_ee_orient[2]):.2f}]")
                print(f"Right EE position: [{right_ee_pos[0]:.4f}, {right_ee_pos[1]:.4f}, {right_ee_pos[2]:.4f}]")
                print(f"Right EE orientation: [{degrees(right_ee_orient[0]):.2f}, {degrees(right_ee_orient[1]):.2f}, {degrees(right_ee_orient[2]):.2f}]")

            viewer.render()
            glfw.poll_events()
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        if ENABLE_INTERPOLATION and interpolator:
            interpolator.stop()
        if ENABLE_SERVO_THREADS:
            left_servo_controller.stop()
            right_servo_controller.stop()
        glfw.terminate()
        print("仿真结束")


if __name__ == "__main__":
    run_simulation()