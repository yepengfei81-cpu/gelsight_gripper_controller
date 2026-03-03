"""
舵机插值控制器 v2
用于平滑关节运动轨迹，同时控制仿真和真机
针对连续拖动控制场景优化
"""

import time
import threading
import numpy as np
from math import copysign
from collections import deque


class JointState:
    """单个关节的状态"""
    def __init__(self, joint_id, initial_position=0.0):
        self.joint_id = joint_id
        self.current_position = initial_position  # 当前插值位置（弧度）
        self.target_position = initial_position   # 目标位置（弧度）
        self.velocity = 0.0                        # 当前速度
        self.last_target = initial_position        # 上一个目标（用于检测目标变化）
        

class TargetFilter:
    """目标位置滤波器 - 平滑输入的目标位置"""
    def __init__(self, alpha=0.3):
        self.alpha = alpha  # 滤波系数 (0-1, 越小越平滑)
        self.filtered_targets = {}
        
    def filter(self, joint_id, raw_target):
        if joint_id not in self.filtered_targets:
            self.filtered_targets[joint_id] = raw_target
            return raw_target
        
        # 一阶低通滤波
        filtered = self.filtered_targets[joint_id] + self.alpha * (raw_target - self.filtered_targets[joint_id])
        self.filtered_targets[joint_id] = filtered
        return filtered
    
    def reset(self, joint_id, value):
        self.filtered_targets[joint_id] = value


class VelocityProfiler:
    """速度曲线规划器 - 生成平滑的速度曲线"""
    def __init__(self, max_velocity=2.0, max_acceleration=8.0, max_jerk=50.0):
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk  # 加加速度限制，使加速更平滑
        
    def compute_velocity(self, current_pos, target_pos, current_vel, dt):
        """计算下一时刻的目标速度"""
        error = target_pos - current_pos
        
        if abs(error) < 1e-6:
            return 0.0, current_pos
            
        direction = 1.0 if error > 0 else -1.0
        
        # 计算停止所需距离（考虑当前速度）
        stop_distance = (current_vel ** 2) / (2 * self.max_acceleration) if current_vel != 0 else 0
        
        # 目标速度
        if abs(error) <= abs(stop_distance) * 1.2:  # 留一点余量
            # 减速阶段
            desired_vel = direction * max(0, abs(current_vel) - self.max_acceleration * dt)
        else:
            # 加速或匀速阶段
            desired_vel = direction * min(abs(error) / 0.1, self.max_velocity)  # 0.1秒到达的速度
        
        # 限制加速度（使速度变化平滑）
        vel_change = desired_vel - current_vel
        max_change = self.max_acceleration * dt
        if abs(vel_change) > max_change:
            desired_vel = current_vel + copysign(max_change, vel_change)
        
        # 限制最大速度
        desired_vel = max(-self.max_velocity, min(self.max_velocity, desired_vel))
        
        # 计算新位置
        new_pos = current_pos + desired_vel * dt
        
        # 防止过冲
        if (desired_vel > 0 and new_pos > target_pos) or (desired_vel < 0 and new_pos < target_pos):
            new_pos = target_pos
            desired_vel = 0.0
            
        return desired_vel, new_pos


class ServoInterpolator:
    """
    舵机插值控制器 v2
    
    改进：
    1. 目标位置滤波 - 平滑UI输入的目标
    2. 速度曲线规划 - 限制加速度和加加速度
    3. 死区处理 - 忽略微小变化
    4. 自适应更新率 - 根据运动状态调整舵机命令频率
    """
    
    # 插值模式
    MODE_LINEAR = 'linear'
    MODE_TRAPEZOID = 'trapezoid'
    MODE_SMOOTH = 'smooth'
    MODE_VELOCITY_PROFILE = 'velocity_profile'  # 新增：速度曲线模式
    
    def __init__(self, 
                 num_joints=10,
                 interpolation_mode='velocity_profile',
                 update_rate=200,
                 max_velocity=2.0,
                 max_acceleration=8.0,
                 smoothing_factor=0.15,
                 target_filter_alpha=0.3,      # 目标滤波系数
                 position_deadzone=0.001,       # 位置死区 (rad)
                 servo_update_interval=0.02,    # 舵机命令最小间隔 (s)
                 servo_move_time=20):           # 舵机移动时间 (ms)
        """
        初始化插值控制器
        """
        self.num_joints = num_joints
        self.interpolation_mode = interpolation_mode
        self.update_rate = update_rate
        self.update_interval = 1.0 / update_rate
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.smoothing_factor = smoothing_factor
        self.position_deadzone = position_deadzone
        self.servo_update_interval = servo_update_interval
        self.servo_move_time = servo_move_time
        
        # 关节状态
        self.joints = {}
        for i in range(num_joints):
            self.joints[i] = JointState(i)
        
        # 目标滤波器
        self.target_filter = TargetFilter(alpha=target_filter_alpha)
        
        # 速度规划器
        self.velocity_profiler = VelocityProfiler(
            max_velocity=max_velocity,
            max_acceleration=max_acceleration
        )
        
        # 线程相关
        self.running = False
        self.interpolation_thread = None
        self.lock = threading.Lock()
        
        # 回调函数
        self.simulation_callback = None
        self.servo_callback = None
        
        # 舵机命令控制
        self.last_servo_update_time = 0
        self.last_servo_positions = {}  # 上次发送的舵机位置
        self.servo_position_threshold = 2  # 舵机位置变化阈值
        
        # 统计信息
        self.update_count = 0
        self.last_update_time = 0
        
    def set_initial_positions(self, positions):
        """设置初始位置（弧度）"""
        with self.lock:
            if isinstance(positions, dict):
                for joint_id, pos in positions.items():
                    if joint_id in self.joints:
                        self.joints[joint_id].current_position = pos
                        self.joints[joint_id].target_position = pos
                        self.joints[joint_id].last_target = pos
                        self.target_filter.reset(joint_id, pos)
            elif isinstance(positions, (list, np.ndarray)):
                for i, pos in enumerate(positions):
                    if i in self.joints:
                        self.joints[i].current_position = pos
                        self.joints[i].target_position = pos
                        self.joints[i].last_target = pos
                        self.target_filter.reset(i, pos)
                        
    def set_target_positions(self, positions, use_filter=True):
        """
        设置目标位置（弧度）- 非阻塞
        
        Args:
            positions: dict或list，关节目标位置
            use_filter: 是否对目标进行滤波
        """
        with self.lock:
            if isinstance(positions, dict):
                for joint_id, pos in positions.items():
                    if joint_id in self.joints:
                        if use_filter:
                            pos = self.target_filter.filter(joint_id, pos)
                        self.joints[joint_id].target_position = pos
            elif isinstance(positions, (list, np.ndarray)):
                for i, pos in enumerate(positions):
                    if i in self.joints:
                        if use_filter:
                            pos = self.target_filter.filter(i, pos)
                        self.joints[i].target_position = pos
                        
    def set_arm_targets(self, left_angles, right_angles, use_filter=True):
        """分别设置左右臂目标角度（弧度）"""
        with self.lock:
            for i in range(5):
                if i in self.joints:
                    pos = left_angles[i]
                    if use_filter:
                        pos = self.target_filter.filter(i, pos)
                    self.joints[i].target_position = pos
                if i + 5 in self.joints:
                    pos = right_angles[i]
                    if use_filter:
                        pos = self.target_filter.filter(i + 5, pos)
                    self.joints[i + 5].target_position = pos
                    
    def get_current_positions(self):
        """获取当前插值位置（弧度）"""
        with self.lock:
            return {i: self.joints[i].current_position for i in range(self.num_joints)}
    
    def get_current_positions_array(self):
        """获取当前插值位置数组（弧度）"""
        with self.lock:
            return np.array([self.joints[i].current_position for i in range(self.num_joints)])
    
    def get_arm_positions(self):
        """获取左右臂当前位置"""
        with self.lock:
            left = [self.joints[i].current_position for i in range(5)]
            right = [self.joints[i + 5].current_position for i in range(5)]
            return left, right
    
    def set_simulation_callback(self, callback):
        """设置仿真更新回调函数"""
        self.simulation_callback = callback
        
    def set_servo_callback(self, callback):
        """设置真机舵机回调函数"""
        self.servo_callback = callback
        
    def set_parameters(self, **kwargs):
        """动态设置参数"""
        with self.lock:
            if 'max_velocity' in kwargs:
                self.max_velocity = kwargs['max_velocity']
                self.velocity_profiler.max_velocity = kwargs['max_velocity']
            if 'max_acceleration' in kwargs:
                self.max_acceleration = kwargs['max_acceleration']
                self.velocity_profiler.max_acceleration = kwargs['max_acceleration']
            if 'smoothing_factor' in kwargs:
                self.smoothing_factor = kwargs['smoothing_factor']
            if 'target_filter_alpha' in kwargs:
                self.target_filter.alpha = kwargs['target_filter_alpha']
            if 'servo_move_time' in kwargs:
                self.servo_move_time = kwargs['servo_move_time']
        
    def start(self):
        """启动插值控制线程"""
        if self.running:
            return
            
        self.running = True
        self.last_update_time = time.time()
        self.last_servo_update_time = time.time()
        self.interpolation_thread = threading.Thread(
            target=self._interpolation_loop, 
            daemon=True
        )
        self.interpolation_thread.start()
        print(f"[ServoInterpolator] 插值控制器已启动 (模式: {self.interpolation_mode}, {self.update_rate}Hz)")
        print(f"  - 最大速度: {self.max_velocity} rad/s")
        print(f"  - 最大加速度: {self.max_acceleration} rad/s²")
        print(f"  - 目标滤波: alpha={self.target_filter.alpha}")
        print(f"  - 舵机命令间隔: {self.servo_update_interval*1000:.0f}ms")
        print(f"  - 舵机移动时间: {self.servo_move_time}ms")
        
    def stop(self):
        """停止插值控制线程"""
        self.running = False
        if self.interpolation_thread:
            self.interpolation_thread.join(timeout=1.0)
        print("[ServoInterpolator] 插值控制器已停止")
        
    def _interpolation_loop(self):
        """插值控制主循环"""
        while self.running:
            loop_start = time.time()
            
            try:
                dt = loop_start - self.last_update_time
                self.last_update_time = loop_start
                
                # 执行插值计算
                with self.lock:
                    self._update_interpolation(dt)
                    current_positions = {i: self.joints[i].current_position 
                                        for i in range(self.num_joints)}
                
                # 仿真回调 - 每次都更新
                if self.simulation_callback:
                    try:
                        self.simulation_callback(current_positions)
                    except Exception as e:
                        print(f"[ServoInterpolator] 仿真回调异常: {e}")
                
                # 舵机回调 - 控制更新频率
                if self.servo_callback:
                    time_since_last_servo = loop_start - self.last_servo_update_time
                    if time_since_last_servo >= self.servo_update_interval:
                        try:
                            left_commands, right_commands = self._prepare_servo_commands(current_positions)
                            if left_commands or right_commands:
                                self.servo_callback(left_commands, right_commands)
                                self.last_servo_update_time = loop_start
                        except Exception as e:
                            print(f"[ServoInterpolator] 舵机回调异常: {e}")
                
                self.update_count += 1
                
            except Exception as e:
                print(f"[ServoInterpolator] 插值循环异常: {e}")
            
            # 精确控制循环频率
            elapsed = time.time() - loop_start
            sleep_time = self.update_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    def _update_interpolation(self, dt):
        """根据插值模式更新所有关节位置"""
        for joint_id, joint in self.joints.items():
            error = joint.target_position - joint.current_position
            
            # 死区处理
            if abs(error) < self.position_deadzone:
                joint.velocity = 0.0
                continue
                
            if self.interpolation_mode == self.MODE_LINEAR:
                joint.current_position = self._linear_interpolate(joint, error, dt)
            elif self.interpolation_mode == self.MODE_TRAPEZOID:
                joint.current_position = self._trapezoid_interpolate(joint, error, dt)
            elif self.interpolation_mode == self.MODE_SMOOTH:
                joint.current_position = self._smooth_interpolate(joint, error, dt)
            elif self.interpolation_mode == self.MODE_VELOCITY_PROFILE:
                joint.velocity, joint.current_position = self.velocity_profiler.compute_velocity(
                    joint.current_position, joint.target_position, joint.velocity, dt
                )
            else:
                joint.current_position = self._smooth_interpolate(joint, error, dt)
                
    def _linear_interpolate(self, joint, error, dt):
        """线性插值"""
        direction = 1.0 if error > 0 else -1.0
        velocity = direction * min(abs(error) / dt, self.max_velocity)
        delta = velocity * dt
        if abs(delta) > abs(error):
            return joint.target_position
        return joint.current_position + delta
    
    def _trapezoid_interpolate(self, joint, error, dt):
        """梯形速度曲线插值"""
        direction = 1.0 if error > 0 else -1.0
        stop_distance = (joint.velocity ** 2) / (2 * self.max_acceleration) if self.max_acceleration > 0 else 0
        
        if abs(error) <= stop_distance:
            new_velocity = joint.velocity - direction * self.max_acceleration * dt
            if direction * new_velocity < 0:
                new_velocity = 0
        elif abs(joint.velocity) < self.max_velocity:
            new_velocity = joint.velocity + direction * self.max_acceleration * dt
            new_velocity = direction * min(abs(new_velocity), self.max_velocity)
        else:
            new_velocity = direction * self.max_velocity
            
        joint.velocity = new_velocity
        delta = joint.velocity * dt
        
        if abs(delta) > abs(error):
            joint.velocity = 0
            return joint.target_position
        return joint.current_position + delta
    
    def _smooth_interpolate(self, joint, error, dt):
        """平滑插值（指数衰减）"""
        alpha = min(self.smoothing_factor * dt * self.update_rate, 1.0)
        new_position = joint.current_position + alpha * error
        joint.velocity = (new_position - joint.current_position) / dt if dt > 0 else 0
        
        if abs(joint.velocity) > self.max_velocity:
            direction = 1.0 if error > 0 else -1.0
            new_position = joint.current_position + direction * self.max_velocity * dt
            joint.velocity = direction * self.max_velocity
            
        return new_position
    
    def _prepare_servo_commands(self, positions):
        """准备舵机命令 - 增加变化检测"""
        left_commands = []
        right_commands = []
        
        for i in range(5):
            # 左臂 (关节 0-4 -> 舵机 1-5)
            if i in positions:
                angle_deg = np.degrees(positions[i])
                servo_pos = self._angle_to_servo_position(i, angle_deg, is_left=True)
                servo_id = i + 1
                
                # 检查位置是否有显著变化
                if self._should_send_servo_command(servo_id, servo_pos):
                    left_commands.append((servo_id, servo_pos, self.servo_move_time))
                    self.last_servo_positions[servo_id] = servo_pos

            # 右臂 (关节 5-9 -> 舵机 6-10)
            if i + 5 in positions:
                angle_deg = np.degrees(positions[i + 5])
                servo_pos = self._angle_to_servo_position(i, angle_deg, is_left=False)              
                servo_id = i + 6

                if self._should_send_servo_command(servo_id, servo_pos):
                    right_commands.append((servo_id, servo_pos, self.servo_move_time))
                    self.last_servo_positions[servo_id] = servo_pos
                    
        return left_commands, right_commands
    
    def _should_send_servo_command(self, servo_id, new_pos):
        """判断是否应该发送舵机命令"""
        if servo_id not in self.last_servo_positions:
            return True
        return abs(new_pos - self.last_servo_positions[servo_id]) >= self.servo_position_threshold
    
    def _angle_to_servo_position(self, joint_index, angle_deg, is_left=True):
        """
        将关节角度转换为舵机位置值
        
        左臂和右臂使用相同的计算公式得到基础舵机值，
        但右臂需要相对于中心点镜像（因为左右臂是镜像安装的）
        
        舵机范围: 0-1500
        中心点: 750 (对于joint1,2,4) 或 500 (对于joint5) 或 750 (对于joint3)
        """
        # 先按左臂的方式计算舵机位置
        if joint_index == 4:  # joint5
            center = 500.0
            servo_pos = int((angle_deg / 0.24) + center)
        elif joint_index == 2:  # joint3
            center = 750.0
            servo_pos = int(((angle_deg + 8.0 * copysign(1, angle_deg)) / 0.24) + center)
        else:  # joint1, joint2, joint4
            center = 750.0
            servo_pos = int((angle_deg / 0.24) + center)
        
        # 右臂需要相对于中心点镜像
        if not is_left:
            if joint_index == 4:  # joint5 中心点是500
                servo_pos = int(2 * 500 - servo_pos)
            else:  # 其他关节中心点是750
                servo_pos = int(2 * 750 - servo_pos)
        
        # 限制在有效范围内
        servo_pos = max(0, min(1500, servo_pos))
        
        return servo_pos


# ==================== 预设配置 ====================
class InterpolatorPresets:
    """插值器预设配置"""
    
    @staticmethod
    def smooth_tracking():
        """平滑跟踪模式 - 适合UI拖动控制"""
        return {
            'interpolation_mode': 'velocity_profile',
            'update_rate': 200,
            'max_velocity': 1.5,           # 较低速度
            'max_acceleration': 5.0,       # 较低加速度
            'target_filter_alpha': 0.2,    # 较强滤波
            'servo_update_interval': 0.025, # 40Hz舵机命令
            'servo_move_time': 30,          # 较长移动时间
        }
    
    @staticmethod
    def responsive():
        """快速响应模式 - 适合点对点运动"""
        return {
            'interpolation_mode': 'velocity_profile',
            'update_rate': 200,
            'max_velocity': 3.0,
            'max_acceleration': 10.0,
            'target_filter_alpha': 0.5,
            'servo_update_interval': 0.02,
            'servo_move_time': 20,
        }
    
    @staticmethod
    def ultra_smooth():
        """超平滑模式 - 最小抖动"""
        return {
            'interpolation_mode': 'velocity_profile',
            'update_rate': 200,
            'max_velocity': 1.0,
            'max_acceleration': 3.0,
            'target_filter_alpha': 0.1,    # 非常强的滤波
            'servo_update_interval': 0.03,  # 33Hz舵机命令
            'servo_move_time': 40,
        }


# ==================== 使用示例 ====================
if __name__ == "__main__":
    import math
    
    # 使用预设创建插值器
    preset = InterpolatorPresets.smooth_tracking()
    interpolator = ServoInterpolator(num_joints=10, **preset)
    
    # 设置初始位置
    initial_positions = {
        0: 0, 1: math.radians(59.37), 2: math.radians(-77.86), 
        3: math.radians(-71.50), 4: 0,
        5: 0, 6: math.radians(59.37), 7: math.radians(-77.86), 
        8: math.radians(-71.50), 9: 0
    }
    interpolator.set_initial_positions(initial_positions)
    
    # 定义回调
    def sim_callback(positions):
        pass  # 仿真更新
        
    def servo_callback(left_cmds, right_cmds):
        if left_cmds or right_cmds:
            print(f"Servo commands: L={len(left_cmds)}, R={len(right_cmds)}")
        
    interpolator.set_simulation_callback(sim_callback)
    interpolator.set_servo_callback(servo_callback)
    
    # 启动
    interpolator.start()
    
    # 模拟UI拖动：连续小步变化
    print("\n模拟UI拖动控制...")
    for step in range(50):
        new_target = math.radians(step * 0.5)  # 每步0.5度
        interpolator.set_target_positions({0: new_target})
        time.sleep(0.02)  # 50Hz更新目标
    
    time.sleep(2)
    interpolator.stop()