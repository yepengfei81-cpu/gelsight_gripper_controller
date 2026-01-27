def run_mujoco_simulation():
    import os
    envpath = '/home/ypf/conda_install/envs/mujoco/lib/python3.8/site-packages/PyQt5/Qt5/plugins/platforms'
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath    
    import math
    import time
    import mujoco_py
    from mujoco_py import load_model_from_path, MjSim, MjViewer, MujocoException
    import glfw
    import numpy as np
    import mujoco as mj
    from mujoco import mju_axisAngle2Quat
    from transforms3d.euler import mat2euler, euler2mat
    from math import pi, atan2, sqrt, sin, cos, tan, radians, degrees, asin, copysign
    from transforms3d.quaternions import mat2quat, quat2mat
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation
    import threading
    from PyQt5.QtWidgets import QApplication
    from cube_controller import CubeController
    import queue

    # Load model
    model_path = '/home/ypf/five_link_grasper/xml/five_link_grasper_with_box.xml'
    if not os.path.exists(model_path):
        # Fallback to original file
        model_path = '/home/ypf/five_link_grasper/xml/five_link_grasper.xml'
        print(f"Warning: Box XML not found, using original model: {model_path}")

    model = load_model_from_path(model_path)
    sim = MjSim(model)
    viewer = MjViewer(sim)

    # Start cube controller UI
    ui_app = QApplication.instance()
    if not ui_app:
        ui_app = QApplication([])
        
    cube_controller = CubeController()
    cube_controller.show()
    cube_controller.raise_()

    # define switch for the servo thread
    ENABLE_SERVO_THREADS = True

    # add thread lock
    pose_data_lock = threading.Lock()

    def process_ui_events():
        if QApplication.instance():
            QApplication.processEvents()

    # define motor control queue
    left_servo_command_queue = queue.Queue()
    right_servo_command_queue = queue.Queue()

    # Get cube joints
    cube_joint_names = ['x_slide', 'y_slide', 'z_slide']

    # Check end-effector sites
    if 'left_ee' not in model.site_names:
        print("Warning: Left end-effector site 'left_ee' not defined")
    if 'right_ee' not in model.site_names:
        print("Warning: Right end-effector site 'right_ee' not defined")

    # Define kinematic chains
    kinematic_chains = {
        "left_arm": ["base_link", "left_link1", "left_link2", "left_link3", "left_link4", "left_link5"],
        "right_arm": ["base_link", "right_link1", "right_link2", "right_link3", "right_link4", "right_link5"]
    }

    # Get joint info
    joint_names = model.joint_names
    print("Controllable joints:", joint_names)

    # Create joint controllers
    joint_controllers = {}
    for joint_name in joint_names:
        try:
            joint_id = model.joint_name2id(joint_name)
            joint_controllers[joint_name] = {
                'joint_id': joint_id,
                'target_position': 0.0
            }
            print(f"Added controller for {joint_name} (ID: {joint_id})")
        except KeyError:
            print(f"Error: Joint {joint_name} not found")

    for joint_name in cube_joint_names:
        try:
            joint_id = model.joint_name2id(joint_name)
            joint_controllers[joint_name] = {
                'joint_id': joint_id,
                'target_position': 0.0
            }
            print(f"Added controller for cube joint {joint_name} (ID: {joint_id})")
        except KeyError:
            print(f"Error: Cube joint {joint_name} not found")

    if not joint_controllers:
        print("Error: No controllable joints found")
        exit()

    # Joint position setters
    def set_left_joint1_position(): return math.radians(-0)
    def set_left_joint2_position(): return math.radians(59.373207084913474)
    def set_left_joint3_position(): return math.radians(-77.85859630095572)
    def set_left_joint4_position(): return math.radians(-71.50461078395774)
    def set_left_joint5_position(): return math.radians(0)

    def set_right_joint1_position(): return math.radians(0)
    def set_right_joint2_position(): return math.radians(59.3732070849135)
    def set_right_joint3_position(): return math.radians(-77.85859630095572)
    def set_right_joint4_position(): return math.radians(-71.50461078395774)
    def set_right_joint5_position(): return math.radians(0)

    def set_cube_x_position(): return 0.0
    def set_cube_y_position(): return 0.0
    def set_cube_z_position(): return 0.0

    # Map joints to position setters
    position_setters = {
        "left_joint1": set_left_joint1_position,
        "left_joint2": set_left_joint2_position,
        "left_joint3": set_left_joint3_position,
        "left_joint4": set_left_joint4_position,
        "left_joint5": set_left_joint5_position,
        "right_joint1": set_right_joint1_position,
        "right_joint2": set_right_joint2_position,
        "right_joint3": set_right_joint3_position,
        "right_joint4": set_right_joint4_position,
        "right_joint5": set_right_joint5_position,
        "x_slide": set_cube_x_position,
        "y_slide": set_cube_y_position,
        "z_slide": set_cube_z_position
    }

    # Set initial positions
    for joint_name, controller in joint_controllers.items():
        if joint_name in position_setters:
            controller['target_position'] = position_setters[joint_name]()
        else:
            controller['target_position'] = 0.0

    # Initialize simulation
    for joint_name, controller in joint_controllers.items():
        sim.data.set_joint_qpos(joint_name, controller['target_position'])
    sim.forward()

    # Find base reference frame
    try:
        base_body_id = model.body_name2id("base_link")
        print(f"Found base 'base_link' (ID: {base_body_id})")
    except:
        try:
            base_body_id = model.body_name2id("base_link_body")
            print(f"Found base 'base_link_body' (ID: {base_body_id})")
        except:
            base_body_id = None
            print("Warning: Base body not found, using world origin")

    def get_base_pose():
        if base_body_id is not None:
            base_pos = sim.data.body_xpos[base_body_id].copy()
            base_rot = sim.data.body_xmat[base_body_id].reshape(3, 3).copy()
            return base_pos, base_rot
        return np.zeros(3), np.eye(3)

    def set_cube_position_in_world_frame(world_pos):
        init_cube_pos = np.array([0, 0, 0])
        displacement = world_pos - init_cube_pos
        
        if 'z_slide' in joint_controllers:
            joint_controllers['z_slide']['target_position'] = displacement[0]
        if 'y_slide' in joint_controllers:
            joint_controllers['y_slide']['target_position'] = -displacement[1]
        if 'x_slide' in joint_controllers:
            joint_controllers['x_slide']['target_position'] = displacement[2]

    def set_cube_orientation_in_world_frame(world_ori):
        init_cube_ori = np.array([0, 0, 0])
        displacement = world_ori - init_cube_ori
        
        if 'z_rot' in joint_controllers:
            joint_controllers['z_rot']['target_position'] = radians(displacement[0])
        if 'y_rot' in joint_controllers:
            joint_controllers['y_rot']['target_position'] = radians(-displacement[1])
        if 'x_rot' in joint_controllers:
            joint_controllers['x_rot']['target_position'] = radians(displacement[2])


    # DH transformation matrix
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

    # Coordinate transform (DH to Mujoco)
    dh_to_mujoco_transform = np.array([
        [0, 0, 1],  # DH y -> Mujoco x
        [0, -1, 0], # DH z -> Mujoco y (inverted)
        [1, 0, 0]   # DH x -> Mujoco z
    ])

    rot_y_180 = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])

    # define thread to control motor
    def left_servo_control_thread():
        while True:
            try:
                left_commands = []
                while not (left_servo_command_queue.empty()):                
                    try:
                        left_commands.append(left_servo_command_queue.get_nowait())
                    except queue.Empty:
                        break      
                        
                if left_commands:
                    for servo_id, position, move_time in left_commands:
                        BusServoCmd.setBusServoPos(servo_id, position, move_time)     

                    time.sleep(0.001)
            except queue.Empty:
                time.sleep(0.001)
                continue
            except KeyboardInterrupt:
                break

    def right_servo_control_thread():
        while True:
            try:
                right_commands = []
                while not (right_servo_command_queue.empty()):
                    try:
                        right_commands.append(right_servo_command_queue.get_nowait())
                    except queue.Empty:
                        break       
                        
                if right_commands:
                    for servo_id, position, move_time in right_commands:
                        BusServoCmd.setBusServoPos(servo_id, position, move_time)      

                    time.sleep(0.001)
            except queue.Empty:
                time.sleep(0.001)
                continue
            except KeyboardInterrupt:
                break

    # Inverse Kinematics class
    class ArmIK:
        def __init__(self, arm_side="left_arm"):
            self.arm_side = arm_side
            self.optimization_enabled = True
            self.position_priority = 1000.0
            self.orientation_priority = 1.0
            self.joint_velocity_priority = 0.1
            self.max_iterations = 50
            self.tolerance = 1e-4

            # Arm-specific parameters
            params = {
                "left_arm": {
                    "a1": -0.042,
                    "d1": 0.041,
                    "d2": 0.02825,
                    "q5_offset": 0,
                    "angle_sign": -1,
                    "q3_sign": -1,
                    "q2_transform": lambda q: radians(90 - q),
                    "q3_transform": lambda q: radians(q),
                    "q4_formula": lambda q2, q3, q234: (pi/2 - q234) - (q2 + q3)
                },
                "right_arm": {
                    "a1": -0.042,
                    "d1": -0.041,
                    "d2": -0.02825,
                    "q5_offset": 0,
                    "angle_sign": 1,
                    "q3_sign": 1,
                    "q2_transform": lambda q: radians(q - 90),
                    "q3_transform": lambda q: radians(-q),
                    "q4_formula": lambda q2, q3, q234: (q234 - pi/2) - (q2 + q3)
                }
            }
            self.params = params[arm_side]
            self.a2 = 0.0935   # l0
            self.a3 = 0.1053   # l1
            self.a4 = 0.0897   # l2        

            # Joint limits
            self.joint_limits = [
                (-180 * (pi / 180), 180 * (pi / 180)),    # joint1
                (-20 * (pi / 180), 180 * (pi / 180)),     # joint2
                (-150 * (pi / 180), 120 * (pi / 180)),     # joint3
                (-100 * (pi / 180), 100 * (pi / 180)),      # joint4
                (-180 * (pi / 180), 180 * (pi / 180))     # joint5
            ]

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

        def compute_fk_for_optimization(self, q):
            temp_sim = MjSim(model)
            joint_prefix = "left" if self.arm_side == "left_arm" else "right"
            for i in range(1, 6):
                joint_name = f"{joint_prefix}_joint{i}"
                if joint_name in sim.model.joint_names:
                    joint_id = sim.model.joint_name2id(joint_name)
                    temp_sim.data.qpos[joint_id] = q[i-1]
            
            temp_sim.forward()
            position, _, rotation_matrix = compute_forward_kinematics(self.arm_side, sim=temp_sim)
            return position, rotation_matrix
        
        def objective_function(self, q, target_pos, target_rot_mat, current_q):
            # calculate current pose
            current_pos, current_rot_mat = self.compute_fk_for_optimization(q)
            # get pose error
            pos_error = np.linalg.norm(current_pos - target_pos)
            rot_error = np.linalg.norm(current_rot_mat - target_rot_mat, 'fro')
            # get joint speed
            joint_velocity = np.linalg.norm(q - current_q)
            # get total error parameters
            total_error = (
                self.position_weight * pos_error**2 +
                self.orientation_weight * rot_error**2 +
                self.joint_velocity_weight * joint_velocity**2
            )
            return total_error        
        
        def solve_with_optimization(self, target_pos, target_rot_mat, current_q):
            bounds = [
                (radians(limit[0]), radians(limit[1])) 
                for limit in self.joint_limits
            ]
            # init position
            q0 = np.array(current_q)
            options = {
                'maxiter': self.max_iterations,
                'ftol': self.tolerance,
                'disp': False
            }       
            result = minimize(
                self.objective_function,
                q0,
                args=(target_pos, target_rot_mat, current_q),
                method='L-BFGS-B',
                bounds=bounds,
                options=options
            )
            if result.success:
                q_optimized = degrees(result.x)
                print(f"Optimize success, position error: {self.objective_function(result.x, target_pos, target_rot_mat, current_q):.6f}")
                return q_optimized, True
            else:
                print(f"Optimize failed: {result.message}")
                return None, False

        def solve(self, target_pos, target_rot_euler):
            # change target euler to mat
            target_rot_mat = euler2mat(
                radians(target_rot_euler[0]),
                radians(target_rot_euler[1]),
                radians(target_rot_euler[2]),
                axes='sxyz'
            ) @ rot_y_180
            current_q = np.zeros(5)
            joint_prefix = "left" if self.arm_side == "left_arm" else "right"
            for i in range(1, 6):
                joint_name = f"{joint_prefix}_joint{i}"
                if joint_name in sim.data.qpos:
                    current_q[i-1] = sim.data.get_joint_qpos(joint_name)

            # tru analytical solution
            q, success = self.try_analytical_solution(target_pos, target_rot_mat)
            # or try optimze solution
            # if (not success) and self.optimization_enabled:
            #     return self.solve_with_optimization(target_pos, target_rot_mat, current_q)
            return q, success
        
        def try_analytical_solution(self, target_pos, target_rot_mat):       
            try:
                p = self.params
                a1 = p["a1"]
                d1 = p["d1"]
                d2 = p["d2"]
                q5_offset = p["q5_offset"]
                angle_sign = p["angle_sign"]
                q3_sign = p["q3_sign"]
                q2_transform = p["q2_transform"]
                q3_transform = p["q3_transform"]
                q4_formula = p["q4_formula"]
                px, py, pz = target_pos
                px -= a1
                pz -= (d1 + d2)         
                rot1 = target_rot_mat[:, 0]
                rot2 = target_rot_mat[:, 1]
                rot3 = target_rot_mat[:, 2]              
                # Calculate q1
                q1_temp = -(atan2(py, -px))
                gap_theta = asin(0.062 / (sqrt(px**2 + py**2)))
                q1 = q1_temp + gap_theta
                # Adjust position
                # r = sqrt(px**2 + py**2) # new x
                r = 0.062 / tan(gap_theta) # new x
                h = pz                     # new y 
                # Calculate q234
                q234 = atan2((rot3[0] * cos(q1) + rot3[1] * sin(q1)), -(rot3[2]))                    
                # Calculate q5
                q5 = atan2(
                    (rot1[1] * cos(q1) - rot1[0] * sin(q1)), 
                    (rot2[1] * cos(q1) - rot2[0] * sin(q1))
                ) + q5_offset
                angle_offset = atan2(0.0085, 0.0893)
                m = self.a4 * sin(q234 + angle_sign * angle_offset) - r
                n = self.a4 * cos(q234 + angle_sign * angle_offset) - h            
                # Calculate q2
                c1 = (self.a3**2 - self.a2**2 - m**2 - n**2) / (2 * self.a2)
                if (m**2 + n**2 - c1**2 < 0):
                    print("No solution for q2")
                    return None, False
                
                q2_1 = (atan2(m, n) + atan2(sqrt(m**2 + n**2 - c1**2), c1)) * (180 / pi)
                q2_2 = (atan2(m, n) - atan2(sqrt(m**2 + n**2 - c1**2), c1)) * (180 / pi)
                if q2_1 > 90.0: q2_1 -= 360.0
                if q2_1 < -90.0: q2_1 += 360.0   
                if q2_2 > 90.0: q2_2 -= 360.0
                if q2_2 < -90.0: q2_2 += 360.0                    
                
                # Calculate q3 and q4
                c2 = (self.a2**2 - self.a3**2 - m**2 - n**2) / (2 * self.a3)
                if (m**2 + n**2 - c2**2 < 0):
                    print("No solution for q2+q3")
                    return None, False
                
                q23_1 = (atan2(m, n) + atan2(sqrt(m**2 + n**2 - c2**2), c2)) * (180 / pi)
                q23_2 = (atan2(m, n) - atan2(sqrt(m**2 + n**2 - c2**2), c2)) * (180 / pi)
                if q23_1 > 90.0: q23_1 -= 360.0
                if q23_1 < -90.0: q23_1 += 360.0   
                if q23_2 > 90.0: q23_2 -= 360.0
                if q23_2 < -90.0: q23_2 += 360.0
                
                sorted_q = sorted([q2_1, q2_2, q23_1, q23_2], reverse=True)
                q3_offset = degrees(atan2(0.01681, 0.1041)) * q3_sign
                
                q2_1_val = sorted_q[0]
                q3_1_val = sorted_q[0] - sorted_q[2] + q3_offset
                q4_1_val = q4_formula((q2_transform(q2_1_val)), (q3_transform(q3_1_val)), q234)
                
                q2_2_val = sorted_q[3]
                q3_2_val = sorted_q[3] - sorted_q[1] + q3_offset
                q4_2_val = q4_formula((q2_transform(q2_2_val)), (q3_transform(q3_2_val)), q234)             

                candidate1 = (q2_1_val, q3_1_val, q4_1_val)
                candidate2 = (q2_2_val, q3_2_val, q4_2_val)

                if self.is_valid_candidate(*candidate2):
                    if (self.arm_side == "right_arm"):
                        q2 = q2_1_val
                        q3 = q3_1_val
                    else:
                        q2 = q2_2_val
                        q3 = q3_2_val
                elif self.is_valid_candidate(*candidate1):
                    q2 = q2_2_val
                    q3 = q3_2_val
                else:
                    q2 = q2_1_val
                    q3 = q3_1_val

                q2_rad = q2_transform(q2)
                q3_rad = q3_transform(q3)

                if self.arm_side == "right_arm":
                    q5_temp = q5 + (2 * pi if q5 < 0 else 0) - pi
                    q234_temp = (q234)
                elif self.arm_side == "left_arm":
                    q5_temp = -q5
                    q234_temp = q234 % (2 * pi)
                q4_rad = q4_formula(q2_rad, q3_rad, q234_temp)
                
                q = [degrees(q1), degrees(q2_rad), degrees(q3_rad), degrees(q4_rad), degrees(q5_temp)]
                return q, True
            except Exception as e:
                print(f"IK error: {str(e)}")
                return None, False


    # Forward kinematics
    def compute_forward_kinematics(arm_side):
        T = np.eye(4)
        
        joint1_name = "left_joint1" if arm_side == "left_arm" else "right_joint1"
        q1 = sim.data.get_joint_qpos(joint1_name)
        joint2_name = "left_joint2" if arm_side == "left_arm" else "right_joint2"
        q2 = sim.data.get_joint_qpos(joint2_name)
        joint3_name = "left_joint3" if arm_side == "left_arm" else "right_joint3"
        q3 = sim.data.get_joint_qpos(joint3_name)    
        joint4_name = "left_joint4" if arm_side == "left_arm" else "right_joint4"
        q4 = sim.data.get_joint_qpos(joint4_name) 
        joint5_name = "left_joint5" if arm_side == "left_arm" else "right_joint5"
        q5 = sim.data.get_joint_qpos(joint5_name)     

        if arm_side == "left_arm":
            T_base_to_joint1 = dh_transform_matrix(0, -0.042, 0.041, 0)
            T_joint1_to_joint2 = dh_transform_matrix(-math.pi/2, 0, 0.02825, q1)
            T_joint2_to_joint3 = dh_transform_matrix(0, -0.0935, 0.062, q2 + 0.0)
            T_joint3_to_joint4 = dh_transform_matrix(0, -0.1053, 0, q3 + 0.16)
            T_joint4_to_joint5 = dh_transform_matrix(math.pi/2, -0.0085, 0, q4 + 1.41)  
            T_joint5_to_end = dh_transform_matrix(0, 0, -0.0893, q5)      
        else:
            T_base_to_joint1 = dh_transform_matrix(0, -0.042, -0.041, 0)
            T_joint1_to_joint2 = dh_transform_matrix(math.pi/2, 0, -0.02825, q1)
            T_joint2_to_joint3 = dh_transform_matrix(0, -0.0935, -0.062, q2 + 0.0)
            T_joint3_to_joint4 = dh_transform_matrix(0, -0.1053, 0, q3 + 0.16)
            T_joint4_to_joint5 = dh_transform_matrix(math.pi/2, -0.0085, 0, q4 + 1.41)
            T_joint5_to_end = dh_transform_matrix(0, 0, -0.0893, q5)  

        T = T_base_to_joint1 @ T_joint1_to_joint2 @ T_joint2_to_joint3 @ T_joint3_to_joint4 @ T_joint4_to_joint5 @ T_joint5_to_end
        
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        roll, pitch, yaw = mat2euler(rotation_matrix, 'sxyz')
        
        return position, (roll, pitch, yaw), rotation_matrix

    # Compute cube pose
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
        
        # Calculate x axis
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Orthogonal Y-axis
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        cube_rot_mat = np.column_stack((x_axis, y_axis, z_axis))
        
        roll, pitch, yaw = mat2euler(cube_rot_mat, 'sxyz')
        
        return cube_pos, (roll, pitch, yaw)

    # Create IK solvers
    left_ik_solver = ArmIK(arm_side="left_arm")
    right_ik_solver = ArmIK(arm_side="right_arm")

    # Initial end-effector poses
    init_left_arm_pos = target_pos_left = np.array([-0.2057, -0.0000, 0.0152])
    init_left_arm_rot = target_rot_left = np.array([180.00, -0.01, -180.00])

    init_right_arm_pos = target_pos_right = np.array([-0.2057, 0.0000, -0.0152])
    init_right_arm_rot = target_rot_right = np.array([0.00, 0.01, -180.00])

    # Main simulation loop
    print("Simulation running... Press ESC to exit")
    start_time = time.time()
    last_print_time = start_time
    last_fk_time = start_time

    print("Controls:")
    print("  R: Reset all joints")
    print("  ESC: Exit")
    print("  Cube controller:")

    # Key state tracking
    key_states = {
        glfw.KEY_R: False,
    }

    # End effector visualization
    class EndEffectorMarker:
        def __init__(self, sim, color, size=0.01, axis_length=0.05):
            self.sim = sim
            self.color = color
            self.size = size
            self.axis_length = axis_length
            
            # Marker IDs
            self.marker_id = None
            self.x_axis_id = None
            self.y_axis_id = None
            self.z_axis_id = None
            self.test_id = None
            
            self.position = np.zeros(3)
            self.rotation = np.eye(3)  # Store rotation matrix
            
        def update(self, position, rotation):
            self.position = position.copy()
            self.rotation = rotation.copy()
            
        def render(self, viewer):
            GEOM_SPHERE = 2
            GEOM_ARROW = 2
            
            # Calculate axis directions
            x_axis = self.rotation @ np.array([1, 0, 0]) * self.axis_length
            y_axis = self.rotation @ np.array([0, 1, 0]) * self.axis_length
            z_axis = self.rotation @ np.array([0, 0, 1]) * self.axis_length
            
            # Calculate endpoints
            x_end = self.position + x_axis
            y_end = self.position + y_axis
            z_end = self.position + z_axis
            
            # Render origin sphere
            if self.marker_id is None:
                self.marker_id = viewer.add_marker(
                    pos=self.position,
                    size=[self.size]*3,
                    rgba=self.color,
                    type=GEOM_SPHERE,
                    label=""
                )
            else:
                viewer.add_marker(
                    pos=self.position,
                    size=[self.size]*3,
                    rgba=self.color,
                    type=GEOM_SPHERE,
                    label="",
                    id=self.marker_id
                )
            
            # Render X axis (red)
            if self.x_axis_id is None:
                self.x_axis_id = viewer.add_marker(
                    pos=x_end,
                    size=[0.003, 0.003, 0.003],
                    rgba=[1, 0, 0, 1],
                    type=GEOM_ARROW,
                    label="X",
                )
            else:
                viewer.add_marker(
                    pos=x_end,
                    size=[0.003, 0.003, 0.003],
                    rgba=[1, 0, 0, 1],
                    type=GEOM_ARROW,
                    label="X",
                    id=self.x_axis_id
                )
            
            # Render Y axis (green)
            if self.y_axis_id is None:
                self.y_axis_id = viewer.add_marker(
                    pos=y_end,
                    size=[0.003, 0.003, 0.003],
                    rgba=[0, 1, 0, 1],
                    type=GEOM_ARROW,
                    label="Y",
                )
            else:
                viewer.add_marker(
                    pos=y_end,
                    size=[0.003, 0.003, 0.003],
                    rgba=[0, 1, 0, 1],
                    type=GEOM_ARROW,
                    label="Y",
                    id=self.y_axis_id
                )
            
            # Render Z axis (blue)
            if self.z_axis_id is None:
                self.z_axis_id = viewer.add_marker(
                    pos=z_end,
                    size=[0.003, 0.003, 0.003],
                    rgba=[0, 0, 1, 1],
                    type=GEOM_ARROW,
                    label="Z",
                )
            else:
                viewer.add_marker(
                    pos=z_end,
                    size=[0.003, 0.003, 0.003],
                    rgba=[0, 0, 1, 1],
                    type=GEOM_ARROW,
                    label="Z",
                    id=self.z_axis_id
                )

    # Create end-effector markers
    left_marker = EndEffectorMarker(sim, [1, 0, 0, 1], size=0.01, axis_length=0.05)  # Red
    right_marker = EndEffectorMarker(sim, [0, 1, 0, 1], size=0.01, axis_length=0.05)  # Green

    # Create target pose markers
    target_left_marker = EndEffectorMarker(sim, [1, 0.5, 0.5, 0.7], size=0.015, axis_length=0.05)
    target_right_marker = EndEffectorMarker(sim, [0.5, 1, 0.5, 0.7], size=0.015, axis_length=0.05)

    # Create estimated cube marker
    cube_estimate_marker = EndEffectorMarker(sim, [1, 0, 0, 1], size=0.01, axis_length=0.05)

    # Initialize positions
    left_ee_pos = np.zeros(3)
    right_ee_pos = np.zeros(3)
    estimate_cube_pos = np.zeros(3)
    left_rot_mat = np.eye(3)
    right_rot_mat = np.eye(3)
    estimate_cube_orient = np.zeros(3)

    # IK state flags
    ik_enabled = False
    ik_solved = False

    # Cube movement speed
    cube_move_speed = 0.001

    # start motor control thread before main thread
    if ENABLE_SERVO_THREADS:
        import BusServoCmd
        left_servo_thread = threading.Thread(target=left_servo_control_thread, daemon=True)
        right_servo_thread = threading.Thread(target=right_servo_control_thread, daemon=True)
        left_servo_thread.start()
        right_servo_thread.start()
    else:
        print("Disable the servo thread")

    try:
        while not glfw.window_should_close(viewer.window):
            current_time = time.time()
            elapsed = current_time - start_time

            # start gelsight thread
            # gelsight = initialize_gelsight()

            # Process UI events
            process_ui_events()
            
            # Update key states
            for key in key_states:
                current_state = glfw.get_key(viewer.window, key) == glfw.PRESS
                key_states[key] = current_state
            
            # Handle key actions        
            if key_states[glfw.KEY_R]:
                print("Resetting all joints")
                for controller in joint_controllers.values():
                    controller['target_position'] = 0.0
            
            # Set cube pose
            cube_dh_pos = cube_controller.get_position()
            cube_dh_ori = cube_controller.get_orientation()
            set_cube_position_in_world_frame(cube_dh_pos)
            set_cube_orientation_in_world_frame(cube_dh_ori)

            # Create rotation matrices
            cube_rot_mat = euler2mat(
                radians(cube_dh_ori[0]),
                radians(cube_dh_ori[1]),
                radians(cube_dh_ori[2]),
                axes='sxyz'
            )        
            init_left_rot_mat = euler2mat(
                radians(init_left_arm_rot[0]),
                radians(init_left_arm_rot[1]),
                radians(init_left_arm_rot[2]),
                axes='sxyz'
            )
            init_right_rot_mat = euler2mat(
                radians(init_right_arm_rot[0]),
                radians(init_right_arm_rot[1]),
                radians(init_right_arm_rot[2]),
                axes='sxyz'
            )        
            # Calculate target positions
            cube_half_width = 0.01
            left_point_in_cube = np.array([0, 0, cube_half_width])
            right_point_in_cube = np.array([0, 0, -cube_half_width])
            target_pos_left = cube_rot_mat.dot(left_point_in_cube) + cube_dh_pos
            target_pos_right = cube_rot_mat.dot(right_point_in_cube) + cube_dh_pos
            target_rot_mat_left = cube_rot_mat.dot(init_left_rot_mat)
            target_rot_mat_right = cube_rot_mat.dot(init_right_rot_mat) 
            left_euler = mat2euler(target_rot_mat_left, axes='sxyz')
            right_euler = mat2euler(target_rot_mat_right, axes='sxyz') 
            target_rot_left = [
                math.degrees(left_euler[0]),
                math.degrees(left_euler[1]),
                math.degrees(left_euler[2])
            ]       
            target_rot_right = [
                math.degrees(right_euler[0]),
                math.degrees(right_euler[1]),
                math.degrees(right_euler[2])
            ]                     

            # Set joint positions
            for joint_name, controller in joint_controllers.items():
                sim.data.set_joint_qpos(joint_name, controller['target_position'])
            
            # Update simulation
            sim.forward()
            
            # Compute forward kinematics periodically
            if current_time - last_fk_time >= 0.1:
                left_ee_pos, _, left_rot_mat = compute_forward_kinematics("left_arm")
                right_ee_pos, _, right_rot_mat = compute_forward_kinematics("right_arm")
                left_rot_mat = left_rot_mat @ rot_y_180
                right_rot_mat = right_rot_mat @ rot_y_180
                left_ee_orient = mat2euler(left_rot_mat, axes='sxyz')
                right_ee_orient = mat2euler(right_rot_mat, axes='sxyz')
                
                # Estimate cube pose
                estimate_cube_pos, estimate_cube_orient = compute_cube_pose(
                    left_ee_pos, left_rot_mat, 
                    right_ee_pos, right_rot_mat, 
                    cube_half_width
                )
                last_fk_time = current_time

                with pose_data_lock:
                    current_target_pos = cube_dh_pos.copy()
                    current_target_ori = cube_dh_ori.copy()
                    current_estimate_pos = estimate_cube_pos.copy()
                    current_estimate_ori = [
                        degrees(estimate_cube_orient[0]),
                        degrees(estimate_cube_orient[1]),
                        degrees(estimate_cube_orient[2])
                    ]                        
                # Update UI
                cube_controller.update_pose_comparison(
                    current_target_pos,
                    current_target_ori,
                    current_estimate_pos,
                    current_estimate_ori
                )                         
                
                print(f"Cube position: [{estimate_cube_pos[0]:.4f}, {estimate_cube_pos[1]:.4f}, {estimate_cube_pos[2]:.4f}]")
                print(f"Cube orientation: [{degrees(estimate_cube_orient[0]):.2f}, {degrees(estimate_cube_orient[1]):.2f}, {degrees(estimate_cube_orient[2]):.2f}]")      
                print(f"Time: {elapsed:.1f}s")
                print(f"Left EE position: [{left_ee_pos[0]:.4f}, {left_ee_pos[1]:.4f}, {left_ee_pos[2]:.4f}]")
                print(f"Left EE orientation: [{degrees(left_ee_orient[0]):.2f}, "
                        f"{degrees(left_ee_orient[1]):.2f}, {degrees(left_ee_orient[2]):.2f}]")
                print(f"Right EE position: [{right_ee_pos[0]:.4f}, {right_ee_pos[1]:.4f}, {right_ee_pos[2]:.4f}]")
                print(f"Right EE orientation: [{degrees(right_ee_orient[0]):.2f}, "
                        f"{degrees(right_ee_orient[1]):.2f}, {degrees(right_ee_orient[2]):.2f}]")
                last_print_time = current_time
                
            # Get base pose
            base_pos, base_rot = get_base_pose()
            
            # Transform estimated cube pose
            left_fk_pos = dh_to_mujoco_transform @ left_ee_pos
            left_pos_world = base_rot @ left_fk_pos + base_pos
            left_ee_orient = mat2euler(left_rot_mat, axes='sxyz')
            left_fk_rot = euler2mat((left_ee_orient[0]), (left_ee_orient[1]), (left_ee_orient[2]), axes='sxyz')
            left_rot_world = base_rot @ dh_to_mujoco_transform @ left_fk_rot

            right_fk_pos = dh_to_mujoco_transform @ right_ee_pos
            right_pos_world = base_rot @ right_fk_pos + base_pos
            right_ee_orient = mat2euler(right_rot_mat, axes='sxyz')
            right_fk_rot = euler2mat((right_ee_orient[0]), (right_ee_orient[1]), (right_ee_orient[2]), axes='sxyz')
            right_rot_world = base_rot @ dh_to_mujoco_transform @ right_fk_rot        

            estimate_cube_pos_mujoco = dh_to_mujoco_transform @ estimate_cube_pos
            estimate_cube_pos_world = base_rot @ estimate_cube_pos_mujoco + base_pos
            estimate_rot_mat_cube = euler2mat((estimate_cube_orient[0]), (estimate_cube_orient[1]), (estimate_cube_orient[2]), axes='sxyz')
            estimate_cube_rot_world = base_rot @ dh_to_mujoco_transform @ estimate_rot_mat_cube
            
            # Update markers
            left_marker.update(left_pos_world, left_rot_world)
            left_marker.render(viewer)
            right_marker.update(right_pos_world, right_rot_world)
            right_marker.render(viewer)
            # cube_estimate_marker.update(estimate_cube_pos_world, estimate_cube_rot_world)
            # cube_estimate_marker.render(viewer)
            
            # Solve IK
            q_left, success_left = left_ik_solver.solve(
                target_pos_left, 
                target_rot_left,
            )
            q_right, success_right = right_ik_solver.solve(
                target_pos_right, 
                target_rot_right,
            )        
            if success_left and success_right:
                # Set joint targets
                for i in range(5):
                    if i == 4:
                        servo_pos_left = int((q_left[i] / 0.24) + 500.0)
                        servo_pos_right = int((q_right[i] / 0.24) + 500.0)
                    elif i == 2:
                        servo_pos_left = int(((q_left[i] + 8.0 * copysign(1, q_left[i])) / 0.24) + 750.0)
                        servo_pos_right = int(((q_right[i] + 8.0 * copysign(1, q_left[i])) / 0.24) + 750.0)                    
                    else:
                        servo_pos_left = int((q_left[i] / 0.24) + 750.0)
                        servo_pos_right = int((q_right[i] / 0.24) + 750.0)

                    joint_controllers[f"left_joint{i+1}"]['target_position'] = radians(q_left[i])
                    left_servo_command_queue.put((i+1, servo_pos_left, 1))
                    joint_controllers[f"right_joint{i+1}"]['target_position'] = radians(q_right[i])
                    right_servo_command_queue.put((i+6, servo_pos_right, 1))     
                # print(q_left)
                # print(q_right) 
            else:
                print("IK solution not found")
            
            # Render
            viewer.render()
            glfw.poll_events()
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        glfw.terminate()
        print("Simulation ended")

if __name__ == "__main__":
    run_mujoco_simulation()
