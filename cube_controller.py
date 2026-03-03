import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QSlider, QLabel, QPushButton, QHBoxLayout, QGroupBox,
                             QTabWidget, QSizePolicy, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time
import queue

class PoseUpdateThread(QThread):
    """Thread for updating pose data"""
    update_completed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.queue = queue.Queue(maxsize=10)  # Limit queue size to avoid excessive memory usage
        self.running = True
    
    def run(self):
        """Thread main loop"""
        while self.running:
            try:
                # Get data from queue, wait max 0.1 seconds
                target_pos, target_ori, estimated_pos, estimated_ori = self.queue.get(timeout=0.1)
                
                # Update pose comparison window
                self.parent().pose_comparison_window.update_plots(
                    target_pos, 
                    target_ori,
                    estimated_pos,
                    estimated_ori
                )
                
                # Emit update completed signal
                self.update_completed.emit()
            except queue.Empty:
                # Queue empty, continue waiting
                continue
            except Exception as e:
                print(f"Pose update thread error: {e}")
    
    def add_pose_data(self, target_pos, target_ori, estimated_pos, estimated_ori):
        """Add new pose data to queue"""
        try:
            # If queue is full, try to remove an old item
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            
            # Add new data
            self.queue.put((target_pos, target_ori, estimated_pos, estimated_ori), timeout=0.01)
        except queue.Full:
            # Queue full, discard data
            pass
        except Exception as e:
            print(f"Error adding pose data: {e}")
    
    def stop(self):
        """Stop thread"""
        self.running = False
        self.wait()

class PoseComparisonWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cube Pose Comparison")
        self.setGeometry(100, 100, 800, 600)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create position comparison tab
        self.position_tab = QWidget()
        self.position_layout = QVBoxLayout(self.position_tab)
        self.tab_widget.addTab(self.position_tab, "Position Comparison")
        
        # Create orientation comparison tab
        self.orientation_tab = QWidget()
        self.orientation_layout = QVBoxLayout(self.orientation_tab)
        self.tab_widget.addTab(self.orientation_tab, "Orientation Comparison")
        
        # Initialize charts
        self.init_position_charts()
        self.init_orientation_charts()
        
        # Data storage
        self.timestamps = []
        self.target_positions = []
        self.estimated_positions = []
        self.target_orientations = []
        self.estimated_orientations = []
        
        # Max data points
        self.max_data_points = 200
        self.start_time = time.time()
    
    def init_position_charts(self):
        # X position chart
        self.x_fig = Figure()
        self.x_canvas = FigureCanvas(self.x_fig)
        self.x_ax = self.x_fig.add_subplot(111)
        self.x_ax.set_title("X Position Comparison")
        self.x_ax.set_xlabel("Time (s)")
        self.x_ax.set_ylabel("Position (m)")
        self.x_ax.grid(True)
        self.position_layout.addWidget(self.x_canvas)
        
        # Create lines for target and estimated positions
        self.x_target_line, = self.x_ax.plot([], [], 'r-', label='Target Position')
        self.x_est_line, = self.x_ax.plot([], [], 'b-', label='Estimated Position')
        self.x_ax.legend()
        
        # Y position chart
        self.y_fig = Figure()
        self.y_canvas = FigureCanvas(self.y_fig)
        self.y_ax = self.y_fig.add_subplot(111)
        self.y_ax.set_title("Y Position Comparison")
        self.y_ax.set_xlabel("Time (s)")
        self.y_ax.set_ylabel("Position (m)")
        self.y_ax.grid(True)
        self.position_layout.addWidget(self.y_canvas)
        
        self.y_target_line, = self.y_ax.plot([], [], 'r-')
        self.y_est_line, = self.y_ax.plot([], [], 'b-')
        
        # Z position chart
        self.z_fig = Figure()
        self.z_canvas = FigureCanvas(self.z_fig)
        self.z_ax = self.z_fig.add_subplot(111)
        self.z_ax.set_title("Z Position Comparison")
        self.z_ax.set_xlabel("Time (s)")
        self.z_ax.set_ylabel("Position (m)")
        self.z_ax.grid(True)
        self.position_layout.addWidget(self.z_canvas)
        
        self.z_target_line, = self.z_ax.plot([], [], 'r-')
        self.z_est_line, = self.z_ax.plot([], [], 'b-')
    
    def init_orientation_charts(self):
        # X rotation chart
        self.rx_fig = Figure()
        self.rx_canvas = FigureCanvas(self.rx_fig)
        self.rx_ax = self.rx_fig.add_subplot(111)
        self.rx_ax.set_title("X Rotation Comparison")
        self.rx_ax.set_xlabel("Time (s)")
        self.rx_ax.set_ylabel("Angle (°)")
        self.rx_ax.grid(True)
        self.orientation_layout.addWidget(self.rx_canvas)
        
        self.rx_target_line, = self.rx_ax.plot([], [], 'r-', label='Target Orientation')
        self.rx_est_line, = self.rx_ax.plot([], [], 'b-', label='Estimated Orientation')
        self.rx_ax.legend()
        
        # Y rotation chart
        self.ry_fig = Figure()
        self.ry_canvas = FigureCanvas(self.ry_fig)
        self.ry_ax = self.ry_fig.add_subplot(111)
        self.ry_ax.set_title("Y Rotation Comparison")
        self.ry_ax.set_xlabel("Time (s)")
        self.ry_ax.set_ylabel("Angle (°)")
        self.ry_ax.grid(True)
        self.orientation_layout.addWidget(self.ry_canvas)
        
        self.ry_target_line, = self.ry_ax.plot([], [], 'r-')
        self.ry_est_line, = self.ry_ax.plot([], [], 'b-')
        
        # Z rotation chart
        self.rz_fig = Figure()
        self.rz_canvas = FigureCanvas(self.rz_fig)
        self.rz_ax = self.rz_fig.add_subplot(111)
        self.rz_ax.set_title("Z Rotation Comparison")
        self.rz_ax.set_xlabel("Time (s)")
        self.rz_ax.set_ylabel("Angle (°)")
        self.rz_ax.grid(True)
        self.orientation_layout.addWidget(self.rz_canvas)
        
        self.rz_target_line, = self.rz_ax.plot([], [], 'r-')
        self.rz_est_line, = self.rz_ax.plot([], [], 'b-')
    
    def update_plots(self, target_pos, target_ori, estimated_pos, estimated_ori):
        current_time = time.time() - self.start_time

        # Process data
        target_pos = np.round(target_pos, 4)
        estimated_pos = np.round(estimated_pos, 4)
        target_ori = np.round(target_ori, 4)
        estimated_ori = np.round(estimated_ori, 4)
                  
        # Add new data points
        self.timestamps.append(current_time)
        self.target_positions.append(target_pos)
        self.estimated_positions.append(estimated_pos)
        self.target_orientations.append(target_ori)
        self.estimated_orientations.append(estimated_ori)
        
        # Limit number of data points
        if len(self.timestamps) > self.max_data_points:
            self.timestamps.pop(0)
            self.target_positions.pop(0)
            self.estimated_positions.pop(0)
            self.target_orientations.pop(0)
            self.estimated_orientations.pop(0)
        
        # Update charts only if window is visible
        if self.isVisible():
            # Update position charts
            self.update_position_plots()
            
            # Update orientation charts
            self.update_orientation_plots()
    
    def update_position_plots(self):
        # Extract X, Y, Z data
        target_x = [pos[0] for pos in self.target_positions]
        target_y = [pos[1] for pos in self.target_positions]
        target_z = [pos[2] for pos in self.target_positions]
        
        estimated_x = [pos[0] for pos in self.estimated_positions]
        estimated_y = [pos[1] for pos in self.estimated_positions]
        estimated_z = [pos[2] for pos in self.estimated_positions]
        
        # Update X position chart
        self.x_target_line.set_data(self.timestamps, target_x)
        self.x_est_line.set_data(self.timestamps, estimated_x)
        self.x_ax.relim()
        self.x_ax.autoscale_view()
        self.x_canvas.draw_idle()  # Non-blocking draw
        
        # Update Y position chart
        self.y_target_line.set_data(self.timestamps, target_y)
        self.y_est_line.set_data(self.timestamps, estimated_y)
        self.y_ax.relim()
        self.y_ax.autoscale_view()
        self.y_canvas.draw_idle()
        
        # Update Z position chart
        self.z_target_line.set_data(self.timestamps, target_z)
        self.z_est_line.set_data(self.timestamps, estimated_z)
        self.z_ax.relim()
        self.z_ax.autoscale_view()
        self.z_canvas.draw_idle()
    
    def update_orientation_plots(self):
        # Extract RX, RY, RZ data
        target_rx = [ori[0] for ori in self.target_orientations]
        target_ry = [ori[1] for ori in self.target_orientations]
        target_rz = [ori[2] for ori in self.target_orientations]
        
        estimated_rx = [ori[0] for ori in self.estimated_orientations]
        estimated_ry = [ori[1] for ori in self.estimated_orientations]
        estimated_rz = [ori[2] for ori in self.estimated_orientations]
        
        # Update X rotation chart
        self.rx_target_line.set_data(self.timestamps, target_rx)
        self.rx_est_line.set_data(self.timestamps, estimated_rx)
        self.rx_ax.relim()
        self.rx_ax.autoscale_view()
        self.rx_canvas.draw_idle()
        
        # Update Y rotation chart
        self.ry_target_line.set_data(self.timestamps, target_ry)
        self.ry_est_line.set_data(self.timestamps, estimated_ry)
        self.ry_ax.relim()
        self.ry_ax.autoscale_view()
        self.ry_canvas.draw_idle()
        
        # Update Z rotation chart
        self.rz_target_line.set_data(self.timestamps, target_rz)
        self.rz_est_line.set_data(self.timestamps, estimated_rz)
        self.rz_ax.relim()
        self.rz_ax.autoscale_view()
        self.rz_canvas.draw_idle()

class CubeController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cube Controller (Position & Orientation)")
        self.setGeometry(100, 100, 800, 700)  # 增加窗口高度
        
        # Initialize position (initial position in DH coordinates)
        self.position = np.array([-0.2057, 0.0, 0.0])
        # Initialize orientation (rotation angles around X/Y/Z axes, in degrees)
        self.orientation = np.array([0.0, 0.0, 0.0])
        # Initialize gripper distance (cube half width in meters)
        self.gripper_half_distance = 0.01  # 默认1cm半宽度，即2cm总距离
        
        # Define offsets
        self.offset_x = -0.2057
        self.offset_y = 0.0
        self.offset_z = 0.0
        self.offset_rx = 0.0
        self.offset_ry = 0.0
        self.offset_rz = 0.0
        
        # Create main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # ===== Gripper Distance Control Area =====
        gripper_group = QGroupBox("Gripper Distance Control")
        gripper_layout = QVBoxLayout()
        gripper_group.setLayout(gripper_layout)
        main_layout.addWidget(gripper_group)
        
        gripper_control_layout = QHBoxLayout()
        
        # Gripper distance slider
        self.gripper_label = QLabel(f"Gripper Distance: {self.gripper_half_distance * 2 * 1000:.1f} mm (Half: {self.gripper_half_distance * 1000:.1f} mm)")
        gripper_control_layout.addWidget(self.gripper_label)
        
        self.gripper_slider = QSlider(Qt.Horizontal)
        self.gripper_slider.setMinimum(5)    # 最小5mm半宽度 = 10mm总距离
        self.gripper_slider.setMaximum(50)   # 最大50mm半宽度 = 100mm总距离
        self.gripper_slider.setValue(10)     # 默认10mm半宽度 = 20mm总距离
        self.gripper_slider.valueChanged.connect(self.update_gripper_distance)
        gripper_control_layout.addWidget(self.gripper_slider)
        
        gripper_layout.addLayout(gripper_control_layout)
        
        # ===== Offset Settings Area =====
        offset_group = QGroupBox("Offset Settings")
        offset_layout = QVBoxLayout()
        offset_group.setLayout(offset_layout)
        main_layout.addWidget(offset_group)
        
        offset_control_layout = QHBoxLayout()
        
        # X offset setting
        x_offset_layout = QVBoxLayout()
        self.x_offset_label = QLabel("X Offset: 0.0000")
        x_offset_layout.addWidget(self.x_offset_label)
        
        self.x_offset_slider = QSlider(Qt.Horizontal)
        self.x_offset_slider.setMinimum(-3000)  # -0.3
        self.x_offset_slider.setMaximum(3000)   # 0.3
        self.x_offset_slider.setValue(0)
        self.x_offset_slider.valueChanged.connect(self.update_offset)
        x_offset_layout.addWidget(self.x_offset_slider)
        offset_control_layout.addLayout(x_offset_layout)
        
        # Y offset setting
        y_offset_layout = QVBoxLayout()
        self.y_offset_label = QLabel("Y Offset: 0.0000")
        y_offset_layout.addWidget(self.y_offset_label)
        
        self.y_offset_slider = QSlider(Qt.Horizontal)
        self.y_offset_slider.setMinimum(-3000)  # -0.3
        self.y_offset_slider.setMaximum(3000)   # 0.3
        self.y_offset_slider.setValue(0)
        self.y_offset_slider.valueChanged.connect(self.update_offset)
        y_offset_layout.addWidget(self.y_offset_slider)
        offset_control_layout.addLayout(y_offset_layout)
        
        # Z offset setting
        z_offset_layout = QVBoxLayout()
        self.z_offset_label = QLabel("Z Offset: 0.0000")
        z_offset_layout.addWidget(self.z_offset_label)
        
        self.z_offset_slider = QSlider(Qt.Horizontal)
        self.z_offset_slider.setMinimum(-3000)  # -0.3
        self.z_offset_slider.setMaximum(3000)   # 0.3
        self.z_offset_slider.setValue(0)
        self.z_offset_slider.valueChanged.connect(self.update_offset)
        z_offset_layout.addWidget(self.z_offset_slider)
        offset_control_layout.addLayout(z_offset_layout)
        
        offset_layout.addLayout(offset_control_layout)
        
        # Rotation offset settings
        rot_offset_layout = QHBoxLayout()
        
        # X rotation offset setting
        rx_offset_layout = QVBoxLayout()
        self.rx_offset_label = QLabel("X Rotation Offset: 0.0°")
        rx_offset_layout.addWidget(self.rx_offset_label)
        
        self.rx_offset_slider = QSlider(Qt.Horizontal)
        self.rx_offset_slider.setMinimum(-180)  # -180 degrees
        self.rx_offset_slider.setMaximum(180)   # 180 degrees
        self.rx_offset_slider.setValue(0)
        self.rx_offset_slider.valueChanged.connect(self.update_offset)
        rx_offset_layout.addWidget(self.rx_offset_slider)
        rot_offset_layout.addLayout(rx_offset_layout)
        
        # Y rotation offset setting
        ry_offset_layout = QVBoxLayout()
        self.ry_offset_label = QLabel("Y Rotation Offset: 0.0°")
        ry_offset_layout.addWidget(self.ry_offset_label)
        
        self.ry_offset_slider = QSlider(Qt.Horizontal)
        self.ry_offset_slider.setMinimum(-180)  # -180 degrees
        self.ry_offset_slider.setMaximum(180)   # 180 degrees
        self.ry_offset_slider.setValue(0)
        self.ry_offset_slider.valueChanged.connect(self.update_offset)
        ry_offset_layout.addWidget(self.ry_offset_slider)
        rot_offset_layout.addLayout(ry_offset_layout)
        
        # Z rotation offset setting
        rz_offset_layout = QVBoxLayout()
        self.rz_offset_label = QLabel("Z Rotation Offset: 0.0°")
        rz_offset_layout.addWidget(self.rz_offset_label)
        
        self.rz_offset_slider = QSlider(Qt.Horizontal)
        self.rz_offset_slider.setMinimum(-180)  # -180 degrees
        self.rz_offset_slider.setMaximum(180)   # 180 degrees
        self.rz_offset_slider.setValue(0)
        self.rz_offset_slider.valueChanged.connect(self.update_offset)
        rz_offset_layout.addWidget(self.rz_offset_slider)
        rot_offset_layout.addLayout(rz_offset_layout)
        
        offset_layout.addLayout(rot_offset_layout)
        
        # ===== Position Control Area =====
        pos_group = QGroupBox("Position Control")
        pos_layout = QVBoxLayout()
        pos_group.setLayout(pos_layout)
        main_layout.addWidget(pos_group)
        
        # X axis control (DH coordinates)
        x_layout = QHBoxLayout()
        self.x_label = QLabel(f"DH Coordinate X: {self.position[0]:.4f}")
        x_layout.addWidget(self.x_label)
        
        self.x_slider = QSlider(Qt.Horizontal)
        self.x_slider.setMinimum(-3000)  # -0.3
        self.x_slider.setMaximum(3000)   # 0.3
        self.x_slider.setValue(0)
        self.x_slider.valueChanged.connect(self.update_position)
        x_layout.addWidget(self.x_slider)
        pos_layout.addLayout(x_layout)
        
        # Y axis control (DH coordinates)
        y_layout = QHBoxLayout()
        self.y_label = QLabel(f"DH Coordinate Y: {self.position[1]:.4f}")
        y_layout.addWidget(self.y_label)
        
        self.y_slider = QSlider(Qt.Horizontal)
        self.y_slider.setMinimum(-3000)  # -0.3
        self.y_slider.setMaximum(3000)   # 0.3
        self.y_slider.setValue(0)
        self.y_slider.valueChanged.connect(self.update_position)
        y_layout.addWidget(self.y_slider)
        pos_layout.addLayout(y_layout)
        
        # Z axis control (DH coordinates)
        z_layout = QHBoxLayout()
        self.z_label = QLabel(f"DH Coordinate Z: {self.position[2]:.4f}")
        z_layout.addWidget(self.z_label)
        
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setMinimum(-3000)  # -0.3
        self.z_slider.setMaximum(3000)   # 0.3
        self.z_slider.setValue(0)
        self.z_slider.valueChanged.connect(self.update_position)
        z_layout.addWidget(self.z_slider)
        pos_layout.addLayout(z_layout)
        
        # ===== Orientation Control Area =====
        rot_group = QGroupBox("Orientation Control (Rotation Angles)")
        rot_layout = QVBoxLayout()
        rot_group.setLayout(rot_layout)
        main_layout.addWidget(rot_group)
        
        # X rotation control (pitch)
        rx_layout = QHBoxLayout()
        self.rx_label = QLabel(f"X Rotation (Pitch): {self.orientation[0]:.1f}°")
        rx_layout.addWidget(self.rx_label)
        
        self.rx_slider = QSlider(Qt.Horizontal)
        self.rx_slider.setMinimum(-180)  # -180 degrees
        self.rx_slider.setMaximum(180)   # 180 degrees
        self.rx_slider.setValue(0)
        self.rx_slider.valueChanged.connect(self.update_orientation)
        rx_layout.addWidget(self.rx_slider)
        rot_layout.addLayout(rx_layout)
        
        # Y rotation control (yaw)
        ry_layout = QHBoxLayout()
        self.ry_label = QLabel(f"Y Rotation (Yaw): {self.orientation[1]:.1f}°")
        ry_layout.addWidget(self.ry_label)
        
        self.ry_slider = QSlider(Qt.Horizontal)
        self.ry_slider.setMinimum(-180)  # -180 degrees
        self.ry_slider.setMaximum(180)   # 180 degrees
        self.ry_slider.setValue(0)
        self.ry_slider.valueChanged.connect(self.update_orientation)
        ry_layout.addWidget(self.ry_slider)
        rot_layout.addLayout(ry_layout)
        
        # Z rotation control (roll)
        rz_layout = QHBoxLayout()
        self.rz_label = QLabel(f"Z Rotation (Roll): {self.orientation[2]:.1f}°")
        rz_layout.addWidget(self.rz_label)
        
        self.rz_slider = QSlider(Qt.Horizontal)
        self.rz_slider.setMinimum(-120)  # -180 degrees
        self.rz_slider.setMaximum(120)   # 180 degrees
        self.rz_slider.setValue(0)
        self.rz_slider.valueChanged.connect(self.update_orientation)
        rz_layout.addWidget(self.rz_slider)
        rot_layout.addLayout(rz_layout)
        
        # ===== Button Area =====
        button_layout = QHBoxLayout()
        
        # Reset position button
        self.reset_position_button = QPushButton("Reset Position")
        self.reset_position_button.clicked.connect(self.reset_position)
        button_layout.addWidget(self.reset_position_button)
        
        # Reset orientation button
        self.reset_orientation_button = QPushButton("Reset Orientation")
        self.reset_orientation_button.clicked.connect(self.reset_orientation)
        button_layout.addWidget(self.reset_orientation_button)
        
        # Reset offset button
        self.reset_offset_button = QPushButton("Reset All Offsets")
        self.reset_offset_button.clicked.connect(self.reset_offset)
        button_layout.addWidget(self.reset_offset_button)
        
        # Pose comparison button
        self.pose_comparison_button = QPushButton("Show Pose Comparison")
        self.pose_comparison_button.clicked.connect(self.show_pose_comparison)
        button_layout.addWidget(self.pose_comparison_button)
        
        main_layout.addLayout(button_layout)
        
        # Status label
        self.status_label = QLabel("Controller started")
        main_layout.addWidget(self.status_label)
        
        # Create pose comparison window
        self.pose_comparison_window = PoseComparisonWindow()
        self.pose_comparison_window.setVisible(False)
        
        # Create pose update thread
        self.pose_update_thread = PoseUpdateThread(self)
        self.pose_update_thread.update_completed.connect(self.on_update_completed)
        self.pose_update_thread.start()
    
    def on_update_completed(self):
        """Handle when pose update is complete"""
        # Can add post-update logic here
        pass
    
    def show_pose_comparison(self):
        if self.pose_comparison_window.isVisible():
            self.pose_comparison_window.hide()
            self.pose_comparison_button.setText("Show Pose Comparison")
        else:
            self.pose_comparison_window.show()
            self.pose_comparison_button.setText("Hide Pose Comparison")
    
    def update_pose_comparison(self, target_pos, target_ori, estimated_pos, estimated_ori):
        """Update pose comparison data (via thread)"""
        # Add data to update thread queue
        self.pose_update_thread.add_pose_data(
            target_pos, 
            target_ori,
            estimated_pos,
            estimated_ori
        )
    
    def closeEvent(self, event):
        """Stop thread when window closes"""
        self.pose_update_thread.stop()
        super().closeEvent(event)
    
    def update_offset(self):
        # Update position offsets
        self.offset_x = self.x_offset_slider.value() / 10000.0
        self.offset_y = self.y_offset_slider.value() / 10000.0
        self.offset_z = self.z_offset_slider.value() / 10000.0
        
        # Update rotation offsets
        self.offset_rx = self.rx_offset_slider.value()
        self.offset_ry = self.ry_offset_slider.value()
        self.offset_rz = self.rz_offset_slider.value()
        
        # Update label display
        self.x_offset_label.setText(f"X Offset: {self.offset_x:.4f}")
        self.y_offset_label.setText(f"Y Offset: {self.offset_y:.4f}")
        self.z_offset_label.setText(f"Z Offset: {self.offset_z:.4f}")
        self.rx_offset_label.setText(f"X Rotation Offset: {self.offset_rx:.1f}°")
        self.ry_offset_label.setText(f"Y Rotation Offset: {self.offset_ry:.1f}°")
        self.rz_offset_label.setText(f"Z Rotation Offset: {self.offset_rz:.1f}°")
        
        # Also update position and orientation display (offsets affect actual values)
        self.update_position_display()
        self.update_orientation_display()
    
    def update_position(self):
        # Calculate base position (slider value)
        base_x = self.x_slider.value() / 10000.0
        base_y = self.y_slider.value() / 10000.0
        base_z = self.z_slider.value() / 10000.0
        
        # Apply offsets to get actual position
        self.position = np.array([
            base_x + self.offset_x,
            base_y + self.offset_y,
            base_z + self.offset_z
        ])
        
        # Update position display
        self.update_position_display()
        
        self.status_label.setText(f"Position updated: X={self.position[0]:.4f}, Y={self.position[1]:.4f}, Z={self.position[2]:.4f}")
    
    def update_orientation(self):
        # Calculate base rotation angles (slider value)
        base_rx = self.rx_slider.value()
        base_ry = self.ry_slider.value()
        base_rz = self.rz_slider.value()
        
        # Apply offsets to get actual rotation angles
        self.orientation = np.array([
            base_rx + self.offset_rx,
            base_ry + self.offset_ry,
            base_rz + self.offset_rz
        ])
        
        # Update orientation display
        self.update_orientation_display()
        
        self.status_label.setText(f"Orientation updated: RX={self.orientation[0]:.1f}°, RY={self.orientation[1]:.1f}°, RZ={self.orientation[2]:.1f}°")

    def update_gripper_distance(self):
        """Update gripper distance (cube half width)"""
        # Slider value is in mm, convert to meters
        self.gripper_half_distance = self.gripper_slider.value() / 1000.0
        
        # Update label
        total_distance_mm = self.gripper_half_distance * 2 * 1000
        half_distance_mm = self.gripper_half_distance * 1000
        self.gripper_label.setText(f"Gripper Distance: {total_distance_mm:.1f} mm (Half: {half_distance_mm:.1f} mm)")
        
        self.status_label.setText(f"Gripper distance updated: {total_distance_mm:.1f} mm")

    def update_position_display(self):
        """Update position labels"""
        self.x_label.setText(f"DH Coordinate X: {self.position[0]:.4f}")
        self.y_label.setText(f"DH Coordinate Y: {self.position[1]:.4f}")
        self.z_label.setText(f"DH Coordinate Z: {self.position[2]:.4f}")
    
    def update_orientation_display(self):
        """Update orientation labels"""
        self.rx_label.setText(f"X Rotation (Pitch): {self.orientation[0]:.1f}°")
        self.ry_label.setText(f"Y Rotation (Yaw): {self.orientation[1]:.1f}°")
        self.rz_label.setText(f"Z Rotation (Roll): {self.orientation[2]:.1f}°")
    
    def reset_position(self):
        # Reset position sliders
        self.x_slider.setValue(0)
        self.y_slider.setValue(0)
        self.z_slider.setValue(0)
        
        # Reset position (with offsets applied)
        self.position = np.array([
            0.0 + self.offset_x,
            0.0 + self.offset_y,
            0.0 + self.offset_z
        ])
        
        # Update position display
        self.update_position_display()
        self.status_label.setText("Position reset (offsets preserved)")
    
    def reset_orientation(self):
        # Reset rotation sliders
        self.rx_slider.setValue(0)
        self.ry_slider.setValue(0)
        self.rz_slider.setValue(0)
        
        # Reset rotation angles (with offsets applied)
        self.orientation = np.array([
            0.0 + self.offset_rx,
            0.0 + self.offset_ry,
            0.0 + self.offset_rz
        ])
        
        # Update orientation display
        self.update_orientation_display()
        self.status_label.setText("Orientation reset (offsets preserved)")
    
    def reset_offset(self):
        # Reset all offset sliders
        self.x_offset_slider.setValue(0)
        self.y_offset_slider.setValue(0)
        self.z_offset_slider.setValue(0)
        self.rx_offset_slider.setValue(0)
        self.ry_offset_slider.setValue(0)
        self.rz_offset_slider.setValue(0)
        
        # Reset offsets
        self.offset_x = -0.2057
        self.offset_y = 0.0
        self.offset_z = 0.0
        self.offset_rx = 0.0
        self.offset_ry = 0.0
        self.offset_rz = 0.0
        
        # Update offset labels
        self.x_offset_label.setText(f"X Offset: {0.0:.4f}")
        self.y_offset_label.setText(f"Y Offset: {0.0:.4f}")
        self.z_offset_label.setText(f"Z Offset: {0.0:.4f}")
        self.rx_offset_label.setText(f"X Rotation Offset: {0.0:.1f}°")
        self.ry_offset_label.setText(f"Y Rotation Offset: {0.0:.1f}°")
        self.rz_offset_label.setText(f"Z Rotation Offset: {0.0:.1f}°")
        
        # Update position and orientation display (offsets affect actual values)
        self.update_position_display()
        self.update_orientation_display()
        self.status_label.setText("All offsets reset")
    
    def get_position(self):
        """Get position coordinates (X, Y, Z)"""
        return self.position.copy()
    
    def get_orientation(self):
        """Get rotation angles (RX, RY, RZ) in degrees"""
        return self.orientation.copy()

    def get_gripper_half_distance(self):
        """Get gripper half distance (cube half width) in meters"""
        return self.gripper_half_distance
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CubeController()
    window.show()
    sys.exit(app.exec_())