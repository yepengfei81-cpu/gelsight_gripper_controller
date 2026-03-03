"""
双臂独立控制界面
分别控制左右臂的末端位姿
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QLabel, QSlider, QDoubleSpinBox, QGridLayout, QPushButton
)
from PyQt5.QtCore import Qt


class DualArmController(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("双臂末端位姿控制")
        self.setMinimumWidth(800)
        
        # 左臂初始位姿
        self.left_pos = np.array([-0.2057, 0.0, 0.0152])
        self.left_ori = np.array([180.0, 0.0, -180.0])
        
        # 右臂初始位姿
        self.right_pos = np.array([-0.2057, 0.0, -0.0152])
        self.right_ori = np.array([0.0, 0.0, -180.0])
        
        # 当前实际位姿（用于显示）
        self.current_left_pos = self.left_pos.copy()
        self.current_left_ori = self.left_ori.copy()
        self.current_right_pos = self.right_pos.copy()
        self.current_right_ori = self.right_ori.copy()
        
        self._init_ui()
        
    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        
        # 左臂控制
        left_group = self._create_arm_control("左臂", "left")
        main_layout.addWidget(left_group)
        
        # 右臂控制
        right_group = self._create_arm_control("右臂", "right")
        main_layout.addWidget(right_group)
        
    def _create_arm_control(self, title, arm_id):
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        
        # 位置控制
        pos_group = QGroupBox("位置 (m)")
        pos_layout = QGridLayout(pos_group)
        
        pos_labels = ['X', 'Y', 'Z']
        pos_ranges = [(-0.3, 0.0), (-0.15, 0.15), (-0.15, 0.15)]
        
        if arm_id == "left":
            init_pos = self.left_pos
        else:
            init_pos = self.right_pos
            
        pos_spinboxes = []
        for i, (label, (min_val, max_val)) in enumerate(zip(pos_labels, pos_ranges)):
            lbl = QLabel(f"{label}:")
            spin = QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(0.001)
            spin.setDecimals(4)
            spin.setValue(init_pos[i])
            spin.valueChanged.connect(lambda v, idx=i, aid=arm_id: self._on_pos_changed(aid, idx, v))
            pos_layout.addWidget(lbl, i, 0)
            pos_layout.addWidget(spin, i, 1)
            pos_spinboxes.append(spin)
            
        if arm_id == "left":
            self.left_pos_spinboxes = pos_spinboxes
        else:
            self.right_pos_spinboxes = pos_spinboxes
            
        layout.addWidget(pos_group)
        
        # 姿态控制
        ori_group = QGroupBox("姿态 (度)")
        ori_layout = QGridLayout(ori_group)
        
        ori_labels = ['Roll', 'Pitch', 'Yaw']
        
        if arm_id == "left":
            init_ori = self.left_ori
        else:
            init_ori = self.right_ori
            
        ori_spinboxes = []
        for i, label in enumerate(ori_labels):
            lbl = QLabel(f"{label}:")
            spin = QDoubleSpinBox()
            spin.setRange(-180, 180)
            spin.setSingleStep(1.0)
            spin.setDecimals(1)
            spin.setValue(init_ori[i])
            spin.valueChanged.connect(lambda v, idx=i, aid=arm_id: self._on_ori_changed(aid, idx, v))
            ori_layout.addWidget(lbl, i, 0)
            ori_layout.addWidget(spin, i, 1)
            ori_spinboxes.append(spin)
            
        if arm_id == "left":
            self.left_ori_spinboxes = ori_spinboxes
        else:
            self.right_ori_spinboxes = ori_spinboxes
            
        layout.addWidget(ori_group)
        
        # 当前位姿显示
        current_group = QGroupBox("当前位姿")
        current_layout = QGridLayout(current_group)
        
        current_labels = []
        for i, name in enumerate(['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']):
            lbl = QLabel(f"{name}: 0.0000")
            current_layout.addWidget(lbl, i // 3, i % 3)
            current_labels.append(lbl)
            
        if arm_id == "left":
            self.left_current_labels = current_labels
        else:
            self.right_current_labels = current_labels
            
        layout.addWidget(current_group)
        
        # 复位按钮
        reset_btn = QPushButton("复位")
        reset_btn.clicked.connect(lambda: self._reset_arm(arm_id))
        layout.addWidget(reset_btn)
        
        return group
        
    def _on_pos_changed(self, arm_id, idx, value):
        if arm_id == "left":
            self.left_pos[idx] = value
        else:
            self.right_pos[idx] = value
            
    def _on_ori_changed(self, arm_id, idx, value):
        if arm_id == "left":
            self.left_ori[idx] = value
        else:
            self.right_ori[idx] = value
            
    def _reset_arm(self, arm_id):
        if arm_id == "left":
            self.left_pos = np.array([-0.2057, 0.0, 0.0152])
            self.left_ori = np.array([180.0, 0.0, -180.0])
            for i, spin in enumerate(self.left_pos_spinboxes):
                spin.setValue(self.left_pos[i])
            for i, spin in enumerate(self.left_ori_spinboxes):
                spin.setValue(self.left_ori[i])
        else:
            self.right_pos = np.array([-0.2057, 0.0, -0.0152])
            self.right_ori = np.array([0.0, 0.0, -180.0])
            for i, spin in enumerate(self.right_pos_spinboxes):
                spin.setValue(self.right_pos[i])
            for i, spin in enumerate(self.right_ori_spinboxes):
                spin.setValue(self.right_ori[i])
                
    def get_left_arm_pose(self):
        """获取左臂目标位姿"""
        return self.left_pos.copy(), self.left_ori.copy()
        
    def get_right_arm_pose(self):
        """获取右臂目标位姿"""
        return self.right_pos.copy(), self.right_ori.copy()
        
    def update_current_poses(self, left_pos, left_ori, right_pos, right_ori):
        """更新当前实际位姿显示"""
        names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
        
        for i in range(3):
            self.left_current_labels[i].setText(f"{names[i]}: {left_pos[i]:.4f}")
            self.left_current_labels[i+3].setText(f"{names[i+3]}: {left_ori[i]:.1f}")
            self.right_current_labels[i].setText(f"{names[i]}: {right_pos[i]:.4f}")
            self.right_current_labels[i+3].setText(f"{names[i+3]}: {right_ori[i]:.1f}")