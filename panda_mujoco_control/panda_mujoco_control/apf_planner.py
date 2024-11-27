import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco
import time

class APFPlanner:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # 调整APF参数，增大运动范围
        self.k_att = 2500.0      # 增大引力
        self.k_rep = 0.0001     # 增大斥力
        self.rho_0 = 0.002      # 增大影响范围
        self.step_size = 0.005 # 减小步长使运动更平滑
        self.max_iterations = 2000
        self.goal_threshold = 0.01
        
        # 障碍物位置调整到路径中间
        self.obstacles = [
            #(np.array([0.4, 0.0, 0.4]), 0.08),
            #(np.array([0.4, 0.1, 0.4]), 0.02),
            #(np.array([0.4, 0.2, 0.4]), 0.08)
            #(np.array([0.4, 0.3, 0.4]), 0.08),
            (np.array([0.4, 0.4, 0.4]), 0.08),
            #(np.array([0.4, 0.5, 0.4]), 0.08),
            #(np.array([0.4, 0.6, 0.4]), 0.08),
            #(np.array([0.4, 0.7, 0.4]), 0.08),
            #(np.array([0.4, 0.8, 0.4]), 0.08),
             
        ]
        
        # 获取站点ID
        self.path_ids = []
        for i in range(100):
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f'path_{i}')
            if site_id != -1:
                self.path_ids.append(site_id)
        
        self.start_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'start_point')
        self.goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'goal_point')
        
        # 初始化时确保所有路径点不可见
        self.hide_all_path_points()

    def hide_all_path_points(self):
        """隐藏所有路径点"""
        for site_id in self.path_ids:
            self.model.site_rgba[site_id][3] = 0.5

    def update_visualization(self, path):
        """更新路径可视化"""
        # 更新起点和终点标记，确保可见
        if self.start_site_id != -1:
            self.model.site_pos[self.start_site_id] = path[0]
            self.model.site_rgba[self.start_site_id][3] = 1.0  # 确保可见
        if self.goal_site_id != -1:
            self.model.site_pos[self.goal_site_id] = path[-1]
            self.model.site_rgba[self.goal_site_id][3] = 1.0   # 确保可见
        
        # 更新路径点
        path_points = len(self.path_ids)
        if len(path) > 2 and path_points > 0:
            indices = np.linspace(1, len(path)-2, path_points).astype(int)
            for i, site_id in enumerate(self.path_ids):
                if i < len(indices):
                    self.model.site_pos[site_id] = path[indices[i]]
                    self.model.site_rgba[site_id][3] = 1.0  # 显示路径点
                else:
                    self.model.site_rgba[site_id][3] = 1.0  # 隐藏未使用的点

    def _attractive_force(self, position, goal):
        """计算引力"""
        diff = goal - position
        distance = np.linalg.norm(diff)
        if distance > self.rho_0:
            return self.k_att * self.rho_0 * diff / distance
        return self.k_att * diff

    def _repulsive_force(self, position):
        """计算所有障碍物产生的斥力"""
        total_force = np.zeros(3)
        for obs_pos, obs_radius in self.obstacles:
            diff = position - obs_pos
            distance = np.linalg.norm(diff)
            
            # 考虑障碍物半径
            distance = max(distance - obs_radius, 0.001)  # 避免除零
            
            if distance < self.rho_0:
                direction = diff / (distance + obs_radius)
                magnitude = self.k_rep * (1.0/distance - 1.0/self.rho_0) * (1.0/(distance**2))
                total_force += magnitude * direction        
        return total_force

    def plan_trajectory(self, start_pos, goal_pos):
        """生成避障轨迹"""
        path = [start_pos]
        current_pos = start_pos.copy()
        
        for i in range(self.max_iterations):
            f_att = self._attractive_force(current_pos, goal_pos)
            f_rep = self._repulsive_force(current_pos)
            total_force = f_att + f_rep
            force_magnitude = np.linalg.norm(total_force)
            if force_magnitude > 1e-6:
                movement = self.step_size * (total_force / force_magnitude)
                current_pos += movement
            
            if i % 5 == 0:  # 每5步记录一个路径点
                path.append(current_pos.copy())
                print('path : ', current_pos.copy())
            
            if np.linalg.norm(current_pos - goal_pos) < self.goal_threshold:
                break
        
        path.append(goal_pos)
        self.update_visualization(path)
        return np.array(path)
