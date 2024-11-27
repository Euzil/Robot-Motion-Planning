import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco
import time

class APFPlanner:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # APF Parameters
        self.k_att = 1.0  # 引力增益
        self.k_rep = 100.0  # 斥力增益
        self.rho_0 = 0.5  # 斥力影响范围
        self.step_size = 0.01  # 步长
        self.max_iterations = 1000  # 最大迭代次数
        self.goal_threshold = 0.01  # 目标阈值
        
        # 障碍物列表 [(position, radius), ...]
        self.obstacles = [
            # 示例球形障碍物
            (np.array([0.4, 0.0, 0.3]), 0.08),
            (np.array([0.4, 0.1, 0.3]), 0.08),
            (np.array([0.4, 0.2, 0.3]), 0.08),
        ]
        
        # 动态添加可视化元素
        self._add_visualization_elements()
    
    def _add_visualization_elements(self):
        """动态添加可视化元素"""
        # 为路径点添加站点
        path_sites = []
        for i in range(self.max_iterations):
            site = {
                'type': mujoco.mjtGeom.mjGEOM_SPHERE,
                'size': np.array([0.01, 0.01, 0.01]),
                'pos': np.zeros(3),
                'rgba': np.array([0, 1, 0, 1])  # 绿色
            }
            path_sites.append(site)
        self.path_sites = path_sites
        
        # 为障碍物添加站点
        obstacle_sites = []
        for obs_pos, obs_radius in self.obstacles:
            site = {
                'type': mujoco.mjtGeom.mjGEOM_SPHERE,
                'size': np.array([obs_radius, obs_radius, obs_radius]),
                'pos': obs_pos,
                'rgba': np.array([1, 0, 0, 0.5])  # 半透明红色
            }
            obstacle_sites.append(site)
        self.obstacle_sites = obstacle_sites
    
    def _attractive_force(self, position, goal):
        """计算引力"""
        diff = goal - position
        distance = np.linalg.norm(diff)
        return self.k_att * diff
    
    def _repulsive_force(self, position):
        """计算所有障碍物产生的斥力"""
        total_force = np.zeros(3)
        
        for obs_pos, obs_radius in self.obstacles:
            diff = position - obs_pos
            distance = np.linalg.norm(diff)
            
            # 仅在影响范围内计算斥力
            if distance <= self.rho_0:
                direction = diff / distance
                magnitude = self.k_rep * (1.0/distance - 1.0/self.rho_0) * (1.0/(distance**2))
                total_force += magnitude * direction
        
        return total_force
    
    def render_scene(self, viewer):
        """渲染场景中的所有可视化元素"""
        # 渲染路径点
        for site in self.path_sites:
            if site['rgba'][3] > 0:  # 只渲染可见的点
                viewer.add_marker(
                    pos=site['pos'],
                    size=site['size'],
                    rgba=site['rgba'],
                    type=site['type']
                )
        
        # 渲染障碍物
        for site in self.obstacle_sites:
            viewer.add_marker(
                pos=site['pos'],
                size=site['size'],
                rgba=site['rgba'],
                type=site['type']
            )
    
    def plan_trajectory(self, start_pos, goal_pos):
        """生成避障轨迹"""
        path = [start_pos]
        current_pos = start_pos.copy()
        
        for i in range(self.max_iterations):
            # 计算合力
            f_att = self._attractive_force(current_pos, goal_pos)
            f_rep = self._repulsive_force(current_pos)
            total_force = f_att + f_rep
            
            # 归一化并应用步长
            force_magnitude = np.linalg.norm(total_force)
            if force_magnitude > 1e-6:
                movement = self.step_size * (total_force / force_magnitude)
                current_pos += movement
            
            # 记录路径
            path.append(current_pos.copy())
            
            # 更新路径点可视化
            if i < len(self.path_sites):
                self.path_sites[i]['pos'] = current_pos
                self.path_sites[i]['rgba'][3] = 1.0  # 设置为可见
            
            # 检查是否达到目标
            if np.linalg.norm(current_pos - goal_pos) < self.goal_threshold:
                break
        
        # 隐藏未使用的路径点
        for i in range(len(path), len(self.path_sites)):
            self.path_sites[i]['rgba'][3] = 0.0
        
        return np.array(path)
