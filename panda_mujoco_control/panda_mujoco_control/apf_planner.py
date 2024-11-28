import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

class ImprovedAPFPlanner:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # 改进的APF参数
        self.k_att = 1.0    # 引力系数
        self.k_rep = 0.3    # 斥力系数
        self.rho_0 = 0.3    # 斥力影响范围
        self.step_size = 0.01  # 步长
        self.max_iterations = 1000
        self.goal_threshold = 0.01
        
        # 势场采样点网格
        self.grid_size = 20  # 每个维度的采样点数
        self.workspace_bounds = {
            'x': (0.1, 0.7),  # 工作空间范围
            'y': (-0.3, 0.3),
            'z': (0.1, 0.7)
        }
        
        # 障碍物配置
        self.obstacles = [
            (np.array([0.4, 0.4, 0.4]), 0.08),
        ]
        
        # 初始化可视化元素
        self._init_visualization()
    
    def _init_visualization(self):
        """初始化势场可视化元素"""
        # 创建势场箭头标记点
        self.force_arrows = []
        x_range = np.linspace(self.workspace_bounds['x'][0], self.workspace_bounds['x'][1], self.grid_size)
        y_range = np.linspace(self.workspace_bounds['y'][0], self.workspace_bounds['y'][1], self.grid_size)
        z_range = np.linspace(self.workspace_bounds['z'][0], self.workspace_bounds['z'][1], self.grid_size)
        
        for x in x_range[::2]:
            for y in y_range[::2]:
                for z in z_range[::2]:
                    pos = np.array([x, y, z])
                    self.force_arrows.append({
                        'pos': pos,
                        'dir': np.zeros(3),
                        'magnitude': 0.0
                    })
    
    def _compute_force_field(self, goal_pos):
        """计算整个工作空间的力场"""
        for arrow in self.force_arrows:
            pos = arrow['pos']
            # 计算引力
            f_att = self._attractive_force(pos, goal_pos)
            # 计算斥力
            f_rep = self._repulsive_force(pos)
            # 合成总力
            total_force = f_att + f_rep
            # 归一化方向向量
            magnitude = np.linalg.norm(total_force)
            if magnitude > 1e-6:
                direction = total_force / magnitude
            else:
                direction = np.zeros(3)
            
            arrow['dir'] = direction
            arrow['magnitude'] = min(magnitude, 0.1)  # 限制箭头长度
    
    def _attractive_force(self, position, goal):
        """计算引力（使用二次型势场）"""
        diff = goal - position
        distance = np.linalg.norm(diff)
        
        if distance <= self.rho_0:
            return self.k_att * diff
        else:
            return self.k_att * self.rho_0 * diff / distance
    
    def _repulsive_force(self, position):
        """计算改进的斥力（考虑障碍物大小）"""
        total_force = np.zeros(3)
        
        for obs_pos, obs_radius in self.obstacles:
            diff = position - obs_pos
            distance = np.linalg.norm(diff)
            
            # 考虑障碍物实际大小
            actual_distance = max(distance - obs_radius, 0.001)
            
            if actual_distance < self.rho_0:
                direction = diff / distance
                magnitude = self.k_rep * (1.0/actual_distance - 1.0/self.rho_0) * (1.0/(actual_distance**2))
                total_force += magnitude * direction
        
        return total_force
    
    def render_force_field(self, viewer):
        """渲染力场可视化"""
        # 渲染势场箭头
        for arrow in self.force_arrows:
            start_pos = arrow['pos']
            direction = arrow['dir']
            magnitude = arrow['magnitude']
            
            if magnitude > 1e-6:
                # 箭头起点
                viewer.add_marker(
                    pos=start_pos,
                    size=np.array([0.005, 0.005, 0.005]),
                    rgba=np.array([0.2, 0.2, 1.0, 0.5]),
                    type=mujoco.mjtGeom.mjGEOM_SPHERE
                )
                
                # 箭头线段
                end_pos = start_pos + direction * magnitude * 0.1
                viewer.add_marker(
                    pos=start_pos,
                    mat=R.from_rotvec(direction).as_matrix(),
                    size=np.array([magnitude * 0.1, 0.002, 0.002]),
                    rgba=np.array([0.2, 0.2, 1.0, 0.5]),
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER
                )
        
        # 渲染障碍物
        for obs_pos, obs_radius in self.obstacles:
            viewer.add_marker(
                pos=obs_pos,
                size=np.array([obs_radius, obs_radius, obs_radius]),
                rgba=np.array([1.0, 0.0, 0.0, 0.5]),
                type=mujoco.mjtGeom.mjGEOM_SPHERE
            )
    
    def plan_trajectory(self, start_pos, goal_pos):
        """生成基于势场的轨迹"""
        path = [start_pos]
        current_pos = start_pos.copy()
        
        # 计算整个工作空间的力场
        self._compute_force_field(goal_pos)
        
        for i in range(self.max_iterations):
            # 计算当前位置的合力
            f_att = self._attractive_force(current_pos, goal_pos)
            f_rep = self._repulsive_force(current_pos)
            total_force = f_att + f_rep
            
            # 使用梯度下降更新位置
            force_magnitude = np.linalg.norm(total_force)
            if force_magnitude > 1e-6:
                movement = self.step_size * (total_force / force_magnitude)
                current_pos += movement
            
            # 记录路径
            if i % 5 == 0:  # 每5步记录一次，减少路径点数量
                path.append(current_pos.copy())
            
            # 检查是否达到目标
            if np.linalg.norm(current_pos - goal_pos) < self.goal_threshold:
                break
        
        path.append(goal_pos)
        return np.array(path)
