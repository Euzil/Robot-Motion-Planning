#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from moveit_msgs.msg import DisplayTrajectory
from sensor_msgs.msg import JointState
import mujoco
import mujoco.viewer
import numpy as np
import time

class APFPlanner:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # 调整APF参数，增大运动范围
        self.k_att = 8000.0      # 增大引力
        self.k_rep = 0.0002     # 增大斥力
        self.rho_0 = 0.4      # 增大影响范围
        self.step_size = 0.005 # 减小步长使运动更平滑
        self.max_iterations = 20000
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
            self.model.site_rgba[site_id][3] = 0.0

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
                    self.model.site_rgba[site_id][3] = 0.0  # 隐藏未使用的点

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

class CartesianController:
    """笛卡尔空间控制器"""
    def __init__(self, model, data, ee_site_name='hand'):
        self.model = model
        self.data = data
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_site_name)
        self.ndof = 7  # Panda机器人的自由度
        
        # 控制参数
        self.damping = 1e-6
        self.max_iterations = 100
        self.tolerance = 1e-2
        
    def get_ee_pose(self):
        """获取末端执行器位置"""
        return self.data.xpos[self.ee_site_id].copy()  # 使用 xpos 而不是 body_xpos
        
    def compute_jacobian(self):
        """计算雅可比矩阵"""
        jacp = np.zeros((3, self.model.nv))  # 只关注位置
        mujoco.mj_jacBody(self.model, self.data, jacp, None, self.ee_site_id)
        return jacp[:, :self.ndof]  # 只使用前7个关节
        
    def solve_ik(self, target_pos, initial_joints=None):
        """使用阻尼最小二乘法求解IK"""
        if initial_joints is not None:
            self.data.qpos[:self.ndof] = initial_joints
            mujoco.mj_forward(self.model, self.data)
        
        for _ in range(self.max_iterations):
            current_pos = self.get_ee_pose()
            error = target_pos - current_pos
            
            
            if np.linalg.norm(error) < self.tolerance:
                return True, self.data.qpos[:self.ndof].copy()
            
            J = self.compute_jacobian()
            JJT = J.dot(J.T)
            damped_JJT = JJT + self.damping * np.eye(3)
            
            # 使用阻尼最小二乘法
            v = np.linalg.solve(damped_JJT, error)
            dq = J.T.dot(v)
            
            # 应用关节角度变化
            self.data.qpos[:self.ndof] += dq
            mujoco.mj_forward(self.model, self.data)
        
        return False, self.data.qpos[:self.ndof].copy()

class MujocoROS2Bridge(Node):
    def __init__(self):
        super().__init__('mujoco_ros2_bridge')
        
        self.model_path = "/home/youran/.mujoco/mujoco210/model/panda.xml"
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        
        # 初始化控制器
        self.controller = CartesianController(self.model, self.data)
        
        # 设置初始姿态
        initial_joints = [0, -0.3, 0, -2.2, 0, 2.0, 0.78539816339]
        self.data.qpos[:7] = initial_joints
        mujoco.mj_forward(self.model, self.data)
        
        self.planner = APFPlanner(self.model, self.data)
        
        # 设置关节信息
        self.joint_names = []
        self.joint_indices = {}
        for i in range(7):  # 只使用主要关节
            name = f'joint{i+1}'
            self.joint_names.append(name)
            self.joint_indices[name] = i
        
        self.joint_state_pub = self.create_publisher(
            JointState, 
            'joint_states', 
            10
        )
        
        self.timer = self.create_timer(0.01, self.publish_joint_states)
        
        self.trajectory_sub = self.create_subscription(
            DisplayTrajectory,
            'display_planned_path',
            self.trajectory_callback,
            10
        )
        
        self.viewer = None
        self.get_logger().info('MuJoCo-ROS2 桥接节点已初始化')

    def initialize_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.get_logger().info('MuJoCo 查看器已启动')

    def update_viewer(self):
        if self.viewer is not None:
            self.viewer.sync()

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        
        positions = []
        velocities = []
        for name in self.joint_names:
            idx = self.joint_indices[name]
            positions.append(self.data.qpos[idx])
            velocities.append(self.data.qvel[idx])
        
        msg.position = positions
        msg.velocity = velocities
        
        self.joint_state_pub.publish(msg)
        self.update_viewer()

    def execute_path(self, path):
        """执行路径，确保末端执行器跟随规划路径"""
        # 降采样路径点
        if len(path) > 50:
            indices = np.linspace(0, len(path)-1, 50).astype(int)
            path = path[indices]
        
        current_joints = self.data.qpos[:7].copy()
        
        for target_pos in path:
            success, new_joints = self.controller.solve_ik(target_pos, current_joints)
            
            if success:
                # 平滑过渡到新的关节角度
                steps = 10
                for i in range(steps):
                    alpha = (i + 1) / steps
                    self.data.qpos[:7] = current_joints * (1-alpha) + new_joints * alpha
                    mujoco.mj_step(self.model, self.data)
                    self.update_viewer()
                    time.sleep(0.02)
                
                current_joints = new_joints
            else:
                self.get_logger().warn(f'IK求解失败，目标位置: {target_pos}')

    def trajectory_callback(self, msg: DisplayTrajectory):
        self.get_logger().info('开始规划和执行轨迹')
        
        try:
            self.initialize_viewer()
            
            # 使用当前末端执行器位置作为起点
            start_pos = np.array([0.7, 0.4, 0.4])
            goal_pos = np.array([0.1, 0.4, 0.4])  # 适当的目标位置
            
            self.get_logger().info(f'起点: {start_pos}')
            self.get_logger().info(f'终点: {goal_pos}')
            
            # 规划并执行路径
            path = self.planner.plan_trajectory(start_pos, goal_pos)
            self.get_logger().info(f'已生成路径点数量: {len(path)}')
            
            self.execute_path(path)
            
        except Exception as e:
            self.get_logger().error(f'执行失败: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

def main(args=None):
    rclpy.init(args=args)
    bridge = None
    
    try:
        bridge = MujocoROS2Bridge()
        bridge.initialize_viewer()
        bridge.get_logger().info('开始运行...')
        
        # 延迟一下再开始执行，让机器人和可视化完全初始化
        time.sleep(1)
        bridge.trajectory_callback(None)
        
        rclpy.spin(bridge)
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        if bridge and hasattr(bridge, 'viewer') and bridge.viewer is not None:
            bridge.viewer.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
