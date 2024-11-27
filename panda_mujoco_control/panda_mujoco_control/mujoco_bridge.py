#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from moveit_msgs.msg import DisplayTrajectory
from sensor_msgs.msg import JointState
import mujoco
import mujoco.viewer
import numpy as np
import time
from panda_mujoco_control.apf_planner import ImprovedAPFPlanner


class CartesianController:
    """改进的笛卡尔空间控制器"""
    def __init__(self, model, data, ee_site_name='hand'):
        self.model = model
        self.data = data
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_site_name)
        self.ndof = 7
        
        # 优化控制参数
        self.damping = 1e-3      # 增大阻尼系数
        self.max_iterations = 150
        self.tolerance = 1e-2    # 放宽容差
        
        # 添加关节限位
        self.joint_limits = {
            'lower': np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
            'upper': np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        }
    def get_ee_pose(self):
        """获取末端执行器位置"""
        return self.data.xpos[self.ee_site_id].copy()
        
    def compute_jacobian(self):
        """计算雅可比矩阵"""
        jacp = np.zeros((3, self.model.nv))  # 只关注位置
        mujoco.mj_jacBody(self.model, self.data, jacp, None, self.ee_site_id)
        return jacp[:, :self.ndof]  # 只使用前7个关节
        
    def solve_ik(self, target_pos, initial_joints=None):
        """改进的IK求解器"""
        if initial_joints is not None:
            self.data.qpos[:self.ndof] = initial_joints
            mujoco.mj_forward(self.model, self.data)
        
        current_joints = self.data.qpos[:self.ndof].copy()
        
        for _ in range(self.max_iterations):
            current_pos = self.get_ee_pose()
            error = target_pos - current_pos
            
            if np.linalg.norm(error) < self.tolerance:
                return True, current_joints
            
            J = self.compute_jacobian()
            JJT = J.dot(J.T)
            damped_JJT = JJT + self.damping * np.eye(3)
            
            try:
                v = np.linalg.solve(damped_JJT, error)
                dq = J.T.dot(v)
                
                # 限制步长
                step_size = 0.05
                if np.linalg.norm(dq) > step_size:
                    dq = step_size * dq / np.linalg.norm(dq)
                
                # 应用关节限位
                new_joints = current_joints + dq
                new_joints = np.clip(new_joints, self.joint_limits['lower'], self.joint_limits['upper'])
                
                # 检查是否有效改进
                if np.allclose(new_joints, current_joints):
                    return False, current_joints
                
                current_joints = new_joints
                self.data.qpos[:self.ndof] = current_joints
                mujoco.mj_forward(self.model, self.data)
                
            except np.linalg.LinAlgError:
                return False, current_joints
        
        return False, current_joints

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
        
        # 使用改进的APF规划器
        self.planner = ImprovedAPFPlanner(self.model, self.data)
        
        # 设置关节信息
        self.joint_names = []
        self.joint_indices = {}
        for i in range(7):
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
        """更新查看器状态"""
        if self.viewer is not None:
            mujoco.mj_forward(self.model, self.data)
            if hasattr(self.planner, 'update_visualization'):
                self.planner.update_visualization()
            self.viewer.sync()

    def publish_joint_states(self):
        """发布关节状态信息"""
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
        
    
    def is_target_reachable(self, target_pos):
        """检查目标点是否在工作空间内"""
        # 简单的球形工作空间检查
        base_pos = np.array([0, 0, 0])  # 机器人基座位置
        max_reach = 0.855  # Panda的最大伸展距离
        distance = np.linalg.norm(target_pos - base_pos)
        return distance < max_reach
    
    def is_joint_valid(self, joints):
        """检查关节配置是否有效"""
        # 检查关节限位
        lower = self.controller.joint_limits['lower']
        upper = self.controller.joint_limits['upper']
        
        if not np.all((joints >= lower) & (joints <= upper)):
            return False
        
        # 检查是否包含NaN值
        if np.any(np.isnan(joints)):
            return False
        
        return True
    
    def trajectory_callback(self, msg: DisplayTrajectory):
        self.get_logger().info('开始规划和执行轨迹')
        
        try:
            self.initialize_viewer()
            
            # 修改起点和终点，使路径更合理
            start_pos = np.array([0.6, 0.4, 0.4])    # 保持起点在机器人前方
            goal_pos = np.array([0.1, 0.4, 0.4])     # 终点只在Y方向有变化，便于测试
            
            self.get_logger().info(f'起点: {start_pos}')
            self.get_logger().info(f'终点: {goal_pos}')
            
            # 设置APF参数
            self.planner.k_att = 2.0        # 减小引力
            self.planner.k_rep = 0.5        # 减小斥力
            self.planner.step_size = 0.01   # 减小步长
            self.planner.rho_0 = 0.3        # 减小影响范围
            
            # 调整障碍物位置
            self.planner.obstacles = [
                (np.array([0.4, 0.15, 0.4]), 0.05),  # 在起点和终点之间放置一个小障碍物
            ]
            
            # 生成路径
            path = self.planner.plan_trajectory(start_pos, goal_pos)
            
            # 对路径进行平滑处理
            smoothed_path = self.smooth_path(path)
            
            self.get_logger().info(f'已生成路径点数量: {len(smoothed_path)}')
            
            # 执行路径
            self.execute_path(smoothed_path)
            
        except Exception as e:
            self.get_logger().error(f'执行失败: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def smooth_path(self, path):
        """平滑路径，去除不必要的振荡"""
        if len(path) < 3:
            return path
            
        smoothed = [path[0]]  # 保留起点
        
        # 使用移动平均进行平滑
        window_size = 5
        for i in range(1, len(path)-1):
            start_idx = max(0, i - window_size//2)
            end_idx = min(len(path), i + window_size//2 + 1)
            window = path[start_idx:end_idx]
            avg_point = np.mean(window, axis=0)
            
            # 确保平滑后的点仍然在工作空间内
            if self.is_target_reachable(avg_point):
                smoothed.append(avg_point)
        
        smoothed.append(path[-1])  # 保留终点
        return np.array(smoothed)
    
    def execute_path(self, path):
        """改进的路径执行"""
        if len(path) > 50:
            indices = np.linspace(0, len(path)-1, 50).astype(int)
            path = path[indices]
        
        current_joints = self.data.qpos[:7].copy()
        last_successful_joints = current_joints.copy()
        
        for target_pos in path:
            if not self.is_target_reachable(target_pos):
                self.get_logger().warn(f'目标点超出工作空间: {target_pos}')
                continue
            
            success, new_joints = self.controller.solve_ik(target_pos, current_joints)
            
            if success:
                try:
                    steps = 20
                    for i in range(steps):
                        alpha = (i + 1) / steps
                        interpolated_joints = current_joints * (1-alpha) + new_joints * alpha
                        
                        if self.is_joint_valid(interpolated_joints):
                            self.data.qpos[:7] = interpolated_joints
                            mujoco.mj_step(self.model, self.data)
                            self.update_viewer()
                            time.sleep(0.02)
                            last_successful_joints = interpolated_joints.copy()
                        else:
                            # 如果插值点无效，回退到上一个成功的构型
                            self.data.qpos[:7] = last_successful_joints
                            break
                    
                    current_joints = new_joints
                except Exception as e:
                    self.get_logger().warn(f'路径执行出错: {str(e)}')
                    # 回退到上一个成功的构型
                    self.data.qpos[:7] = last_successful_joints
            else:
                self.get_logger().warn(f'IK求解失败，目标位置: {target_pos}')

def main(args=None):
    rclpy.init(args=args)
    bridge = None
    
    try:
        bridge = MujocoROS2Bridge()
        bridge.initialize_viewer()
        bridge.get_logger().info('开始运行...')
        
        # 延迟初始化
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
