#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.action import ExecuteTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

class MujocoROS2Bridge(Node):
    def __init__(self):
        super().__init__('mujoco_ros2_bridge')
        
        # 初始化 MuJoCo
        self.model_path = "/home/youran/.mujoco/mujoco210/model/panda.xml"
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)
        
        # 记录关节名称和索引
        self.joint_names = []
        self.joint_indices = {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.joint_names.append(name)
                self.joint_indices[name] = i
        
        # ROS2 发布器和订阅器
        self.joint_state_pub = self.create_publisher(
            JointState, 
            'joint_states', 
            10
        )
        
        # MoveIt2 执行轨迹的动作客户端
        self.execute_trajectory_client = ActionClient(
            self, 
            ExecuteTrajectory, 
            'execute_trajectory'
        )
        
        # 创建定时器，定期发布关节状态
        self.timer = self.create_timer(0.01, self.publish_joint_states)
        
        # 状态订阅
        self.trajectory_sub = self.create_subscription(
            DisplayTrajectory,
            'display_planned_path',
            self.trajectory_callback,
            10
        )
        
        # 初始化 MuJoCo 查看器
        self.viewer = None
        
        self.get_logger().info('MuJoCo-ROS2 桥接节点已初始化')
        self.get_logger().info(f'已识别的关节: {", ".join(self.joint_names)}')

    def initialize_viewer(self):
        """初始化 MuJoCo 查看器"""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.get_logger().info('MuJoCo 查看器已启动')

    def update_viewer(self):
        """更新查看器"""
        if self.viewer is not None:
            self.viewer.sync()

    def publish_joint_states(self):
        """发布当前关节状态"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        
        # 获取关节位置和速度
        positions = []
        velocities = []
        for name in self.joint_names:
            idx = self.joint_indices[name]
            positions.append(self.data.qpos[idx])
            velocities.append(self.data.qvel[idx])
        
        msg.position = positions
        msg.velocity = velocities
        
        self.joint_state_pub.publish(msg)
        
        # 更新查看器
        self.update_viewer()

    def trajectory_callback(self, msg: DisplayTrajectory):
        """处理来自 MoveIt2 的轨迹"""
        self.get_logger().info('收到新的轨迹')
        
        try:
            # 确保查看器已初始化
            self.initialize_viewer()
            
            # 获取轨迹
            trajectory = msg.trajectory[0].joint_trajectory
            self.get_logger().info(f'轨迹包含 {len(trajectory.points)} 个点')
            
            # 创建关节名称到索引的映射
            traj_joint_indices = {name: i for i, name in enumerate(trajectory.joint_names)}
            
            # 遍历轨迹点
            for i, point in enumerate(trajectory.points):
                self.get_logger().info(f'执行轨迹点 {i+1}/{len(trajectory.points)}')
                
                # 为每个关节更新位置
                new_qpos = self.data.qpos.copy()
                new_qvel = self.data.qvel.copy()
                
                for joint_name in self.joint_indices.keys():
                    if joint_name in traj_joint_indices:
                        traj_idx = traj_joint_indices[joint_name]
                        mj_idx = self.joint_indices[joint_name]
                        new_qpos[mj_idx] = point.positions[traj_idx]
                        if point.velocities:
                            new_qvel[mj_idx] = point.velocities[traj_idx]
                        else:
                            new_qvel[mj_idx] = 0.0
                
                # 更新 MuJoCo 状态
                self.data.qpos[:] = new_qpos
                self.data.qvel[:] = new_qvel
                
                # 运行 MuJoCo 仿真步进
                mujoco.mj_step(self.model, self.data)
                
                # 更新查看器
                self.update_viewer()
                
                # 模拟实时执行
                time.sleep(0.01)
                
        except Exception as e:
            self.get_logger().error(f'执行轨迹时出错: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def execute_trajectory_async(self, trajectory: JointTrajectory):
        """异步执行轨迹"""
        goal_msg = ExecuteTrajectory.Goal()
        goal_msg.trajectory = trajectory
        
        self.execute_trajectory_client.wait_for_server()
        
        return self.execute_trajectory_client.send_goal_async(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        bridge = MujocoROS2Bridge()
        
        # 确保查看器被初始化
        bridge.initialize_viewer()
        bridge.get_logger().info('开始运行...')
        
        rclpy.spin(bridge)
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 清理查看器
        if bridge.viewer is not None:
            bridge.viewer.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
