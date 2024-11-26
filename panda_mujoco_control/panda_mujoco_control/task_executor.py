#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from moveit_msgs.msg import DisplayTrajectory
from geometry_msgs.msg import Pose, Point, Quaternion
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, 
    WorkspaceParameters, 
    Constraints,
    JointConstraint,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
)
from geometry_msgs.msg import PoseStamped, Vector3
from shape_msgs.msg import SolidPrimitive
from rclpy.executors import MultiThreadedExecutor

class TaskExecutor(Node):
    def __init__(self):
        super().__init__('task_executor')
        
        self.callback_group = ReentrantCallbackGroup()
        
        # 创建动作客户端
        self.move_action_client = ActionClient(
            self,
            MoveGroup,
            'move_action',
            callback_group=self.callback_group
        )
        
        # 等待动作服务器
        self.move_action_client.wait_for_server()
        
        self.get_logger().info('任务执行器已初始化')

    def create_constraints(self, position, orientation=None):
        """创建运动约束"""
        constraints = Constraints()
        
        # 位置约束
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = "panda_link0"
        position_constraint.link_name = "panda_hand"
        
        # 定义约束区域
        bounding_volume = BoundingVolume()
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.01]  # 1cm 的球形约束
        
        bounding_volume.primitives.append(primitive)
        
        primitive_pose = Pose()
        primitive_pose.position.x = position[0]
        primitive_pose.position.y = position[1]
        primitive_pose.position.z = position[2]
        primitive_pose.orientation.w = 1.0
        
        bounding_volume.primitive_poses.append(primitive_pose)
        
        position_constraint.constraint_region = bounding_volume
        position_constraint.weight = 1.0
        
        constraints.position_constraints.append(position_constraint)
        
        # 姿态约束
        if orientation is None:
            orientation = R.from_euler('xyz', [np.pi, 0, 0]).as_quat()
            
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = "panda_link0"
        orientation_constraint.orientation.x = orientation[0]
        orientation_constraint.orientation.y = orientation[1]
        orientation_constraint.orientation.z = orientation[2]
        orientation_constraint.orientation.w = orientation[3]
        orientation_constraint.link_name = "panda_hand"
        orientation_constraint.absolute_x_axis_tolerance = 0.1
        orientation_constraint.absolute_y_axis_tolerance = 0.1
        orientation_constraint.absolute_z_axis_tolerance = 0.1
        orientation_constraint.weight = 1.0
        
        constraints.orientation_constraints.append(orientation_constraint)
        
        return constraints

    async def plan_to_pose(self, position, orientation=None):
        """规划到指定位姿"""
        try:
            # 创建运动规划请求
            goal_msg = MoveGroup.Goal()
            motion_request = MotionPlanRequest()
            
            # 设置工作空间参数
            workspace = WorkspaceParameters()
            workspace.header.frame_id = "panda_link0"
            workspace.header.stamp = self.get_clock().now().to_msg()
            workspace.min_corner.x = -1.0
            workspace.min_corner.y = -1.0
            workspace.min_corner.z = -1.0
            workspace.max_corner.x = 1.0
            workspace.max_corner.y = 1.0
            workspace.max_corner.z = 1.0
            motion_request.workspace_parameters = workspace
            
            # 设置约束
            motion_request.goal_constraints = [self.create_constraints(position, orientation)]
            
            # 设置规划组
            motion_request.group_name = "panda_arm"
            
            # 设置规划时间
            motion_request.allowed_planning_time = 5.0
            
            # 设置最大速度缩放因子
            motion_request.max_velocity_scaling_factor = 0.3
            motion_request.max_acceleration_scaling_factor = 0.3
            
            goal_msg.request = motion_request
            
            # 发送目标并等待结果
            self.get_logger().info('发送运动规划请求')
            send_goal_future = await self.move_action_client.send_goal_async(goal_msg)
            
            if not send_goal_future.accepted:
                self.get_logger().error('目标被拒绝')
                return False
                
            self.get_logger().info('目标被接受，等待执行结果')
            goal_handle = send_goal_future.result()
            result_future = await goal_handle.get_result_async()
            
            success = result_future.result().success
            if success:
                self.get_logger().info('运动执行成功')
            else:
                self.get_logger().error('运动执行失败')
            return success
            
        except Exception as e:
            self.get_logger().error(f'规划失败: {str(e)}')
            return False

    async def execute_picking_task(self):
        """执行抓取任务"""
        try:
            # 1. 移动到准备位置
            self.get_logger().info('移动到准备位置')
            if not await self.plan_to_pose([0.3, 0.0, 0.5]):
                return False
            
            # 2. 移动到物体上方
            self.get_logger().info('移动到物体上方')
            if not await self.plan_to_pose([0.4, 0.2, 0.4]):
                return False
                
            # 3. 下降到抓取位置
            self.get_logger().info('下降到抓取位置')
            if not await self.plan_to_pose([0.4, 0.2, 0.2]):
                return False
                
            # 4. 提起物体
            self.get_logger().info('提起物体')
            if not await self.plan_to_pose([0.4, 0.2, 0.4]):
                return False
                
            # 5. 移动到放置位置
            self.get_logger().info('移动到放置位置')
            if not await self.plan_to_pose([0.4, -0.2, 0.4]):
                return False
                
            # 6. 放下物体
            self.get_logger().info('放下物体')
            if not await self.plan_to_pose([0.4, -0.2, 0.2]):
                return False
                
            # 7. 返回准备位置
            self.get_logger().info('返回准备位置')
            if not await self.plan_to_pose([0.3, 0.0, 0.5]):
                return False
                
            return True
            
        except Exception as e:
            self.get_logger().error(f'执行任务时出错: {str(e)}')
            return False

def main(args=None):
    rclpy.init(args=args)
    
    try:
        executor = TaskExecutor()
        
        # 使用多线程执行器
        rclpy_executor = MultiThreadedExecutor()
        rclpy_executor.add_node(executor)
        
        # 创建并运行任务
        import asyncio
        loop = asyncio.get_event_loop()
        
        # 运行主任务
        loop.run_until_complete(executor.execute_picking_task())
        
        try:
            rclpy_executor.spin()
        except KeyboardInterrupt:
            pass
            
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
