#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from moveit_msgs.msg import DisplayTrajectory
from sensor_msgs.msg import JointState
import mujoco
import mujoco.viewer
import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt
from panda_mujoco_control.apf_planner import ImprovedAPFPlanner


class CartesianController:
    """Cartesian Space Controller"""
    def __init__(self, model, data, ee_site_name='hand'):
        self.model = model
        self.data = data
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_site_name)
        self.ndof = 7
        
        # Optimize control parameters
        self.damping = 1e-3      # Increase damping coefficient
        self.max_iterations = 150
        self.tolerance = 1e-2    # relax tolerance
        
        # Add joint limits
        self.joint_limits = {
            'lower': np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
            'upper': np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        }
    def get_ee_pose(self):
        """Get the end effector position"""
        return self.data.xpos[self.ee_site_id].copy()
        
    def compute_jacobian(self):
        """Calculate the Jacobian matrix"""
        jacp = np.zeros((3, self.model.nv))  
        mujoco.mj_jacBody(self.model, self.data, jacp, None, self.ee_site_id)
        return jacp[:, :self.ndof] 
        
    def solve_ik(self, target_pos, initial_joints=None):
        """IK solver"""
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
                
                # Limit step size
                step_size = 0.05
                if np.linalg.norm(dq) > step_size:
                    dq = step_size * dq / np.linalg.norm(dq)
                
                # Apply joint limits
                new_joints = current_joints + dq
                new_joints = np.clip(new_joints, self.joint_limits['lower'], self.joint_limits['upper'])
                
                # Check whether the improvement is effective
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
        
        #Initialize the controller
        self.controller = CartesianController(self.model, self.data)
        
        # Set initial posture
        initial_joints = [0, -0.3, 0, -2.2, 0, 2.0, 0.78539816339]
        self.data.qpos[:7] = initial_joints
        mujoco.mj_forward(self.model, self.data)
        
        # Use the APF planner
        self.planner = ImprovedAPFPlanner(self.model, self.data)
        
        #Set joint information
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
        self.get_logger().info('MuJoCo-ROS2 bridge node initialized')

    def initialize_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.get_logger().info('MuJoCo viewer started')

    def update_viewer(self):
        """Update viewer status"""
        if self.viewer is not None:
            mujoco.mj_forward(self.model, self.data)
            if hasattr(self.planner, 'update_visualization'):
                self.planner.update_visualization()
            self.viewer.sync()

    def publish_joint_states(self):
        """Publish joint status information"""
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
        """Check whether the target point is within the workspace"""
        # Simple spherical workspace check
        base_pos = np.array([0, 0, 0])  #Robot base position
        max_reach = 0.855  # Maximum stretch distance of Panda
        distance = np.linalg.norm(target_pos - base_pos)
        return distance < max_reach
    
    def is_joint_valid(self, joints):
        """Check whether the joint configuration is valid"""
        # Check joint limits
        lower = self.controller.joint_limits['lower']
        upper = self.controller.joint_limits['upper']
        
        if not np.all((joints >= lower) & (joints <= upper)):
            return False
        
        # Check if it contains NaN value
        if np.any(np.isnan(joints)):
            return False
        
        return True
    
    
    def smooth_path(self, path):
        """Smooth paths and remove unnecessary oscillations"""
        if len(path) < 3:
            return path
            
        smoothed = [path[0]]  
        
        # Use moving average for smoothing
        window_size = 5
        for i in range(1, len(path)-1):
            start_idx = max(0, i - window_size//2)
            end_idx = min(len(path), i + window_size//2 + 1)
            window = path[start_idx:end_idx]
            avg_point = np.mean(window, axis=0)
            
            # Make sure the smoothed points are still within the work space
            if self.is_target_reachable(avg_point):
                smoothed.append(avg_point)
        
        smoothed.append(path[-1])  
        return np.array(smoothed)
    
    def execute_path(self, path):
        """path execution"""
        if len(path) > 50:
            indices = np.linspace(0, len(path)-1, 50).astype(int)
            path = path[indices]
        
        current_joints = self.data.qpos[:7].copy()
        last_successful_joints = current_joints.copy()
        
        for target_pos in path:
            if not self.is_target_reachable(target_pos):
                self.get_logger().warn(f'Target point exceeds workspace: {target_pos}')
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
                            # If the interpolation point is invalid, fall back to the last successful configuration
                            self.data.qpos[:7] = last_successful_joints
                            break
                    
                    current_joints = new_joints
                except Exception as e:
                    self.get_logger().warn(f'Path execution error: {str(e)}')
                    # Roll back to the last successful configuration
                    self.data.qpos[:7] = last_successful_joints
            else:
                self.get_logger().warn(f'IK solve failed, target position: {target_pos}')

    # Use examples in your code
    def trajectory_callback(self, msg: DisplayTrajectory):
        self.get_logger().info('Start planning and executing trajectories')
        
        try:
            # Set start and end points
            start_pos = np.array([0.1, 0.4, 0.4])
            goal_pos = np.array([0.6, 0.4, 0.4])
            
            # Modify obstacle configuration
            self.planner.obstacles = [
                (np.array([0.4, 0.15, 0.4]), 0.08),  # Place obstacles in the middle of the path
            ]
            
            # Adjust APF parameters to enhance obstacle avoidance effect
            self.planner.k_att = 1.0    
            self.planner.k_rep = 5.0    
            self.planner.rho_0 = 0.3    
            self.planner.step_size = 0.01 
            
            
            
            #Plan path
            path = self.planner.plan_trajectory(start_pos, goal_pos)
            self.current_path = path  # Save path for drawing
            smoothed_path = self.smooth_path(path)
            
            # Generate visualization (this time including path)
            self.plot_potential_field(self.planner, start_pos, goal_pos, smoothed_path)
            self.plot_3d_potential(self.planner, start_pos, goal_pos, smoothed_path)
            
            #Execution path
            self.execute_path(smoothed_path)
            
        except Exception as e:
            self.get_logger().error(f'Execution failed: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    def plot_potential_field(self, planner, start_pos, goal_pos, path):
        
        # Reduce sampling density to make arrow distribution clearer
        x = np.linspace(-0.2, 1.0, 20) 
        y = np.linspace(-0.6, 0.8, 20)
        X, Y = np.meshgrid(x, y)
        
        # Store potential energy and force fields
        potential = np.zeros_like(X)
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        # Calculate potential field
        for i in range(len(x)):
            for j in range(len(y)):
                pos = np.array([X[j,i], Y[j,i], start_pos[2]])
                
                # Calculate force using planner method
                f_att = planner._attractive_force(pos, goal_pos)
                f_rep = planner._repulsive_force(pos)
                total_force = f_att + f_rep
                
                # Store the normalized force field direction
                magnitude = np.linalg.norm(total_force[:2])
                if magnitude > 1e-6:
                    U[j,i] = total_force[0] / magnitude
                    V[j,i] = total_force[1] / magnitude
                
                
                dist_to_goal = np.linalg.norm(pos - goal_pos)
                att_potential = 0.5 * planner.k_att * dist_to_goal**2
                
                rep_potential = 0
                for obs_pos, obs_radius in planner.obstacles:
                    dist_to_obs = np.linalg.norm(pos - obs_pos) - obs_radius
                    if dist_to_obs < planner.rho_0:
                        rep_potential += 0.5 * planner.k_rep * (1.0/dist_to_obs - 1.0/planner.rho_0)**2
                
                potential[j,i] = att_potential + rep_potential
        
        
        fig = plt.figure(figsize=(20, 8))
        
        ax1 = fig.add_subplot(121)
        contour = ax1.contour(X, Y, potential, levels=20, colors='black', alpha=0.4)
        contourf = ax1.contourf(X, Y, potential, levels=20, cmap='viridis', alpha=0.6)
        plt.colorbar(contourf, ax=ax1, label='Potential Value')
        
        ax1.quiver(X, Y, U, V, scale=30, alpha=0.5, color='white', width=0.003)
        
        ax2 = fig.add_subplot(122)
        
        magnitude = np.sqrt(U**2 + V**2)
        ax2.quiver(X, Y, U, V, magnitude, scale=30, cmap='coolwarm', 
                alpha=0.7, width=0.003)
        plt.colorbar(ax2.collections[0], ax=ax2, label='Force Magnitude')
        
        
        for ax in [ax1, ax2]:
            if path is not None:
                ax.plot(path[:,0], path[:,1], 'r--', linewidth=2, label='Planned Path')
                
            ax.plot(start_pos[0], start_pos[1], 'go', markersize=12, label='Start')
            ax.plot(goal_pos[0], goal_pos[1], 'ro', markersize=12, label='Goal')
            
            
            for obs_pos, obs_radius in planner.obstacles:
                circle = plt.Circle((obs_pos[0], obs_pos[1]), obs_radius, 
                                color='red', alpha=0.2)
                ax.add_artist(circle)
            
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)
            ax.legend(fontsize=10)
            ax.axis('equal')
            ax.set_xlim([-0.2, 1.0])
            ax.set_ylim([-0.6, 0.8])
        
        ax1.set_title('Potential Field with Force Directions', fontsize=12)
        ax2.set_title('Force Field Directions', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('potential_field_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_3d_potential(self, planner, start_pos, goal_pos, path):
        x = np.linspace(0.1, 0.7, 50)
        y = np.linspace(-0.3, 0.5, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(len(x)):
            for j in range(len(y)):
                pos = np.array([X[j,i], Y[j,i], start_pos[2]])
                
                dist_to_goal = np.linalg.norm(pos - goal_pos)
                att_potential = 0.5 * planner.k_att * dist_to_goal**2
                
                dist_to_start = np.linalg.norm(pos - start_pos)
                start_rep_potential = (0.5 * planner.k_rep * (1/dist_to_start - 1/planner.rho_0)**2 
                                    if dist_to_start < planner.rho_0 else 0)
                
                obs_rep_potential = 0
                for obs_pos, obs_radius in planner.obstacles:
                    dist_to_obs = np.linalg.norm(pos - obs_pos) - obs_radius
                    if dist_to_obs < planner.rho_0:
                        obs_rep_potential += 0.5 * planner.k_rep * (1/dist_to_obs - 1/planner.rho_0)**2
                
                Z[j,i] = att_potential + start_rep_potential + obs_rep_potential

        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8,
                            linewidth=0, antialiased=True)
        
       
        offset = np.min(Z)
        ax.contour(X, Y, Z, zdir='z', offset=offset, levels=20, cmap='viridis')
        

        if path is not None:
           
            path_Z = np.zeros(len(path))
            for i, pos in enumerate(path):
                dist_to_goal = np.linalg.norm(pos - goal_pos)
                path_Z[i] = 0.5 * planner.k_att * dist_to_goal**2  
            
            
            ax.plot3D(path[:,0], path[:,1], path_Z, 'r--', linewidth=2, label='Planned Path')
        
       
        ax.scatter(start_pos[0], start_pos[1], np.max(Z), color='g', s=100, label='Start')
        ax.scatter(goal_pos[0], goal_pos[1], np.min(Z), color='r', s=100, label='Goal')
        
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Potential')
        ax.set_title('3D Potential Field Surface with Path')
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Potential Value')
        
        ax.view_init(elev=35, azim=45)
        
        plt.savefig('potential_field_3d.png', dpi=300, bbox_inches='tight')
        plt.close()


def main(args=None):
    rclpy.init(args=args)
    bridge = None
    
    try:
        bridge = MujocoROS2Bridge()
        bridge.initialize_viewer()
        bridge.get_logger().info('Start running...')
        
        time.sleep(1)
        bridge.trajectory_callback(None)
        
        rclpy.spin(bridge)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        if bridge and hasattr(bridge, 'viewer') and bridge.viewer is not None:
            bridge.viewer.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
