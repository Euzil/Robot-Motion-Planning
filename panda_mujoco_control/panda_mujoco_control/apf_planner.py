import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

class ImprovedAPFPlanner:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        #APF parameter
        self.k_att = 1.0    # Attraction coefficient
        self.k_rep = 0.3    # Repulsion coefficient
        self.rho_0 = 0.3    # Repulsive force range of influence
        self.step_size = 0.01  # step
        self.max_iterations = 1000
        self.goal_threshold = 0.01
        
        # Potential field sampling point grid
        self.grid_size = 20  # Number of sampling points in each dimension
        self.workspace_bounds = {
            'x': (0.1, 0.7), 
            'y': (-0.3, 0.3),
            'z': (0.1, 0.7)
        }
        
        # Obstacle configuration
        self.obstacles = [
            (np.array([0.4, 0.4, 0.4]), 0.08),
        ]
        
        # Initialize visual elements
        self._init_visualization()
    
    def _init_visualization(self):
        # Create potential field arrow mark points
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
        """Calculate the force field for the entire workspace"""
        for arrow in self.force_arrows:
            pos = arrow['pos']
            f_att = self._attractive_force(pos, goal_pos)
            f_rep = self._repulsive_force(pos)
            total_force = f_att + f_rep
            # Normalized direction vector
            magnitude = np.linalg.norm(total_force)
            if magnitude > 1e-6:
                direction = total_force / magnitude
            else:
                direction = np.zeros(3)
            
            arrow['dir'] = direction
            arrow['magnitude'] = min(magnitude, 0.1)
    
    def _attractive_force(self, position, goal):
        """Calculate attract (using quadratic potential field)"""
        diff = goal - position
        distance = np.linalg.norm(diff)
        
        if distance <= self.rho_0:
            return self.k_att * diff
        else:
            return self.k_att * self.rho_0 * diff / distance
    
    def _repulsive_force(self, position):
        """Calculate repulsion (taking into account obstacle size)"""
        total_force = np.zeros(3)
        
        for obs_pos, obs_radius in self.obstacles:
            diff = position - obs_pos
            distance = np.linalg.norm(diff)
            
            actual_distance = max(distance - obs_radius, 0.001)
            
            if actual_distance < self.rho_0:
                direction = diff / distance
                magnitude = self.k_rep * (1.0/actual_distance - 1.0/self.rho_0) * (1.0/(actual_distance**2))
                total_force += magnitude * direction
        
        return total_force
    
    def render_force_field(self, viewer):
        # Render potential field arrows
        for arrow in self.force_arrows:
            start_pos = arrow['pos']
            direction = arrow['dir']
            magnitude = arrow['magnitude']
            
            if magnitude > 1e-6:
                #Start point of arrow
                viewer.add_marker(
                    pos=start_pos,
                    size=np.array([0.005, 0.005, 0.005]),
                    rgba=np.array([0.2, 0.2, 1.0, 0.5]),
                    type=mujoco.mjtGeom.mjGEOM_SPHERE
                )
                
                # Arrow segment
                end_pos = start_pos + direction * magnitude * 0.1
                viewer.add_marker(
                    pos=start_pos,
                    mat=R.from_rotvec(direction).as_matrix(),
                    size=np.array([magnitude * 0.1, 0.002, 0.002]),
                    rgba=np.array([0.2, 0.2, 1.0, 0.5]),
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER
                )
        
        # Render obstacles
        for obs_pos, obs_radius in self.obstacles:
            viewer.add_marker(
                pos=obs_pos,
                size=np.array([obs_radius, obs_radius, obs_radius]),
                rgba=np.array([1.0, 0.0, 0.0, 0.5]),
                type=mujoco.mjtGeom.mjGEOM_SPHERE
            )
    
    def plan_trajectory(self, start_pos, goal_pos):
        """Generate trajectories based on potential fields"""
        path = [start_pos]
        current_pos = start_pos.copy()
        
        # Calculate the force field for the entire workspace
        self._compute_force_field(goal_pos)
        
        for i in range(self.max_iterations):
            # Calculate the resultant force at the current position
            f_att = self._attractive_force(current_pos, goal_pos)
            f_rep = self._repulsive_force(current_pos)
            total_force = f_att + f_rep
            
            # Update position using gradient descent
            force_magnitude = np.linalg.norm(total_force)
            if force_magnitude > 1e-6:
                movement = self.step_size * (total_force / force_magnitude)
                current_pos += movement
            
            # Record path
            if i % 5 == 0: 
                path.append(current_pos.copy())
            
            # Check if the goal is reached
            if np.linalg.norm(current_pos - goal_pos) < self.goal_threshold:
                break
        
        path.append(goal_pos)
        return np.array(path)
