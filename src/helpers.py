import matplotlib.pyplot as plt
import numpy as np

class FlowVisualizer:
    """Handles visualization of the flow field."""
    
    def __init__(self, grid, flow_field):
        self.grid = grid
        self.flow_field = flow_field
    
    def _interpolate_to_visualization_grid(self):
        """Interpolate velocity fields to pressure grid for visualization."""
        u_vis = 0.5 * (self.flow_field.u[:, :-1] + self.flow_field.u[:, 1:])
        v_vis = 0.5 * (self.flow_field.v[:-1, :] + self.flow_field.v[1:, :])
        return self.grid.X_p, self.grid.Y_p, self.flow_field.p, u_vis, v_vis
    
    def plot(self):
        """Create visualization of pressure field with velocity vectors and streamlines."""
        X_vis, Y_vis, p_vis, u_vis, v_vis = self._interpolate_to_visualization_grid()
        
        plt.style.use("dark_background")
        plt.figure(figsize=(8, 8))
        
        plt.contourf(X_vis, Y_vis, p_vis, cmap="coolwarm", alpha=0.8)
        plt.colorbar(label='Pressure')
        
        skip = 3
        plt.quiver(X_vis[::skip, ::skip], Y_vis[::skip, ::skip],
                   u_vis[::skip, ::skip], v_vis[::skip, ::skip],
                   color="white", scale=5)
        
        plt.streamplot(X_vis, Y_vis, u_vis, v_vis, color="cyan",
                       density=1.5, linewidth=0.5)
        
        plt.title('Pressure Field in Lid-Driven Cavity')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('lid_driven_cavity.png')
        plt.close()

class BenchmarkValidator:
    """Validates simulation results against Ghia et al. (1982) benchmark data."""
    
    def __init__(self, grid, flow_field):
        self.grid = grid
        self.flow_field = flow_field
    
    def _load_ghia_data(self):
        """Load benchmark data from files."""
        ghia_y, ghia_u = [], []
        with open('ghia_u.txt', 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                values = line.strip().split()
                if len(values) >= 2:
                    ghia_y.append(float(values[0]))
                    ghia_u.append(float(values[1]))
        
        ghia_x, ghia_v = [], []
        with open('ghia_v.txt', 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                values = line.strip().split()
                if len(values) >= 2:
                    ghia_x.append(float(values[0]))
                    ghia_v.append(float(values[1]))
        
        return ghia_y, ghia_u, ghia_x, ghia_v
    
    def validate(self):
        """Compare simulation results with benchmark data."""
        X_vis, Y_vis, _, u_vis, v_vis = FlowVisualizer(self.grid, self.flow_field)._interpolate_to_visualization_grid()
        ghia_y, ghia_u, ghia_x, ghia_v = self._load_ghia_data()
        
        mid_x_index = np.argmin(np.abs(X_vis[0, :] - 0.5))
        sim_y = Y_vis[:, mid_x_index]
        sim_u = u_vis[:, mid_x_index]
        
        mid_y_index = np.argmin(np.abs(Y_vis[:, 0] - 0.5))
        sim_x = X_vis[mid_y_index, :]
        sim_v = v_vis[mid_y_index, :]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(sim_u, sim_y, 'b-', label='Simulation')
        plt.plot(ghia_u, ghia_y, 'ro', label='Ghia et al. (1982)')
        plt.xlabel('U-velocity')
        plt.ylabel('Y-coordinate')
        plt.title('U-velocity along vertical centerline')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(sim_x, sim_v, 'b-', label='Simulation')
        plt.plot(ghia_x, ghia_v, 'ro', label='Ghia et al. (1982)')
        plt.xlabel('X-coordinate')
        plt.ylabel('V-velocity')
        plt.title('V-velocity along horizontal centerline')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('benchmark_comparison.png')
        plt.close()