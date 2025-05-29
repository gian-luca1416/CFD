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
        
        plt.title('Pressure Field in Channel Flow')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('channel_flow.png')
        plt.close()

class BenchmarkValidator:
    """Validates simulation results against analytical solution for channel flow."""
    
    def __init__(self, grid, flow_field):
        self.grid = grid
        self.flow_field = flow_field
    
    def _compute_analytical_solution(self):
        """Compute analytical solution for channel flow: u(y) = -1/2 * Re * (dp/dx) * y * (y-h)"""
        Re = 1.0 / self.flow_field.config.kinematic_viscosity
        dp_dx = self.flow_field.config.delta_p / self.grid.domain_size  # Korrigierter Druckgradient
        h = self.grid.domain_size_y
        y = self.grid.Y_p[:, 0]  # Y-Werte entlang der vertikalen Achse
        u_analytical = -0.5 * Re * dp_dx * y * (y - h)
        return y, u_analytical
    
    def validate(self):
        """Compare simulation results with analytical solution."""
        visualizer = FlowVisualizer(self.grid, self.flow_field)
        X_vis, Y_vis, _, u_vis, v_vis = visualizer._interpolate_to_visualization_grid()
        y_analytical, u_analytical = self._compute_analytical_solution()
        
        mid_x_index = np.argmin(np.abs(X_vis[0, :] - self.grid.domain_size / 2))
        sim_y = Y_vis[:, mid_x_index]
        sim_u = u_vis[:, mid_x_index]
        
        plt.figure(figsize=(8, 5))
        plt.plot(sim_u, sim_y, 'b-', label='Simulation')
        plt.plot(u_analytical, y_analytical, 'r--', label='Analytical Solution')
        plt.xlabel('U-velocity')
        plt.ylabel('Y-coordinate')
        plt.title('U-velocity along vertical centerline (Channel Flow)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('channel_flow_comparison.png')
        plt.close()