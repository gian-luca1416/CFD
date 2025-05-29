import numpy as np
import os
from helpers import FlowVisualizer, BenchmarkValidator
from solver import FlowSolver

class CavityFlowConfig:
    """Configuration parameters for the lid-driven cavity simulation."""
    
    def __init__(self):
        self.grid_size = 100  # imax
        self.grid_size_y = 20  # jmax
        self.domain_size = 10.0  # xlength
        self.domain_size_y = 2.0  # ylength
        self.iterations = int(30 / 0.0025)  # t_end / dt
        self.time_step = 0.0025  # dt
        self.pressure_iterations = 50  # itermax
        self.stability_safety_factor = 0.5  # tau
        self.resume_calculation = False
        self.kinematic_viscosity = 0.1  # 1/Re, Re=10
        self.density = 1.0
        self.top_wall_velocity = 0.0  # Keine lid-driven Bedingung mehr
        self.delta_p = 4.0  # Druckgradient für Kanalströmung

class StaggeredGrid:
    """Manages the staggered grid setup for velocity and pressure fields."""
    
    def __init__(self, config):
        self.domain_size = config.domain_size
        self.domain_size_y = config.domain_size_y
        self.grid_size = config.grid_size
        self.grid_size_y = config.grid_size_y
        self.cell_size = self.domain_size / (self.grid_size - 1)
        self.cell_size_y = self.domain_size_y / (self.grid_size_y - 1)
        
        # Initialize grid coordinates
        self._create_grids()
        
    def _create_grids(self):
        """Create staggered grids for pressure and velocity components."""
        half_cell_x = self.cell_size / 2
        half_cell_y = self.cell_size_y / 2
        pressure_points_x = np.linspace(half_cell_x, self.domain_size - half_cell_x, self.grid_size - 1)
        pressure_points_y = np.linspace(half_cell_y, self.domain_size_y - half_cell_y, self.grid_size_y - 1)
        velocity_points_x = np.linspace(0.0, self.domain_size, self.grid_size)
        velocity_points_y = np.linspace(0.0, self.domain_size_y, self.grid_size_y)
        
        # Pressure grid (cell centers)
        self.X_p, self.Y_p = np.meshgrid(pressure_points_x, pressure_points_y)
        
        # u-velocity grid (staggered in x-direction)
        self.X_u, self.Y_u = np.meshgrid(velocity_points_x, pressure_points_y)
        
        # v-velocity grid (staggered in y-direction)
        self.X_v, self.Y_v = np.meshgrid(pressure_points_x, velocity_points_y)

class FlowField:
    """Manages the velocity and pressure fields with boundary conditions."""
    
    def __init__(self, grid, config):
        self.grid = grid
        self.config = config
        self.u = np.ones_like(grid.X_u)  # Initiale Bedingung: u=1
        self.v = np.zeros_like(grid.X_v)  # Initiale Bedingung: v=0
        self.p = np.zeros_like(grid.X_p)  # pressure
        
    def apply_boundary_conditions(self):
        """Apply boundary conditions for channel flow."""
        # u-velocity boundaries
        self.u[0, :] = 0.0  # Bottom (no-slip)
        self.u[-1, :] = 0.0  # Top (no-slip)
        self.u[:, 0] = 1.0  # Left (inflow: u=1)
        self.u[:, -1] = self.u[:, -2]  # Right (outflow: du/dx=0)
        
        # v-velocity boundaries
        self.v[0, :] = 0.0  # Bottom (no-slip)
        self.v[-1, :] = 0.0  # Top (no-slip)
        self.v[:, 0] = 0.0  # Left (inflow: v=0)
        self.v[:, -1] = self.v[:, -2]  # Right (outflow: dv/dx=0)

class SimulationManager:
    """Manages the overall simulation workflow."""
    
    def __init__(self):
        self.config = CavityFlowConfig()
        self.result_file = 'cfd_results.npz'
        
    def save_results(self, grid, flow_field):
        """Save simulation results to file."""
        np.savez(self.result_file, X_p=grid.X_p, Y_p=grid.Y_p,
                 u=flow_field.u, v=flow_field.v, p=flow_field.p)
        print(f"Simulation results saved to {self.result_file}")
    
    def load_results(self):
        """Load simulation results from file."""
        data = np.load(self.result_file)
        return data['X_p'], data['Y_p'], data['u'], data['v'], data['p']
    
    def run(self):
        """Execute the simulation workflow."""
        grid = StaggeredGrid(self.config)
        flow_field = FlowField(grid, self.config)
        
        if os.path.exists(self.result_file) and self.config.resume_calculation:
            print(f"Resuming calculation from {self.result_file}...")
            flow_field.X_p, flow_field.Y_p, flow_field.u, flow_field.v, flow_field.p = self.load_results()
            solver = FlowSolver(self.config, grid, flow_field)
            print(f"Continuing for {self.config.iterations} more iterations...")
            solver.run(self.config.iterations)
            self.save_results(grid, flow_field)
        
        elif os.path.exists(self.result_file):
            print(f"Loading existing results from {self.result_file}")
            flow_field.X_p, flow_field.Y_p, flow_field.u, flow_field.v, flow_field.p = self.load_results()
        
        else:
            print("Running new simulation...")
            flow_field.apply_boundary_conditions()
            solver = FlowSolver(self.config, grid, flow_field)
            solver.run(self.config.iterations)
            self.save_results(grid, flow_field)
        
        print("Visualizing results...")
        visualizer = FlowVisualizer(grid, flow_field)
        visualizer.plot()
        
        print("Validating against benchmark data...")
        validator = BenchmarkValidator(grid, flow_field)
        validator.validate()

if __name__ == "__main__":
    sim = SimulationManager()
    sim.run()