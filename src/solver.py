import numpy as np
from tqdm import tqdm
import vtk
import os

class FlowSolver:
    """Handles the numerical solution of the Navier-Stokes equations."""
    
    def __init__(self, config, grid, flow_field):
        self.config = config
        self.grid = grid
        self.flow_field = flow_field
        self._check_stability()
        self.vtk_output_dir = "vtk_output"
        self.vtk_file_counter = 0
        os.makedirs(self.vtk_output_dir, exist_ok=True)
        
    def _check_stability(self):
        """Ensure time step satisfies stability condition."""
        max_time_step = 0.5 * self.grid.cell_size**2 / self.config.kinematic_viscosity
        if self.config.time_step > self.config.stability_safety_factor * max_time_step:
            raise RuntimeError("Time step too large for stability.")
    
    def _compute_derivative(self, field, axis, cell_size):
        """Compute derivative of a field along specified axis."""
        if axis == 'x':
            result = np.zeros((field.shape[0], field.shape[1] - 1))
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    # Compute forward difference for x-derivative
                    result[i, j] = (field[i, j + 1] - field[i, j]) / cell_size
        else:
            result = np.zeros((field.shape[0] - 1, field.shape[1]))
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    # Compute forward difference for y-derivative
                    result[i, j] = (field[i + 1, j] - field[i, j]) / cell_size
        return result
    
    def _compute_pressure_gradient(self, pressure, axis, cell_size):
        """Compute pressure gradient on velocity grids."""
        if axis == 'x':
            result = np.zeros((pressure.shape[0], pressure.shape[1] + 1))
            for i in range(pressure.shape[0]):
                for j in range(1, pressure.shape[1]):
                    # Compute forward difference for x-gradient
                    result[i, j] = (pressure[i, j] - pressure[i, j - 1]) / cell_size
            result[:, 0] = result[:, 1] # Left: Neumann
            result[:, -1] = result[:, -2] # Right: Neumann
        else: 
            result = np.zeros((pressure.shape[0] + 1, pressure.shape[1]))
            for i in range(1, pressure.shape[0]):
                for j in range(pressure.shape[1]):
                    # Compute forward difference for y-gradient
                    result[i, j] = (pressure[i, j] - pressure[i - 1, j]) / cell_size
            result[0, :] = result[1, :] # Bottom: Neumann
            result[-1, :] = result[-2, :] # Top: Neumann
        return result
    
    def compute_intermediate_velocity(self):
        """Compute intermediate velocities for u and v components using the momentum equations."""
        # TODO: add equations

        # Copy current velocity fields to preserve original values
        u_intermediate = self.flow_field.u.copy()
        v_intermediate = self.flow_field.v.copy()
        
        # Interpolate u-velocity to pressure grid points (average of adjacent u-values)
        u_on_pressure_grid = 0.5 * (self.flow_field.u[:, :-1] + self.flow_field.u[:, 1:])
        
        # Interpolate v-velocity to pressure grid points (average of adjacent v-values)
        v_on_pressure_grid = 0.5 * (self.flow_field.v[:-1, :] + self.flow_field.v[1:, :])
        
        # Interpolate u-velocity to v-grid points for v-advection term
        u_at_v_points = np.zeros_like(self.flow_field.v)
        for i in range(u_at_v_points.shape[0]):
            for j in range(1, u_at_v_points.shape[1]):
                # We are at v-grid point (i, j)
                # Average u-values from four surrounding points
                # To estimate u-velocity at a v-grid point
                u_at_v_points[i, j] = 0.25 * (
                    self.flow_field.u[max(0, i - 1), j - 1] + 
                    self.flow_field.u[max(0, i - 1), j] +    
                    self.flow_field.u[min(u_intermediate.shape[0] - 1, i), j - 1] + 
                    self.flow_field.u[min(u_intermediate.shape[0] - 1, i), j]       
                )
        
        # Interpolate v-velocity to u-grid points for u-advection term
        v_at_u_points = np.zeros_like(self.flow_field.u)
        for i in range(1, v_at_u_points.shape[0]):
            for j in range(v_at_u_points.shape[1]):
                # We are at u-grid point (i, j)
                # Average v-values from four surrounding points
                # To estimate v-velocity at a u-grid point
                v_at_u_points[i, j] = 0.25 * (
                    self.flow_field.v[i - 1, max(0, j - 1)] +  
                    self.flow_field.v[i - 1, min(v_intermediate.shape[1] - 1, j)] +  
                    self.flow_field.v[i, max(0, j - 1)] +   
                    self.flow_field.v[i, min(v_intermediate.shape[1] - 1, j)]     
                )
        
        # Update u-velocity at interior points (no boundary points)
        for i in range(1, u_intermediate.shape[0] - 1):
            for j in range(1, u_intermediate.shape[1] - 1):
                # Advection in x-direction: u * (du/dx)
                advection_x = (u_on_pressure_grid[i, j] * 
                            (self.flow_field.u[i, j + 1] - self.flow_field.u[i, j - 1]) / 
                            (2 * self.grid.cell_size) if j > 0 and j < u_intermediate.shape[1] - 2 else 0)
                
                # Advection in y-direction: v * (du/dy)
                advection_y = (v_at_u_points[i, j] * 
                            (self.flow_field.u[i + 1, j] - self.flow_field.u[i - 1, j]) / 
                            (2 * self.grid.cell_size))
                
                # Diffusion term: viscosity * Laplacian(u)
                diffusion = (self.flow_field.u[i + 1, j] + self.flow_field.u[i - 1, j] + 
                            self.flow_field.u[i, j + 1] + self.flow_field.u[i, j - 1] - 
                            4 * self.flow_field.u[i, j]) / (self.grid.cell_size**2)
                
                # Update u-velocity using momentum equation
                u_intermediate[i, j] = (self.flow_field.u[i, j] + self.config.time_step * 
                                        (-advection_x - advection_y + self.config.kinematic_viscosity * diffusion))
        
        # Update v-velocity at interior points (no boundary points)
        for i in range(1, v_intermediate.shape[0] - 1):
            for j in range(1, v_intermediate.shape[1] - 1):
                # Advection in x-direction: u * (dv/dx)
                advection_x = (u_at_v_points[i, j] * 
                            (self.flow_field.v[i, j + 1] - self.flow_field.v[i, j - 1]) / 
                            (2 * self.grid.cell_size))
                
                # Advection in y-direction: v * (dv/dy)
                advection_y = (v_on_pressure_grid[i, j] * 
                            (self.flow_field.v[i + 1, j] - self.flow_field.v[i - 1, j]) / 
                            (2 * self.grid.cell_size) if i > 0 and i < v_intermediate.shape[0] - 2 else 0)
                
                # Diffusion term: viscosity * Laplacian(v)
                diffusion = (self.flow_field.v[i + 1, j] + self.flow_field.v[i - 1, j] + 
                            self.flow_field.v[i, j + 1] + self.flow_field.v[i, j - 1] - 
                            4 * self.flow_field.v[i, j]) / (self.grid.cell_size**2)
                
                # Update v-velocity using momentum equation
                v_intermediate[i, j] = (self.flow_field.v[i, j] + self.config.time_step * 
                                        (-advection_x - advection_y + self.config.kinematic_viscosity * diffusion))
        
        return u_intermediate, v_intermediate
    
    def solve_pressure_poisson(self, u_temp, v_temp):
        """Solve the pressure Poisson equation to ensure incompressibility."""
        # du/dx + dv/dy = 0
        # du/dt + u * du/dx + v * du/dy = -1/ρ * dp/dx + ν * (d²u/dx² + d²u/dy²)
        # dv/dt + u * dv/dx + v * dv/dy = -1/ρ * dp/dy + ν * (d²v/dx² + d²v/dy²)

        # Pressure Poisson Equation
        # d²p/dx² + d²p/dy² = ρ / Δt * (du_temp/dx + dv_temp/dy)
        # (p[i,j-1] + p[i-1,j] + p[i,j+1] + p[i+1,j] - 4*p[i,j]) / Δx² = rhs[i,j]
        # p[i,j] = (p[i,j-1] + p[i-1,j] + p[i,j+1] + p[i+1,j] - Δx² * rhs[i,j]) / 4
        du_dx = self._compute_derivative(u_temp, 'x', self.grid.cell_size)
        dv_dy = self._compute_derivative(v_temp, 'y', self.grid.cell_size)
        rhs = self.config.density / self.config.time_step * (du_dx + dv_dy)
        
        p_next = self.flow_field.p.copy()
        for _ in range(self.config.pressure_iterations):
            p_iter = np.zeros_like(p_next)
            for i in range(1, p_next.shape[0] - 1):
                for j in range(1, p_next.shape[1] - 1):
                    # Update pressure using finite difference method
                    p_iter[i, j] = 0.25 * (
                        p_next[i, j - 1] + p_next[i - 1, j] +
                        p_next[i, j + 1] + p_next[i + 1, j] -
                        self.grid.cell_size**2 * rhs[i, j]
                    )
            p_iter[0, :] = p_iter[1, :]     # Bottom: Neumann (dp/dy = 0)
            p_iter[-1, :] = 0               # Top: Dirichlet
            p_iter[:, 0] = p_iter[:, 1]     # Left: Neumann
            p_iter[:, -1] = p_iter[:, -2]   # Right: Neumann
            p_next = p_iter
        
        dp_dx = self._compute_pressure_gradient(p_next, 'x', self.grid.cell_size)
        dp_dy = self._compute_pressure_gradient(p_next, 'y', self.grid.cell_size)
        
        return dp_dx, dp_dy, p_next
    
    def _write_vtk(self, iteration, current_time):
        """Write flow field data to a VTK file."""
        # Interpolate velocities to pressure grid (cell-centered)
        u_vis = 0.5 * (self.flow_field.u[:, :-1] + self.flow_field.u[:, 1:])
        v_vis = 0.5 * (self.flow_field.v[:-1, :] + self.flow_field.v[1:, :])
        p_vis = self.flow_field.p
        
        # Create a VTK rectilinear grid
        grid = vtk.vtkRectilinearGrid()
        grid.SetDimensions(self.grid.grid_size - 1, self.grid.grid_size - 1, 1)
        
        # Set X and Y coordinates (cell-centered pressure grid)
        x_coords = vtk.vtkFloatArray()
        y_coords = vtk.vtkFloatArray()
        z_coords = vtk.vtkFloatArray()
        
        x_points = np.linspace(self.grid.cell_size / 2, self.grid.domain_size - self.grid.cell_size / 2, self.grid.grid_size - 1)
        y_points = np.linspace(self.grid.cell_size / 2, self.grid.domain_size - self.grid.cell_size / 2, self.grid.grid_size - 1)
        
        for x in x_points:
            x_coords.InsertNextValue(x)
        for y in y_points:
            y_coords.InsertNextValue(y)
        z_coords.InsertNextValue(0.0)  # 2D simulation, single z-plane
        
        grid.SetXCoordinates(x_coords)
        grid.SetYCoordinates(y_coords)
        grid.SetZCoordinates(z_coords)
        
        # Add pressure as cell data
        pressure_array = vtk.vtkFloatArray()
        pressure_array.SetName("Pressure")
        for i in range(self.grid.grid_size - 1):
            for j in range(self.grid.grid_size - 1):
                pressure_array.InsertNextValue(p_vis[i, j])
        grid.GetCellData().AddArray(pressure_array)
        
        # Add velocity as point data
        velocity_array = vtk.vtkFloatArray()
        velocity_array.SetName("Velocity")
        velocity_array.SetNumberOfComponents(3)  # 3D vector, z-component is 0
        for i in range(self.grid.grid_size - 1):
            for j in range(self.grid.grid_size - 1):
                velocity_array.InsertNextTuple3(u_vis[i, j], v_vis[i, j], 0.0)
        grid.GetPointData().AddArray(velocity_array)
        
        # Write to file
        writer = vtk.vtkRectilinearGridWriter()
        writer.SetFileName(os.path.join(self.vtk_output_dir, f"flow_{self.vtk_file_counter:06d}.vtk"))
        writer.SetInputData(grid)
        writer.Write()
        print(f"Saved VTK file: flow_{self.vtk_file_counter:06d}.vtk at time {current_time:.2f}")
        self.vtk_file_counter += 1

    def run(self, iterations):
        """Run the simulation for specified number of iterations."""
        output_interval = int(0.1 / self.config.time_step) # or 1
        current_time = 0.0
        
        for i in tqdm(range(iterations), desc="Simulation Progress"):
            u_temp, v_temp = self.compute_intermediate_velocity()
            self.flow_field.apply_boundary_conditions()
            dp_dx, dp_dy, p_next = self.solve_pressure_poisson(u_temp, v_temp)
            self.flow_field.u = u_temp - self.config.time_step / self.config.density * dp_dx
            self.flow_field.v = v_temp - self.config.time_step / self.config.density * dp_dy
            self.flow_field.apply_boundary_conditions()
            self.flow_field.p = p_next
            
            current_time += self.config.time_step
            if (i + 1) % output_interval == 0:
                self._write_vtk(i + 1, current_time)