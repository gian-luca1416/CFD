import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# This code is inspired by https://github.com/Ceyron/lid-driven-cavity-python/tree/main

# Meta information
GRID_SIZE = 128
DOMAIN_SIZE = 1.0
ITERATIONS = 2000
DELTA_T = 0.001
PRESSURE_POISSON_ITERATIONS = 50
STABILITY_SAFETY_FACTOR = 0.5
RESUME_CALCULATION = False  # Set to True to resume from existing results

# Information
KINEMATIC_VISCOSITY = 0.01
DENSITY = 1.0

# Boundary conditions
U_TOP = 1.0

def create_staggered_grid():
    """Create staggered grids for pressure, u-velocity, and v-velocity"""
    # Cell centers for pressure
    x_p = np.linspace(0.0 + DOMAIN_SIZE/(2*(GRID_SIZE-1)), DOMAIN_SIZE - DOMAIN_SIZE/(2*(GRID_SIZE-1)), GRID_SIZE-1)
    y_p = np.linspace(0.0 + DOMAIN_SIZE/(2*(GRID_SIZE-1)), DOMAIN_SIZE - DOMAIN_SIZE/(2*(GRID_SIZE-1)), GRID_SIZE-1)
    
    # u-velocity grid (staggered in x-direction)
    x_u = np.linspace(0.0, DOMAIN_SIZE, GRID_SIZE)
    y_u = y_p.copy()
    
    # v-velocity grid (staggered in y-direction)
    x_v = x_p.copy()
    y_v = np.linspace(0.0, DOMAIN_SIZE, GRID_SIZE)
    
    # Create 2D meshgrids
    X_p, Y_p = np.meshgrid(x_p, y_p)
    X_u, Y_u = np.meshgrid(x_u, y_u)
    X_v, Y_v = np.meshgrid(x_v, y_v)
    
    return X_p, Y_p, X_u, Y_u, X_v, Y_v

def differentiate_u_x(u, element_length):
    """Differentiate u with respect to x on the pressure grid"""
    result = np.zeros((u.shape[0], u.shape[1]-1))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = (u[i, j+1] - u[i, j]) / element_length
    return result

def differentiate_v_y(v, element_length):
    """Differentiate v with respect to y on the pressure grid"""
    result = np.zeros((v.shape[0]-1, v.shape[1]))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = (v[i+1, j] - v[i, j]) / element_length
    return result

def differentiate_p_x(p, element_length):
    """Differentiate pressure with respect to x on the u-grid"""
    result = np.zeros((p.shape[0], p.shape[1]+1))
    # Interior points
    for i in range(p.shape[0]):
        for j in range(1, p.shape[1]):
            result[i, j] = (p[i, j] - p[i, j-1]) / element_length
    
    # Boundary points (extrapolation)
    result[:, 0] = result[:, 1]
    result[:, -1] = result[:, -2]
    return result

def differentiate_p_y(p, element_length):
    """Differentiate pressure with respect to y on the v-grid"""
    result = np.zeros((p.shape[0]+1, p.shape[1]))
    # Interior points
    for i in range(1, p.shape[0]):
        for j in range(p.shape[1]):
            result[i, j] = (p[i, j] - p[i-1, j]) / element_length
    
    # Boundary points (extrapolation)
    result[0, :] = result[1, :]
    result[-1, :] = result[-2, :]
    return result

def laplace_u(u, element_length):
    """Calculate Laplacian of u on u-grid"""
    result = np.zeros_like(u)
    # Interior points
    for i in range(1, u.shape[0]-1):
        for j in range(1, u.shape[1]-1):
            result[i, j] = (
                u[i, j-1] + u[i-1, j] - 4*u[i, j] + u[i, j+1] + u[i+1, j]
            ) / (element_length**2)
    return result

def laplace_v(v, element_length):
    """Calculate Laplacian of v on v-grid"""
    result = np.zeros_like(v)
    # Interior points
    for i in range(1, v.shape[0]-1):
        for j in range(1, v.shape[1]-1):
            result[i, j] = (
                v[i, j-1] + v[i-1, j] - 4*v[i, j] + v[i, j+1] + v[i+1, j]
            ) / (element_length**2)
    return result

def interpolate_u_to_v(u, v):
    """Interpolate u-velocity to v-points for advection terms"""
    u_at_v = np.zeros_like(v)
    for i in range(v.shape[0]):
        for j in range(1, v.shape[1]-1):
            u_at_v[i, j] = 0.25 * (u[i, j-1] + u[i, j] + u[max(0, i-1), j-1] + u[max(0, i-1), j])
    return u_at_v

def interpolate_v_to_u(v, u):
    """Interpolate v-velocity to u-points for advection terms"""
    v_at_u = np.zeros_like(u)
    for i in range(1, u.shape[0]-1):
        for j in range(u.shape[1]):
            v_at_u[i, j] = 0.25 * (v[i-1, j] + v[i, j] + v[i-1, max(0, j-1)] + v[i, max(0, j-1)])
    return v_at_u

def ensure_stability(element_length):
    maximum_possible_time_step_length = (0.5 * element_length**2 / KINEMATIC_VISCOSITY)
    if DELTA_T > (STABILITY_SAFETY_FACTOR * maximum_possible_time_step_length):
        raise RuntimeError("Stability is not guaranteed.")

def solve_momentum(u, v, element_length):
    """
    Calculate the temporary velocities using momentum equations on a staggered grid.
    """
    # Get array shapes
    ny_u, nx_u = u.shape
    ny_v, nx_v = v.shape
    
    # Create temporary arrays for the next time step
    u_temp = u.copy()
    v_temp = v.copy()
    
    # Interpolate u to pressure points (for u-advection of u)
    u_on_p = np.zeros((ny_u, nx_u - 1))
    for j in range(nx_u - 1):
        u_on_p[:, j] = 0.5 * (u[:, j] + u[:, j+1])
    
    # Interpolate v to pressure points (for v-advection of v)
    v_on_p = np.zeros((ny_v - 1, nx_v))
    for i in range(ny_v - 1):
        v_on_p[i, :] = 0.5 * (v[i, :] + v[i+1, :])
    
    # Interpolate u to v-points (for u-advection of v)
    u_at_v = np.zeros_like(v)
    for i in range(ny_v):
        for j in range(1, nx_v):
            u_at_v[i, j] = 0.25 * (u[max(0, i-1), j-1] + u[max(0, i-1), j] + 
                                   u[min(ny_u-1, i), j-1] + u[min(ny_u-1, i), j])
    
    # Interpolate v to u-points (for v-advection of u)
    v_at_u = np.zeros_like(u)
    for i in range(1, ny_u):
        for j in range(nx_u):
            v_at_u[i, j] = 0.25 * (v[i-1, max(0, j-1)] + v[i-1, min(nx_v-1, j)] + 
                                   v[i, max(0, j-1)] + v[i, min(nx_v-1, j)])
    
    # Calculate advection terms for u (internal points only)
    for i in range(1, ny_u - 1):
        for j in range(1, nx_u - 1):
            # Advection in x-direction
            if j > 0 and j < nx_u - 2:
                adv_u_x = u[i, j] * (u[i, j+1] - u[i, j-1]) / (2 * element_length)
            else:
                adv_u_x = 0
            
            # Advection in y-direction
            adv_u_y = v_at_u[i, j] * (u[i+1, j] - u[i-1, j]) / (2 * element_length)
            
            # Diffusion term
            diff_u = (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4*u[i, j]) / (element_length**2)
            
            # Update u
            u_temp[i, j] = u[i, j] + DELTA_T * (
                -adv_u_x - adv_u_y + KINEMATIC_VISCOSITY * diff_u
            )
    
    # Calculate advection terms for v (internal points only)
    for i in range(1, ny_v - 1):
        for j in range(1, nx_v - 1):
            # Advection in x-direction
            adv_v_x = u_at_v[i, j] * (v[i, j+1] - v[i, j-1]) / (2 * element_length)
            
            # Advection in y-direction
            if i > 0 and i < ny_v - 2:
                adv_v_y = v[i, j] * (v[i+1, j] - v[i-1, j]) / (2 * element_length)
            else:
                adv_v_y = 0
                
            # Diffusion term
            diff_v = (v[i+1, j] + v[i-1, j] + v[i, j+1] + v[i, j-1] - 4*v[i, j]) / (element_length**2)
            
            # Update v
            v_temp[i, j] = v[i, j] + DELTA_T * (
                -adv_v_x - adv_v_y + KINEMATIC_VISCOSITY * diff_v
            )
    
    return u_temp, v_temp

def solve_pressure(u_temp, v_temp, p, element_length):
    # Calculate divergence on pressure grid
    du_dx = differentiate_u_x(u_temp, element_length)
    dv_dy = differentiate_v_y(v_temp, element_length)
    
    # RHS of pressure Poisson equation
    rhs = DENSITY / DELTA_T * (du_dx + dv_dy)
    
    # Solve pressure Poisson equation
    p_next = p.copy()
    for _ in range(PRESSURE_POISSON_ITERATIONS):
        p_next_iter = np.zeros_like(p_next)
        
        # Interior points
        for i in range(1, p_next.shape[0]-1):
            for j in range(1, p_next.shape[1]-1):
                p_next_iter[i, j] = 0.25 * (
                    p_next[i, j-1] + p_next[i-1, j] + 
                    p_next[i, j+1] + p_next[i+1, j] - 
                    element_length**2 * rhs[i, j]
                )
        
        # Boundary conditions for pressure
        p_next_iter[0, :] = p_next_iter[1, :]      # Top: Neumann
        p_next_iter[-1, :] = 0                     # Bottom: Dirichlet
        p_next_iter[:, 0] = p_next_iter[:, 1]      # Left: Neumann
        p_next_iter[:, -1] = p_next_iter[:, -2]    # Right: Neumann
        
        p_next = p_next_iter
    
    # Calculate pressure gradients
    dp_dx = differentiate_p_x(p_next, element_length)
    dp_dy = differentiate_p_y(p_next, element_length)
    
    return dp_dx, dp_dy, p_next

def enforce_boundary_conditions(u, v):
    # u velocity boundary conditions
    u[0, :] = 0.0                # Bottom boundary (no slip)
    u[-1, :] = 2*U_TOP - u[-2, :] # Top boundary (lid-driven with extrapolation)
    u[:, 0] = 0.0                # Left boundary (no slip)
    u[:, -1] = 0.0               # Right boundary (no slip)
    
    # v velocity boundary conditions
    v[0, :] = 0.0                # Bottom boundary (no slip)
    v[-1, :] = 0.0               # Top boundary (no slip)
    v[:, 0] = 0.0                # Left boundary (no slip)
    v[:, -1] = 0.0               # Right boundary (no slip)
    
    return u, v

def interpolate_to_visualization_grid(X_p, Y_p, p, u, v):
    """Interpolate staggered grid values to a common visualization grid"""
    X_vis, Y_vis = X_p, Y_p
    
    # Interpolate u to pressure points
    u_vis = np.zeros_like(X_vis)
    for i in range(X_vis.shape[0]):
        for j in range(X_vis.shape[1]):
            u_vis[i, j] = 0.5 * (u[i, j] + u[i, j+1])
    
    # Interpolate v to pressure points
    v_vis = np.zeros_like(Y_vis)
    for i in range(Y_vis.shape[0]):
        for j in range(Y_vis.shape[1]):
            v_vis[i, j] = 0.5 * (v[i, j] + v[i+1, j])
    
    return X_vis, Y_vis, p, u_vis, v_vis

def visualization(X_p, Y_p, p, u, v):
    # Interpolate to visualization grid
    X_vis, Y_vis, p_vis, u_vis, v_vis = interpolate_to_visualization_grid(X_p, Y_p, p, u, v)
    
    plt.style.use("dark_background")
    plt.figure(figsize=(8, 8))
    
    # Plot velocity vectors and streamlines
    speed = np.sqrt(u_vis**2 + v_vis**2)
    plt.contourf(X_vis, Y_vis, speed, cmap="coolwarm", alpha=0.8)
    plt.colorbar(label='Velocity Magnitude')
    
    # Add velocity vectors (at reduced resolution for clarity)
    skip = 3
    plt.quiver(X_vis[::skip, ::skip], Y_vis[::skip, ::skip], 
               u_vis[::skip, ::skip], v_vis[::skip, ::skip], 
               color="white", scale=5)
    
    # Add streamlines
    plt.streamplot(X_vis, Y_vis, u_vis, v_vis, color="cyan", 
                  density=1.5, linewidth=0.5)
    
    plt.title('Lid-Driven Cavity Flow')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def validate_with_ghia(X_p, Y_p, p, u, v):
    """Compare simulation results with Ghia et al. (1982) benchmark data"""
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Parse Ghia's data for Re=100 (column index 1)
    ghia_y = []
    ghia_u = []
    with open('ghia_u.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            values = line.strip().split()
            if len(values) >= 2:
                ghia_y.append(float(values[0]))
                ghia_u.append(float(values[1]))  # Re = 100
    
    ghia_x = []
    ghia_v = []
    with open('ghia_v.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            values = line.strip().split()
            if len(values) >= 2:
                ghia_x.append(float(values[0]))
                ghia_v.append(float(values[1]))  # Re = 100
    
    # Interpolate to visualization grid
    X_vis, Y_vis, _, u_vis, v_vis = interpolate_to_visualization_grid(X_p, Y_p, p, u, v)
    
    # Extract centerline velocities
    # U velocity along vertical centerline (x = 0.5)
    mid_x_index = np.argmin(np.abs(X_vis[0, :] - 0.5))
    sim_y = Y_vis[:, mid_x_index]
    sim_u = u_vis[:, mid_x_index]
    
    # V velocity along horizontal centerline (y = 0.5)
    mid_y_index = np.argmin(np.abs(Y_vis[:, 0] - 0.5))
    sim_x = X_vis[mid_y_index, :]
    sim_v = v_vis[mid_y_index, :]
    
    # Plot comparisons
    plt.figure(figsize=(12, 5))
    
    # U velocity along vertical centerline
    plt.subplot(1, 2, 1)
    plt.plot(sim_u, sim_y, 'b-', label='Simulation')
    plt.plot(ghia_u, ghia_y, 'ro', label='Ghia et al. (1982)')
    plt.xlabel('U-velocity')
    plt.ylabel('Y-coordinate')
    plt.title('U-velocity along vertical centerline')
    plt.legend()
    plt.grid(True)
    
    # V velocity along horizontal centerline
    plt.subplot(1, 2, 2)
    plt.plot(sim_x, sim_v, 'b-', label='Simulation')
    plt.plot(ghia_x, ghia_v, 'ro', label='Ghia et al. (1982)')
    plt.xlabel('X-coordinate')
    plt.ylabel('V-velocity')
    plt.title('V-velocity along horizontal centerline')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def save_simulation_results(X_p, Y_p, u, v, p, filename='cfd_results.npz'):
    """Save simulation results to a file"""
    np.savez(filename, X_p=X_p, Y_p=Y_p, u=u, v=v, p=p)
    print(f"Simulation results saved to {filename}")

def load_simulation_results(filename='cfd_results.npz'):
    """Load simulation results from a file"""
    data = np.load(filename)
    return data['X_p'], data['Y_p'], data['u'], data['v'], data['p']

def run_simulation(u, v, p, h, iterations):
    for _ in tqdm(range(iterations)):
        u_temp, v_temp = solve_momentum(u, v, h)
        u_temp, v_temp = enforce_boundary_conditions(u_temp, v_temp)
        dp_dx, dp_dy, p_next = solve_pressure(u_temp, v_temp, p, h)
        u_next = u_temp - DELTA_T / DENSITY * dp_dx
        v_next = v_temp - DELTA_T / DENSITY * dp_dy
        u_next, v_next = enforce_boundary_conditions(u_next, v_next)
        u, v, p = u_next, v_next, p_next
    return u, v, p

def main():
    result_file = 'cfd_results.npz'

    if os.path.exists(result_file) and RESUME_CALCULATION:
        print(f"Resuming calculation from {result_file}...")
        X_p, Y_p, u, v, p = load_simulation_results(result_file)
        h = DOMAIN_SIZE / (GRID_SIZE - 1)
        X_p, Y_p, X_u, Y_u, X_v, Y_v = create_staggered_grid()
        ensure_stability(h)
        print(f"Continuing for {ITERATIONS} more iterations...")
        u, v, p = run_simulation(u, v, p, h, ITERATIONS)
        save_simulation_results(X_p, Y_p, u, v, p, result_file)

    elif os.path.exists(result_file):
        print(f"Loading existing results from {result_file}")
        X_p, Y_p, u, v, p = load_simulation_results(result_file)

    else:
        print("Running new simulation...")
        h = DOMAIN_SIZE / (GRID_SIZE - 1)
        X_p, Y_p, X_u, Y_u, X_v, Y_v = create_staggered_grid()
        u = np.zeros_like(X_u)
        v = np.zeros_like(X_v)
        p = np.zeros_like(X_p)
        ensure_stability(h)
        u, v = enforce_boundary_conditions(u, v)
        u, v, p = run_simulation(u, v, p, h, ITERATIONS)
        save_simulation_results(X_p, Y_p, u, v, p, result_file)

    print("Visualizing results...")
    visualization(X_p, Y_p, p, u, v)

    print("Validating against benchmark data...")
    validate_with_ghia(X_p, Y_p, p, u, v)

if __name__ == "__main__":
    main()