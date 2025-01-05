import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# This code is inspired by https://github.com/Ceyron/lid-driven-cavity-python/tree/main

# Meta information
GRID_SIZE = 41 
DOMAIN_SIZE = 1.0
ITERATIONS = 1000
DELTA_T = 0.001
PRESSURE_POISSON_ITERATIONS = 50
STABILITY_SAFETY_FACTOR = 0.5

# Information
KINEMATIC_VISCOSITY = 0.01
DENSITY = 1.0

# Boundary conditions
U_TOP = 1.0

def create_grid():
    x = np.linspace(0.0, DOMAIN_SIZE, GRID_SIZE)
    y = np.linspace(0.0, DOMAIN_SIZE, GRID_SIZE)

    X, Y = np.meshgrid(x, y)
    return X, Y

def differentiate_x(values, element_length):
    result = np.zeros_like(values)
    # Loop over rows
    for i in range(1, values.shape[0] - 1):
        # Loop over cols
        for j in range(1, values.shape[1] - 1):
            # Differentiate i, j field numerically with respect to x
            # (value right - value left) / (2 * element_length)
            result[i, j] = (values[i, j+1] - values[i, j-1]) / (2 * element_length)
    
    return result
    
def differentiate_y(values, element_length):
    result = np.zeros_like(values)
    # Loop over rows
    for i in range(1, values.shape[0] - 1):
        # Loop over cols
        for j in range(1, values.shape[1] - 1):
            # Differentiate i, j field numerically with respect to y
            # (value below - value above) / (2 * element_length)
            result[i, j] = (values[i+1, j] - values[i-1, j]) / (2 * element_length)

    return result

def laplace(values, element_length):
    result = np.zeros_like(values)
    # Loop over rows
    for i in range(1, values.shape[0] - 1):
        # Loop over cols
        for j in range(1, values.shape[1] - 1):
            result[i, j] = (
                values[i, j - 1]   # left
                + values[i - 1, j] # up
                - 4 * values[i, j] # center
                + values[i, j + 1] # right
                + values[i + 1, j] # down
            ) / (element_length**2)

    return result

def ensure_stability(element_length):
    # From https://github.com/Ceyron/lid-driven-cavity-python/tree/main
    maximum_possible_time_step_length = (0.5 * element_length**2 / KINEMATIC_VISCOSITY)
    if DELTA_T > (STABILITY_SAFETY_FACTOR * maximum_possible_time_step_length):
        raise RuntimeError("Stability is not guarenteed.")

def solve_momentum(u, v, element_length):
    du_dx = differentiate_x(u, element_length) 
    du_dy = differentiate_y(u, element_length)
    dv_dx = differentiate_x(v, element_length)
    dv_dy = differentiate_y(v, element_length)
    laplace_u = laplace(u, element_length)
    laplace_v = laplace(v, element_length)
    
    # Calculate temporary u_{t+1} and v_{t+1}
    u_temp = (u + DELTA_T * (-(u * du_dx + v * du_dy) + KINEMATIC_VISCOSITY * laplace_u))
    v_temp = (v + DELTA_T * (-(u * dv_dx + v * dv_dy) + KINEMATIC_VISCOSITY * laplace_v))

    return u_temp, v_temp

def solve_pressure(u_temp, v_temp, p, element_length): 
    du_temp_dx = differentiate_x(u_temp, element_length)
    dv_temp_dy = differentiate_y(v_temp, element_length)
    
    # Calculate the right hand side of \nabla^2 p=\frac{\rho}{\Delta t} \cdot \nabla u
    # Nabla times u is a scalar field
    rhs = (DENSITY / DELTA_T * (du_temp_dx + dv_temp_dy))
    
    # Calculate preassure by iterating
    for _ in range(PRESSURE_POISSON_ITERATIONS):
        p_next = np.zeros_like(p)

        # Loop over rows
        for i in range(1, p.shape[0] - 1):
            # Loop over cols
            for j in range(1, p.shape[1] - 1):
                p_next[i, j] = 0.25 * (
                    p[i, j - 1]     # left
                    + p[i - 1, j]   # up
                    + p[i, j + 1]   # right
                    + p[i + 1, j]   # down
                    - (element_length**2) * rhs[i, j]) 
    
        # Pressure Boundary Conditions
        p_next[:, -1] = p_next[:, -2]  # right: Neumann
        p_next[0,  :] = p_next[1,  :]  # top: Neumann
        p_next[:,  0] = p_next[:,  1]  # left: Neumann
        p_next[-1, :] = 0.0            # bottom: Dirichlet (y is inverted)
    
        p = p_next
    
    dp_next_dx = differentiate_x(p_next, element_length)
    dp_next_dy = differentiate_y(p_next, element_length)
    return dp_next_dx, dp_next_dy, p_next

def enforce_boundary_conditions(u, v):
    u[0, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    u[-1, :] = U_TOP # y is inverted
    v[0, :] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0
    v[-1, :] = 0.0

    return u, v

def visualization(X, Y, p_next, u_next, v_next):
    # From https://github.com/Ceyron/lid-driven-cavity-python/tree/main
    # The [::2, ::2] selects only every second entry (less cluttering plot)
    plt.style.use("dark_background")
    plt.figure()
    plt.contourf(X[::2, ::2], Y[::2, ::2], p_next[::2, ::2], cmap="coolwarm")
    plt.colorbar()

    plt.quiver(X[::2, ::2], Y[::2, ::2], u_next[::2, ::2], v_next[::2, ::2], color="black")
    #plt.streamplot(X[::2, ::2], Y[::2, ::2], u_next[::2, ::2], v_next[::2, ::2], color="black")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.show()

def main():
    # Grid
    element_length = DOMAIN_SIZE / (GRID_SIZE - 1)
    X, Y = create_grid() 

    # Initial values
    u_init = np.zeros_like(X)
    v_init = np.zeros_like(X)
    p_init = np.zeros_like(X)

    # Numerical stability
    ensure_stability(element_length)

    # Prepare calculation
    u = u_init
    v = v_init
    p = p_init

    # Main loop
    for _ in tqdm(range(ITERATIONS)):
        # Get temp u and v
        u_temp, v_temp = solve_momentum(u, v, element_length)

        # Velocity boundary conditions
        u_temp, v_temp = enforce_boundary_conditions(u_temp, v_temp)

        # Get next p, next dp/dx and next dp/dy
        dp_next_dx, dp_next_dy, p_next = solve_pressure(u_temp, v_temp, p, element_length)

        # Correct the velocities
        u_next = (u_temp - DELTA_T / DENSITY * dp_next_dx)
        v_next = (v_temp - DELTA_T / DENSITY * dp_next_dy)

        # Velocity boundary conditions
        u_next, v_next = enforce_boundary_conditions(u_next, v_next)

        # Prepare next step
        u, v, p = u_next, v_next, p_next

    # Visualize
    visualization(X, Y, p_next, u_next, v_next)

if __name__ == "__main__":
    main()