import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Meta information
GRID_SIZE = 41 #41
DOMAIN_SIZE = 1.0
ITERATIONS = 500
DELTA_T = 0.001
KINEMATIC_VISCOSITY = 0.1 #TODO this will be 0.01
DENSITY = 1.0

# Boundary conditions
U_TOP = 1.0

# Other
PRESSURE_POISSON_ITERATIONS = 50
STABILITY_SAFETY_FACTOR = 0.5

def create_grid():
    x = np.linspace(0.0, DOMAIN_SIZE, GRID_SIZE)
    y = np.linspace(0.0, DOMAIN_SIZE, GRID_SIZE)

    X, Y = np.meshgrid(x, y)
    return X, Y

def central_difference_x(f, element_length):
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (
        f[1:-1, 2:  ]
        -
        f[1:-1, 0:-2]
    ) / (
        2 * element_length
    )
    return diff
    
def central_difference_y(f, element_length):
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (
        f[2:  , 1:-1]
        -
        f[0:-2, 1:-1]
    ) / (
        2 * element_length
    )
    return diff

def laplace(f, element_length):
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (
        f[1:-1, 0:-2]
        +
        f[0:-2, 1:-1]
        -
        4
        *
        f[1:-1, 1:-1]
        +
        f[1:-1, 2:  ]
        +
        f[2:  , 1:-1]
    ) / (
        element_length**2
    )
    return diff

def ensure_stability(element_length):
    maximum_possible_time_step_length = (0.5 * element_length**2 / KINEMATIC_VISCOSITY)
    if DELTA_T > (STABILITY_SAFETY_FACTOR * maximum_possible_time_step_length):
        raise RuntimeError("Stability is not guarenteed.")

def solve_momentum(u, v, element_length):
    du_dx = central_difference_x(u, element_length) 
    du_dy = central_difference_y(u, element_length)
    dv_dx = central_difference_x(v, element_length)
    dv_dy = central_difference_y(v, element_length)
    laplace_u = laplace(u, element_length)
    laplace_v = laplace(v, element_length)
    
    # Calculate temporary u_{i+1} and v_{i+1}
    u_temp = (u + DELTA_T * (-(u * du_dx + v * du_dy) + KINEMATIC_VISCOSITY * laplace_u))
    v_temp = (v + DELTA_T * (-(u * dv_dx + v * dv_dy) + KINEMATIC_VISCOSITY * laplace_v))

    return u_temp, v_temp

def solve_pressure(u_temp, v_temp, p, element_length):
    du_temp_dx = central_difference_x(u_temp, element_length)
    dv_temp_dy = central_difference_y(v_temp, element_length)
    
    # Compute a pressure correction by solving the pressure-poisson equation
    rhs = (DENSITY / DELTA_T * (du_temp_dx + dv_temp_dy))
    
    for _ in range(PRESSURE_POISSON_ITERATIONS):
        p_next = np.zeros_like(p)
        p_next[1:-1, 1:-1] = 1/4 * (
            +
            p[1:-1, 0:-2]
            +
            p[0:-2, 1:-1]
            +
            p[1:-1, 2:  ]
            +
            p[2:  , 1:-1]
            -
            element_length**2
            *
            rhs[1:-1, 1:-1]
        )
    
        # Pressure Boundary Conditions: Homogeneous Neumann Boundary
        # Conditions everywhere except for the top, where it is a
        # homogeneous Dirichlet BC
        p_next[:, -1] = p_next[:, -2]
        p_next[0,  :] = p_next[1,  :]
        p_next[:,  0] = p_next[:,  1]
        p_next[-1, :] = 0.0
    
        p_prev = p_next
    
    dp_next_dx = central_difference_x(p_next, element_length)
    dp_next_dy = central_difference_y(p_next, element_length)
    return dp_next_dx, dp_next_dy, p_next

def visualization(X, Y, p_next, u_next, v_next):
    # The [::2, ::2] selects only every second entry (less cluttering plot)
    plt.style.use("dark_background")
    plt.figure()
    plt.contourf(X[::2, ::2], Y[::2, ::2], p_next[::2, ::2], cmap="coolwarm")
    plt.colorbar()

    plt.quiver(X[::2, ::2], Y[::2, ::2], u_next[::2, ::2], v_next[::2, ::2], color="black")
    # plt.streamplot(X[::2, ::2], Y[::2, ::2], u_next[::2, ::2], v_next[::2, ::2], color="black")
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
        u_temp[0, :] = 0.0
        u_temp[:, 0] = 0.0
        u_temp[:, -1] = 0.0
        u_temp[-1, :] = U_TOP
        v_temp[0, :] = 0.0
        v_temp[:, 0] = 0.0
        v_temp[:, -1] = 0.0
        v_temp[-1, :] = 0.0

        dp_next_dx, dp_next_dy, p_next = solve_pressure(u_temp, v_temp, p, element_length)

        # Correct the velocities such that the fluid stays incompressible
        u_next = (u_temp - DELTA_T / DENSITY * dp_next_dx)
        v_next = (v_temp - DELTA_T / DENSITY * dp_next_dy)

        # Velocity boundary conditions
        u_next[0, :] = 0.0
        u_next[:, 0] = 0.0
        u_next[:, -1] = 0.0
        u_next[-1, :] = U_TOP
        v_next[0, :] = 0.0
        v_next[:, 0] = 0.0
        v_next[:, -1] = 0.0
        v_next[-1, :] = 0.0

        # Advance in time
        u = u_next
        v = v_next
        p = p_next

    # Visualize
    visualization(X, Y, p_next, u_next, v_next)

if __name__ == "__main__":
    main()