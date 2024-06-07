import numpy as np
import heyoka as hy

def build_taylor_pmp(L, MU, GAMMA, OMEGA):
    """Builds an integrator for the state costate equation of the PMP
       for an optimal transfer with constant acceleration in a rotating frame (axis z)

    Args:
        L (float): units for length (in m)
        MU (float): units for the gravitational parametrs (in kg m^2/s^3)
        GAMMA (float): constant acceleration (in N)
        OMEGA (float): angular velocity (in rad/sec)

    Returns:
        [hy.taylor_adaptive]: the adaptive integartor with state (x,y,z,vx,vy,vz,lx,ly,lz,lvx,lvy,lvz)
    """
    # Unit definitions:
    TIME   = np.sqrt(L**3/MU)                           # Unit for time (period)
    ACC    = L/TIME**2                                  # Unit for accelerations
    
    # Non-dimensionalize:
    mu = 1.
    GAMMA = GAMMA / ACC
    OMEGA = OMEGA * TIME
    
    # Create symbolic variables
    x, y, z, vx, vy, vz, lambda_x, lambda_y, lambda_z, lambda_vx, lambda_vy, lambda_vz = hy.make_vars("x", "y", "z", "vx", "vy", "vz", "lambda_x", "lambda_y", "lambda_z", "lambda_vx", "lambda_vy", "lambda_vz")

    # Optimal thrust angle theta and phi
    ix = -lambda_vx/((lambda_vx**2+lambda_vy**2+lambda_vz**2)**0.5)
    iy = -lambda_vy/((lambda_vx**2+lambda_vy**2+lambda_vz**2)**0.5)
    iz = -lambda_vz/((lambda_vx**2+lambda_vy**2+lambda_vz**2)**0.5)
    
    # Hamiltonian
    H = lambda_x*vx+ \
        lambda_y*vy+ \
        lambda_z*vz+ \
        lambda_vx*(-mu*x/((x**2+y**2+z**2)**(3/2)) + 2*OMEGA*vy + OMEGA**2*x + GAMMA*ix)+ \
        lambda_vy*(-mu*y/((x**2+y**2+z**2)**(3/2)) - 2*OMEGA*vx + OMEGA**2*y + GAMMA*iy)+ \
        lambda_vz*(-mu*z/((x**2+y**2+z**2)**(3/2)) + GAMMA*iz ) \
        + 1
    
    # Create Taylor integrators (for the dynamics we do not use the hamiltonian derivative as that would bloat uselessly the expression)
    ta = hy.taylor_adaptive(sys = [
        (x,  hy.diff(H,lambda_x) ),
        (y,  hy.diff(H,lambda_y) ),
        (z,  hy.diff(H,lambda_z) ),
        (vx, -mu*x/((x**2+y**2+z**2)**(3/2)) + 2*OMEGA*vy + OMEGA**2*x + GAMMA*ix),
        (vy, -mu*y/((x**2+y**2+z**2)**(3/2)) - 2*OMEGA*vx + OMEGA**2*y + GAMMA*iy) ,
        (vz, -mu*z/((x**2+y**2+z**2)**(3/2)) + GAMMA*iz) ,
        (lambda_x,  -hy.diff(H,x) ),
        (lambda_y,  -hy.diff(H,y) ),
        (lambda_z,  -hy.diff(H,z) ),
        (lambda_vx, -hy.diff(H,vx) ),
        (lambda_vy, -hy.diff(H,vy) ),
        (lambda_vz, -hy.diff(H,vz) )],
        # Initial conditions:
        state = [1.] * 12,
        # Initial value of time variable:
        time = 0.)
    return ta

def build_taylor_ffnn(L, MU, GAMMA, OMEGA, thrust_ffnn, sma_callback = None, tol=1e-16):
    """Builds an integrator for the state equation of the optimal transfer with constant acceleration
      in a rotating frame (axis z). The Thrust direction is given by a ffnn

    Args:
        L (float): units for length (in m)
        MU (float): units for the gravitational parametrs (in kg m^2/s^3)
        GAMMA (float): constant acceleration (in N)
        OMEGA (float): angular velocity (in rad/sec)
        thrust_ffnn (heyoka expression): the ffnn
        sma_callback (callback for terminal event): adds the event tracking on sma

    Returns:
        [hy.taylor_adaptive]: the adaptive integartor with state (x,y,z,vx,vy,vz)
    """
       # Unit definitions:
    TIME   = np.sqrt(L**3/MU)                           # Unit for time (period)
    ACC    = L/TIME**2                                  # Unit for accelerations
    
    # Non-dimensionalize:
    mu = 1.
    GAMMA = GAMMA / ACC
    OMEGA = OMEGA * TIME
    
    # Create symbolic variables
    x, y, z, vx, vy, vz = hy.make_vars("x", "y", "z", "vx", "vy", "vz")

    events = []
    # Create event (if callback)
    if sma_callback:
        r = hy.sqrt(x**2+y**2+z**2)
        v2 = (vx-OMEGA*y)**2+(vy+OMEGA*x)**2+vz**2
        t_ev = hy.t_event(
            # The event equation.
            v2/2.-mu/r+mu/2./1.3,
            # The callback.
            callback = sma_callback)
        events.append(t_ev)

    # Optimal thrust angle theta and phi
    inputs = [ x, y, z, vx, vy, vz]
    ix, iy, iz = thrust_ffnn.compute_heyoka_expression(inputs)
    norm_outputs = hy.sqrt(ix**2+iy**2+iz**2)
    ix = ix / norm_outputs
    iy = iy / norm_outputs
    iz = iz / norm_outputs
    # Create Taylor integrator
    ta = hy.taylor_adaptive(sys = [
        (x, vx),
        (y, vy),
        (z, vz),
        (vx, -mu*x/((x**2+y**2+z**2)**(3/2)) + 2*OMEGA*vy + OMEGA**2*x + GAMMA*ix),
        (vy, -mu*y/((x**2+y**2+z**2)**(3/2)) - 2*OMEGA*vx + OMEGA**2*y + GAMMA*iy),
        (vz, -mu*z/((x**2+y**2+z**2)**(3/2)) + GAMMA*iz),
        ],
        # Initial conditions:
        state = [1., 1., 1., 1., 1., 1.],
        # Initial value of time variable:
        time = 0.,
        compact_mode=True,
        t_events = events,
        tol=tol)
    return ta