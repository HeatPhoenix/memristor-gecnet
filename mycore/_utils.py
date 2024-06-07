import numpy as np
import pykep as pk

def edelbaum(a0, af, delta_i, acc=1e-4, mu=pk.MU_SUN):
    """The Edelbaum approximation estimates the DV and time-of-flight of time-optimal low-thrust
    trajectories between circular orbits

    See: Kluever, Craig A. "Using edelbaum's method to compute Low-Thrust transfers with earth-shadow eclipses."
         Journal of Guidance, Control, and Dynamics 34.1 (2011): 300-303.

    Args:
        a0 (float): starting orbit semi-major axis (m)
        af (float): target orbit semi-major axis (m)
        delta_i (float): inclination change (rad)
        acc (float): low-thrust acceleration
        mu (float): gravitational parameter of central body

    Returns:
        DV (float): estimate for the total DV needed (m/sec)
        DT (float): estimate for the transfer time (sec)
    """
    # circular velocity at departure
    v0 = np.sqrt(mu / a0)
    # circular velocity at arrival
    vf = np.sqrt(mu / af)
    # Edelbaum formula for the DV
    DV = np.sqrt(v0 ** 2 + vf ** 2 - 2 * v0 * vf * np.cos(np.pi / 2 * delta_i))
    # Edelbaum formula for the DT
    DT = DV / acc
    return DV, DT