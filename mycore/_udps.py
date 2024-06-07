import numpy as np
import heyoka as hy
import pykep as pk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from . import build_taylor_pmp

class ast2station_rotating:
    """ This udp represents a shooting function that solves the OCP
        from an asteroid to a point fixed in the rotating frame

        The decision vector will contain:
            [lx,ly,lz,lvx,lvy,lvz,tof]
    """
    def __init__(self, x0, r_target):
        """Constructor

        Args:
            x0 (list): initial state x,y,z,vx,vy,vz (in the rotating frame, in SI)
            r_target (float): target radius (in SI)
        """
        # Problem definition
        self.x0 = np.array(x0)
        self.r_target = r_target
        
        # Instantiates the Taylor integrator for the augmented dynamics
        self.GAMMA = 1e-4
        self.OMEGA = np.sqrt(pk.MU_SUN/r_target**3)
        self.ta = build_taylor_pmp(L = pk.AU, MU = pk.MU_SUN, GAMMA = self.GAMMA, OMEGA = self.OMEGA)

        # Unit definitions
        self.L = pk.AU
        self.TIME = np.sqrt(self.L**3/pk.MU_SUN)            # Unit fot times
        self.V = self.L / self.TIME                         # Unit for velocities
        self.ACC = self.L / self.TIME**2                    # Unit for accelerations
        
        # Non dimensional 
        self.x0[:3] = self.x0[:3] / self.L
        self.x0[3:] = self.x0[3:] / self.V
        self.GAMMA = self.GAMMA / self.ACC
        self.OMEGA = self.OMEGA * self.TIME
        self.r_target = self.r_target / self.L
           
    def fitness(self, x):
        """Definition of fitness function and equality constraints
        
        Args:
            chromosome (list): initial conditions: [lambda_x0, lambda_y0, lambda_z0, lambda_vx0, lambda_vy0, lambda_vz0]

        Returns:
            [list]: 1 objective function and 7 equality constraints
        """
        obj = 1.
        
        # We build part of the initial conditions from the decision vector
        self.ta.state[6:] = x[:-1]
        
        # .. and reset the initial conditions on the states and time (the ta object has been used before)
        self.ta.state[:6] = self.x0
        self.ta.time = 0.
                
        # Propagate and extract final states and co-states
        self.ta.propagate_until(x[-1])
        states = self.ta.state
        xf, yf, zf, vxf, vyf, vzf, lambda_xf, lambda_yf, lambda_zf, lambda_vxf, lambda_vyf, lambda_vzf = states
        
        # Optimal thrust angle theta and phi
        ixf = -lambda_vxf/((lambda_vxf**2+lambda_vyf**2+lambda_vzf**2)**0.5)
        iyf = -lambda_vyf/((lambda_vxf**2+lambda_vyf**2+lambda_vzf**2)**0.5)
        izf = -lambda_vzf/((lambda_vxf**2+lambda_vyf**2+lambda_vzf**2)**0.5)

        # Hamiltonian
        mu = 1.
        # Hamiltonian
        Hf = lambda_xf*vxf+ \
             lambda_yf*vyf+ \
             lambda_zf*vzf+ \
             lambda_vxf*(-mu*xf/((xf**2+yf**2+zf**2)**(3/2)) + 2*self.OMEGA*vyf + self.OMEGA**2*xf + self.GAMMA*ixf)+ \
             lambda_vyf*(-mu*yf/((xf**2+yf**2+zf**2)**(3/2)) - 2*self.OMEGA*vxf + self.OMEGA**2*yf + self.GAMMA*iyf)+ \
             lambda_vzf*(-mu*zf/((xf**2+yf**2+zf**2)**(3/2)) + self.GAMMA*izf ) \
             + 1
        
        # Target states
        xt, yt, zt = self.r_target, 0., 0.
        vxt, vyt, vzt = 0., 0., 0.

        ce1 = xf - xt
        ce2 = yf - yt
        ce3 = zf - zt
        ce4 = vxf - vxt
        ce5 = vyf - vyt
        ce6 = vzf - vzt
        ce7 = (Hf)
        #ce7 = np.linalg.norm(x[:-1]) - 1.
        
        return [obj, ce1, ce2, ce3, ce4, ce5, ce6, ce7]        
    
    def get_nec(self):
        """
        - Returns number of equality constraints
        
        """ 
        return 7  

    def get_bounds(self):
        """
        - Return bounds for initial co-states

        """ 
        return ([-1e4, -1e4, -1e4, -1e4, -1e4, -1e4, 0.01], [1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 100])
    
    def plot_initial_conditions(self, axes = None):
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111, projection="3d")
        theta = np.linspace(0, 2*np.pi)
        X = self.r_target * np.cos(theta) 
        Y = self.r_target * np.sin(theta) 
        Z = X*0
        axes.plot(X,Y,Z)
        axes.scatter(self.r_target, 0., 0.)
        axes.scatter(self.x0[0], self.x0[1], self.x0[2])
        return axes
    
    def plot(self, x, axes=None, N=100, inertial = False):
        # Creating the plot axes
        axes = self.plot_initial_conditions(axes)

        # Initial conditions
        self.ta.state[6:] = x[:-1]
        self.ta.state[:6] = self.x0
        self.ta.time = 0.
                
        # Propagate on the time grid
        t_grid = np.linspace(0, x[-1], N)
        sol = self.ta.propagate_grid(t_grid)[4]
        
        if inertial:
            sol_inertial_x = sol[:,0] * np.cos(self.OMEGA * t_grid) - sol[:,1] * np.sin(self.OMEGA * t_grid)
            sol_inertial_y = sol[:,1] * np.cos(self.OMEGA * t_grid) + sol[:,0] * np.sin(self.OMEGA * t_grid)
            # We plot
            axes.plot(sol_inertial_x, sol_inertial_y, sol[:, 2], "k")
            target_x = self.r_target  * np.cos(self.OMEGA * x[-1])
            target_y = self.r_target  * np.sin(self.OMEGA * x[-1])
            axes.scatter(target_x, target_y, 0., 'r')
        else:
            # We plot
            axes.plot(sol[:, 0], sol[:, 1], sol[:, 2], "k")
        return axes

    
class ast2station_rotating2:
    """ This udp represents a shooting function that solves the OCP from an
        asteroid to target point fixed in he circular frame. It also normalizes the
        costates (see Baoyin paper) adding a degree of freedom in a coefficient before J

        The decision vector will contain:
            [lx,ly,lz,lvx,lvy,lvz,lJ,tof]
    """
    def __init__(self, x0, r_target):
        """Constructor

        Args:
            x0 (list): initial state x,y,z,vx,vy,vz (in the rotating frame, in SI)
            r_target (float): target radius (in SI)
        """
        # Problem definition
        self.x0 = np.array(x0)
        self.r_target = r_target
        
        # Instantiates the Taylor integrator for the augmented dynamics
        self.GAMMA = 1e-4
        self.OMEGA = np.sqrt(pk.MU_SUN/r_target**3)
        self.ta = build_taylor_pmp(L = pk.AU, MU = pk.MU_SUN, GAMMA = self.GAMMA, OMEGA = self.OMEGA)

        # Unit definitions
        self.L = pk.AU
        self.TIME = np.sqrt(self.L**3/pk.MU_SUN)            # Unit fot times
        self.V = self.L / self.TIME                         # Unit for velocities
        self.ACC = self.L / self.TIME**2                    # Unit for accelerations
        
        # Non dimensional 
        self.x0[:3] = self.x0[:3] / self.L
        self.x0[3:] = self.x0[3:] / self.V
        self.GAMMA = self.GAMMA / self.ACC
        self.OMEGA = self.OMEGA * self.TIME
        self.r_target = self.r_target / self.L
           
    def fitness(self, x):
        """Definition of fitness function and equality constraints
        
        Args:
            chromosome (list): initial conditions: [lambda_x0, lambda_y0, lambda_z0, lambda_vx0, lambda_vy0, lambda_vz0]

        Returns:
            [list]: 1 objective function and 7 equality constraints
        """
        obj = 1.
        
        # We build part of the initial conditions from the decision vector
        self.ta.state[6:] = x[:-2]
        
        # .. and reset the initial conditions on the states and time (the ta object has been used before)
        self.ta.state[:6] = self.x0
        self.ta.time = 0.
                
        # Propagate and extract final states and co-states
        self.ta.propagate_until(x[-1])
        states = self.ta.state
        xf, yf, zf, vxf, vyf, vzf, lambda_xf, lambda_yf, lambda_zf, lambda_vxf, lambda_vyf, lambda_vzf = states
        
        # Optimal thrust angle theta and phi
        ixf = -lambda_vxf/((lambda_vxf**2+lambda_vyf**2+lambda_vzf**2)**0.5)
        iyf = -lambda_vyf/((lambda_vxf**2+lambda_vyf**2+lambda_vzf**2)**0.5)
        izf = -lambda_vzf/((lambda_vxf**2+lambda_vyf**2+lambda_vzf**2)**0.5)

        # Hamiltonian
        mu = 1.
        # Hamiltonian
        Hf = lambda_xf*vxf+ \
             lambda_yf*vyf+ \
             lambda_zf*vzf+ \
             lambda_vxf*(-mu*xf/((xf**2+yf**2+zf**2)**(3/2)) + 2*self.OMEGA*vyf + self.OMEGA**2*xf + self.GAMMA*ixf)+ \
             lambda_vyf*(-mu*yf/((xf**2+yf**2+zf**2)**(3/2)) - 2*self.OMEGA*vxf + self.OMEGA**2*yf + self.GAMMA*iyf)+ \
             lambda_vzf*(-mu*zf/((xf**2+yf**2+zf**2)**(3/2)) + self.GAMMA*izf ) \
             + x[-2]
        
        # Target states
        xt, yt, zt = self.r_target, 0., 0.
        vxt, vyt, vzt = 0., 0., 0.

        ce1 = xf - xt
        ce2 = yf - yt
        ce3 = zf - zt
        ce4 = vxf - vxt
        ce5 = vyf - vyt
        ce6 = vzf - vzt
        ce7 = Hf
        ce8 = np.linalg.norm(x[:-1]) - 1.
        
        return [obj, ce1, ce2, ce3, ce4, ce5, ce6, ce7, ce8]        

    def set_x0(self, x0):
        x0 = np.array(x0)
        self.x0[:3] = x0[:3] / self.L
        self.x0[3:] = x0[3:] / self.V
    
    def get_nec(self):
        """
        - Returns number of equality constraints
        
        """ 
        return 8

    def get_bounds(self):
        """
        - Return bounds for initial co-states

        """ 
        return ([-1e4, -1e4, -1e4, -1e4, -1e4, -1e4, 0, 0.01], [1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 100])
    
    def plot_initial_conditions(self, axes = None):
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111, projection="3d")
        theta = np.linspace(0, 2*np.pi)
        X = self.r_target * np.cos(theta) 
        Y = self.r_target * np.sin(theta) 
        Z = X*0
        axes.plot(X,Y,Z)
        axes.scatter(self.r_target, 0., 0.)
        axes.scatter(self.x0[0], self.x0[1], self.x0[2])
        return axes
    
    def plot(self, x, axes=None, N=100, inertial = False):
        # Creating the plot axes
        axes = self.plot_initial_conditions(axes)

        # Initial conditions
        self.ta.state[6:] = x[:-2]
        self.ta.state[:6] = self.x0
        self.ta.time = 0.
                
        # Propagate on the time grid
        t_grid = np.linspace(0, x[-1], N)
        sol = self.ta.propagate_grid(t_grid)[4]
        
        if inertial:
            sol_inertial_x = sol[:,0] * np.cos(self.OMEGA * t_grid) - sol[:,1] * np.sin(self.OMEGA * t_grid)
            sol_inertial_y = sol[:,1] * np.cos(self.OMEGA * t_grid) + sol[:,0] * np.sin(self.OMEGA * t_grid)
            # We plot
            axes.plot(sol_inertial_x, sol_inertial_y, sol[:, 2], "k")
            target_x = self.r_target * np.cos(self.OMEGA * x[-1])
            target_y = self.r_target * np.sin(self.OMEGA * x[-1])
            axes.scatter(target_x, target_y, 0., 'r')
        else:
            # We plot
            axes.plot(sol[:, 0], sol[:, 1], sol[:, 2], "k")
        return axes
