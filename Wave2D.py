import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation


x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        L = 1
        x = np.linspace(0, L, N+1)
        y = np.linspace(0, L, N+1)

        self.xij, self.yij = np.meshgrid(x, y, indexing='ij')
        self.h = L / N  # mesh step size

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1 
        D[-1, -4:] = -1, 4, -5, 2
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        k_x = self.mx * sp.pi
        k_y = self.my * sp.pi
        w = self.c * sp.sqrt((k_x**2 + k_y **2))
        return w

    def ue(self, mx, my):
        """Return the exact standing wave"""
        # Dirichlet real stationary wave solution (eq.q 1.4) where k = m*pi
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.create_mesh(N)

        u_exact = sp.lambdify((x,y,t), self.ue(mx, my))

        self.Unm1 = u_exact(self.xij, self.yij, 0) # U_0
        self.Un = u_exact(self.xij, self.yij, self.dt) # U_1


    @property
    def dt(self):
        """Return the time step"""
        dt = self.cfl * self.h / self.c
        return dt

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        # using eq 1.4 as exact
        u_exact = sp.lambdify((x,y,t), self.ue(self.mx, self.my))
        u_e = u_exact(self.xij, self.yij, t0)
                
        l2_error = np.sqrt((self.h**2) * np.sum((u - u_e)**2))
        return l2_error
    

    def apply_bcs(self):
        # Fixed boundaries at 0
        self.Unp1[0, :] = self.Unp1[-1, :] = self.Unp1[:, 0] = self.Unp1[:, -1] = 0

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        # self.Nt = Nt
        self.cfl = cfl
        self.c = c
        self.mx = mx
        self.my = my

        # Make the 3 matrices that a re necessary at each timestep
        self.initialize(N, mx, my)
        self.Unp1, self.Un, self.Unm1 = np.zeros((N+1, N+1)), self.Un, self.Unm1

        # Set up differentioation matrix 
        D = self.D2(N) / self.h**2
        dt = self.dt

        # plotdata = {}
        # Plotting data
        if store_data > 0:
            plotdata = {0: self.Unm1.copy()}
        elif store_data == -1:
            l2_errors = np.zeros(Nt)
            l2_errors[0] = self.l2_error(self.Unm1, 0)
        

        for n in range(1, Nt):
            # Compute next timestep using central difference in time and space
            self.Unp1[:] = 2*self.Un - self.Unm1 + (c*dt)**2*(D @ self.Un + self.Un @ D.T)
            
            # Set boundary conditions
            self.apply_bcs()

            # Swap solutions
            self.Unm1[:] = self.Un
            self.Un[:] = self.Unp1
            if store_data > 0:
                plotdata[n]= self.Un.copy()
            elif store_data == -1:
                l2_errors[n] = self.l2_error(self.Unm1, n * dt)

        # For `convergence_rates`
        if store_data == -1:
            return (self.h, l2_errors)
        elif store_data > 0:
            return plotdata


    def convergence_rates(self, m=7, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :2] = -2, 2
        D[-1, -2:] = 2, -2
        return D

    def ue(self, mx, my):
        # Dirichlet real stationary wave solution (eq.q 1.5) where k = m*pi
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)

    def apply_bcs(self):
        # moved setting if boundary conditions to D2
        pass


def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    CFL = 1 / np.sqrt(2)
    mx = my = 2

    sol = Wave2D()
    h, l2_e = sol(N=100, Nt=10 , cfl=CFL, mx=mx, my=my, store_data=-1)
    # print(l2_e[])
    assert l2_e[-1] < 1e-12

    solN = Wave2D_Neumann()
    h, l2_e = sol(N=100, Nt=10 , cfl=CFL, mx=mx, my=my, store_data=-1)
    assert l2_e[-1] < 1e-12
    
    


if __name__ == '__main__':
    # tests
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    test_exact_wave2d()

    N = 200
    Nt = 200
    cfl = 1 / np.sqrt(2)
    mx = 2
    my = 2

    wave = Wave2D_Neumann()
    plotdata = wave(N, Nt, cfl=cfl, mx=mx, my=my, store_data=200)

    # Neumann MOVIE
    xij = wave.xij
    yij = wave.yij
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    frames = []
    for n, val in plotdata.items():
        if n % 2 == 0:
            frame = ax.plot_surface(xij, yij, val, vmin=-0.5*plotdata[0].max(),
                                vmax=plotdata[0].max(), cmap=cm.ocean,
                                linewidth=.5, antialiased=False)
            frames.append([frame])

    ani = animation.ArtistAnimation(fig, frames, interval=600, blit=True,
                                    repeat_delay=1000)
    # ani.save('report/wavemovie_neumann.gif', writer='pillow', fps=30)

    # Dirichlet MOVIE
    wave = Wave2D()
    plotdata = wave(N, Nt, cfl=cfl, mx=mx, my=my, store_data=200)

    # Neumann MOVIE
    xij = wave.xij
    yij = wave.yij
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    frames = []
    for n, val in plotdata.items():
        if n % 2 == 0:
            frame = ax.plot_surface(xij, yij, val, vmin=-0.5*plotdata[0].max(),
                                vmax=plotdata[0].max(), cmap=cm.ocean,
                                linewidth=.5, antialiased=False)
            frames.append([frame])

    ani = animation.ArtistAnimation(fig, frames, interval=600, blit=True,
                                    repeat_delay=1000)
    # ani.save('report/wavemovie_dirichlet.gif', writer='pillow', fps=30)

