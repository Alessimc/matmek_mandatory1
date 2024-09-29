import numpy as np
import sympy as sp
import scipy.sparse as sparse
from scipy.interpolate import interpn

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    def __init__(self, L, ue):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L
        self.ue = ue
        self.f = sp.diff(self.ue, x, 2)+sp.diff(self.ue, y, 2)

    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = N
        x = np.linspace(0, self.L, N+1)
        y = np.linspace(0, self.L, N+1)
        self.x_grid = x # for interpolation later
        self.y_grid = y
        self.xij, self.yij = np.meshgrid(x, y, indexing='ij')
        self.h = self.L / N  # mesh step size
        return self.xij, self.yij

    def D2(self):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    def laplace(self):
        """Return vectorized Laplace operator"""

        D2x = (1./self.h**2)*self.D2()
        D2y = (1./self.h**2)*self.D2()
        return (sparse.kron(D2x, sparse.eye(self.N+1)) +
                sparse.kron(sparse.eye(self.N+1), D2y))

    def get_boundary_indices(self):
        """Return indices of vectorized matrix that belongs to the boundary"""
        B = np.ones((self.N+1, self.N+1), dtype=bool)
        B[1:-1, 1:-1] = 0
        bnds = np.where(B.ravel() == 1)[0]
        return bnds

    def assemble(self):
        """Return assembled matrix A and right hand side vector b"""
        N = self.N+1  # including boundaries
        lap = self.laplace()
        
        # Vectorized grid points for x and y
        f_vals = np.vectorize(sp.lambdify((x, y), self.f))(self.xij, self.yij)
        
        # Flatten the source term and apply step size scaling
        b = f_vals.ravel()
        
        # Apply Dirichlet boundary conditions
        bnds = self.get_boundary_indices()
        A = lap.tolil()

        # Modify A and b to account for boundary conditions
        for idx in bnds:
            A[idx, :] = 0
            A[idx, idx] = 1
            b[idx] = np.vectorize(sp.lambdify((x, y), self.ue))(self.xij.ravel()[idx], self.yij.ravel()[idx])

        return A, b

    def l2_error(self, u):
        """Return l2-error norm"""
        # Evaluate exact solution on the grid
        u_exact = np.vectorize(sp.lambdify((x, y), self.ue))(self.xij, self.yij)
        
        # Compute error and L2 norm
        error = u - u_exact
        l2_norm = np.sqrt(np.sum(error**2) * self.h**2) 
        return l2_norm

    def __call__(self, N):
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array

        """
        self.create_mesh(N)
        A, b = self.assemble()
        A = A.tocsr()
        self.U = sparse.linalg.spsolve(A, b.flatten()).reshape((N+1, N+1))
        return self.U

    def convergence_rates(self, m=6):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretization levels to use

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
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(self.h)
            N0 *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

    def eval(self, x, y):
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """
        point = [x, y]
        grid_points = (self.x_grid, self.y_grid)

        # cubic interpolation using interpn
        u = interpn(grid_points, self.U, point, method='cubic')

        return u

def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2

def test_interpolation():
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h/2, y: 1-sol.h/2}).n()) < 1e-3


# testing

test_convergence_poisson2d()
test_interpolation()