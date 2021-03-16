import dolfinx
import ufl
import sympy
from mpi4py import MPI
import numpy as np


def test_against_sympy():
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    zero = sympy.Integer(0)
    half = sympy.Rational(1, 2)
    quarter = sympy.Rational(1, 4)

    # A = grad(u) * grad(v) * dx
    # b = f * v * dx
    f = lambda x: -2 * x[0] * (1 - x[0]) - 2 * x[1] * (1 - x[1])
    dirichlet = lambda x: zero

    # Coarse mesh
    # 6-----7-----8
    # |    /|    /|
    # |   / |   / |
    # |  /  |  /  |
    # | /   | /   |
    # |/    |/    |
    # 3-----4-----5
    # |    /|    /|
    # |   / |   / |
    # |  /  |  /  |
    # | /   | /   |
    # |/    |/    |
    # 0-----1-----2
    coarse_vertices = [(sympy.Rational(j, 2), sympy.Rational(i, 2))
                       for i in range(3) for j in range(3)]

    # Fine mesh
    # 20-21-22-23-24
    # | /| /| /| /|
    # |/ |/ |/ |/ |
    # 15-16-17-18-19
    # | /| /| /| /|
    # |/ |/ |/ |/ |
    # 10-11-12-13-14
    # | /| /| /| /|
    # |/ |/ |/ |/ |
    # 5--6--7--8--9
    # | /| /| /| /|
    # |/ |/ |/ |/ |
    # 0--1--2--3--4
    fine_vertices = [(sympy.Rational(j, 4), sympy.Rational(i, 4))
                     for i in range(5) for j in range(5)]

    def restrict(v):
        return [v[0], v[2], v[4], v[10], v[12], v[14], v[20], v[22], v[24]]

    def interpolate(v):
        return [v[0], (v[0] + v[1]) / 2, v[1], (v[1] + v[2]) / 2, v[2],
                (v[0] + v[3]) / 2, (v[0] + v[4]) / 2, (v[1] + v[4]) / 2, (v[1] + v[5]) / 2, (v[2] + v[5]) / 2,
                v[3], (v[3] + v[4]) / 2, v[4], (v[4] + v[5]) / 2, v[5],
                (v[3] + v[6]) / 2, (v[3] + v[7]) / 2, (v[3] + v[7]) / 2, (v[4] + v[8]) / 2, (v[5] + v[8]) / 2,
                v[6], (v[6] + v[7]) / 2, v[7], (v[7] + v[8]) / 2, v[8]]

    def solve(A, b):
        return sympy.Matrix(A).inv() * sympy.Matrix(b)

    def jacobi(A, b, xk):
        dim = len(A)
        return [1 / A[i,i] * (b[i] - sum(A[i, j] * xk[j] for j in range(dim) if j != i))
                for i in range(dim)]

    A_coarse = [[0 for i in range(9)] for j in range(9)]
    b_coarse = [0 for i in range(9)]

    # Lower triangles in coarse mesh
    coarse_basis = [1 - 2 * x, 2 * x - 2 * y, 2 * y]
    A_coarse_sub = [[(i.diff(x) * j.diff(x) + i.diff(y) * j.diff(y)).integrate((x, y, half), (y, 0, half))
                     for j in coarse_basis] for i in coarse_basis]
    for v0 in [0, 1, 3, 4]:
        triangle = [v0, v0 + 1, v0 + 4]
        for i, dof_i in enumerate(triangle):
            for j, dof_j in enumerate(triangle):
                A_coarse[dof_i][dof_j] += A_coarse_sub[i][j]
            integrand = f((x + coarse_vertices[v0][0], y + coarse_vertices[v0][1])) * coarse_basis[i]
            b_coarse[dof_i] += integrand.integrate((x, y, half), (y, 0, half))
    # Upper triangles in coarse mesh
    coarse_basis = [1 - 2 * y, 2 * x, 2 * y - 2 * x]
    A_coarse_sub = [[(i.diff(x) * j.diff(x) + i.diff(y) * j.diff(y)).integrate((x, 0, y), (y, 0, half))
                     for j in coarse_basis] for i in coarse_basis]
    for v0 in [0, 1, 3, 4]:
        triangle = [v0, v0 + 4, v0 + 3]
        for i, dof_i in enumerate(triangle):
            for j, dof_j in enumerate(triangle):
                A_coarse[dof_i][dof_j] += A_coarse_sub[i][j]
            integrand = f((x + coarse_vertices[v0][0],
                           y + coarse_vertices[v0][1])) * coarse_basis[i]
            b_coarse[dof_i] += integrand.integrate((x, 0, y), (y, 0, half))

    A_fine = [[0 for i in range(25)] for j in range(25)]
    b_fine = [0 for i in range(25)]
    # Lower triangles in fine mesh
    fine_basis = [1 - 4 * x, 4 * x - 4 * y, 4 * y]
    A_fine_sub = [[(i.diff(x) * j.diff(x) + i.diff(y) * j.diff(y)).integrate((x, y, quarter), (y, 0, quarter))
                     for j in fine_basis] for i in fine_basis]
    for v0 in [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18]:
        triangle = [v0, v0 + 1, v0 + 6]
        for i, dof_i in enumerate(triangle):
            for j, dof_j in enumerate(triangle):
                A_fine[dof_i][dof_j] += A_fine_sub[i][j]
            integrand = f((x + fine_vertices[v0][0],
                           y + fine_vertices[v0][1])) * fine_basis[i]
            b_fine[dof_i] += integrand.integrate((x, y, quarter), (y, 0, quarter))
    # Upper triangles in fine mesh
    fine_basis = [1 - 4 * y, 4 * x, 4 * y - 4 * x]
    A_fine_sub = [[(i.diff(x) * j.diff(x) + i.diff(y) * j.diff(y)).integrate((x, 0, y), (y, 0, quarter))
                     for j in fine_basis] for i in fine_basis]
    for v0 in [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18]:
        triangle = [v0, v0 + 6, v0 + 5]
        for i, dof_i in enumerate(triangle):
            for j, dof_j in enumerate(triangle):
                A_fine[dof_i][dof_j] += A_fine_sub[i][j]
            integrand = f((x + fine_vertices[v0][0],
                           y + fine_vertices[v0][1])) * fine_basis[i]
            b_fine[dof_i] += integrand.integrate((x, 0, y), (y, 0, quarter))

    # Use DOLFINx
    coarse_mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 2, 2)
    coarse_space = dolfinx.FunctionSpace(coarse_mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(coarse_space)
    v = ufl.TestFunction(coarse_space)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    A_coarse_dolf = dolfinx.fem.assemble_matrix(a)
    A_coarse_dolf.assemble()

    x_coarse = ufl.SpatialCoordinate(coarse_mesh)

    b = ufl.inner(f(x_coarse), v) * ufl.dx
    b_coarse_dolf = dolfinx.fem.assemble_vector(b)

    fine_mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 4, 4)
    fine_space = dolfinx.FunctionSpace(fine_mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(fine_space)
    v = ufl.TestFunction(fine_space)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    A_fine_dolf = dolfinx.fem.assemble_matrix(a)
    A_fine_dolf.assemble()

    x_fine = ufl.SpatialCoordinate(fine_mesh)

    b = ufl.inner(f(x_fine), v) * ufl.dx
    b_fine_dolf = dolfinx.fem.assemble_vector(b)

    # Test coarse matrix and vector
    dolfin_to_sympy = []
    for i, co in enumerate(coarse_space.tabulate_dof_coordinates()):
        for j, co2 in enumerate(coarse_vertices):
            if np.allclose(co[:2], [float(k) for k in co2]):
                dolfin_to_sympy.append(j)
        assert len(dolfin_to_sympy) == i + 1

    for i0, i1 in enumerate(dolfin_to_sympy):
        assert np.isclose(float(b_coarse[i1]), b_coarse_dolf[i0])
        for j0, j1 in enumerate(dolfin_to_sympy):
            assert np.isclose(A_coarse_dolf[i0,j0], float(A_coarse[i1][j1]))

    # Test fine matrix and vector
    dolfin_to_sympy = []
    for i, co in enumerate(fine_space.tabulate_dof_coordinates()):
        for j, co2 in enumerate(fine_vertices):
            if np.allclose(co[:2], [float(k) for k in co2]):
                dolfin_to_sympy.append(j)
        assert len(dolfin_to_sympy) == i + 1

    for i0, i1 in enumerate(dolfin_to_sympy):
        assert np.isclose(float(b_fine[i1]), b_fine_dolf[i0])
        for j0, j1 in enumerate(dolfin_to_sympy):
            assert np.isclose(A_fine_dolf[i0,j0], float(A_fine[i1][j1]))

