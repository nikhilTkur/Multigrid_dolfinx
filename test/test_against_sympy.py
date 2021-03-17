import dolfinx
import ufl
import sympy
import scipy
from mpi4py import MPI
import numpy as np
from multigrid import getJacobiMatrices, FullMultiGrid, writing_error_for_mesh_to_csv, writing_residual_for_mesh_to_csv, initialize_problem


x = sympy.Symbol("x")
y = sympy.Symbol("y")
zero = sympy.Integer(0)
one = sympy.Integer(1)
half = sympy.Rational(1, 2)
quarter = sympy.Rational(1, 4)


# A = grad(u) * grad(v) * dx
# b = f * v * dx
# Dirichlet condition 0
def f(x):
    return -2 * x[0] * (1 - x[0]) - 2 * x[1] * (1 - x[1])


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
coarse_boundary = [0, 1, 2, 3, 5, 6, 7, 8]

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
fine_boundary = [0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24]


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
    return [1 / A[i][i] * (b[i] - sum(A[i][j] * xk[j] for j in range(dim) if j != i))
            for i in range(dim)]


def make_sympy_coarse_Ab():
    # COARSE
    A = [[0 for i in range(9)] for j in range(9)]
    b = [0 for i in range(9)]

    # Lower triangles in coarse mesh
    basis = [1 - 2 * x, 2 * x - 2 * y, 2 * y]
    A_sub = [[(i.diff(x) * j.diff(x) + i.diff(y) * j.diff(y)).integrate((x, y, half), (y, 0, half))
                     for j in basis] for i in basis]
    for v0 in [0, 1, 3, 4]:
        triangle = [v0, v0 + 1, v0 + 4]
        for i, dof_i in enumerate(triangle):
            for j, dof_j in enumerate(triangle):
                A[dof_i][dof_j] += A_sub[i][j]
            integrand = f((x + coarse_vertices[v0][0], y + coarse_vertices[v0][1])) * basis[i]
            b[dof_i] += integrand.integrate((x, y, half), (y, 0, half))

    # Upper triangles in coarse mesh
    basis = [1 - 2 * y, 2 * x, 2 * y - 2 * x]
    A_sub = [[(i.diff(x) * j.diff(x) + i.diff(y) * j.diff(y)).integrate((x, 0, y), (y, 0, half))
                     for j in basis] for i in basis]
    for v0 in [0, 1, 3, 4]:
        triangle = [v0, v0 + 4, v0 + 3]
        for i, dof_i in enumerate(triangle):
            for j, dof_j in enumerate(triangle):
                A[dof_i][dof_j] += A_sub[i][j]
            integrand = f((x + coarse_vertices[v0][0],
                           y + coarse_vertices[v0][1])) * basis[i]
            b[dof_i] += integrand.integrate((x, 0, y), (y, 0, half))

    # apply boundary conditions
    for i in coarse_boundary:
        for j in range(9):
            A[i][j] = 0
            A[j][i] = 0
        A[i][i] = one
        b[i] = zero
    return A, b


def make_sympy_fine_Ab():
    # FINE
    A = [[0 for i in range(25)] for j in range(25)]
    b = [0 for i in range(25)]
    # Lower triangles in fine mesh
    basis = [1 - 4 * x, 4 * x - 4 * y, 4 * y]
    A_sub = [[(i.diff(x) * j.diff(x) + i.diff(y) * j.diff(y)).integrate((x, y, quarter), (y, 0, quarter))
                     for j in basis] for i in basis]
    for v0 in [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18]:
        triangle = [v0, v0 + 1, v0 + 6]
        for i, dof_i in enumerate(triangle):
            for j, dof_j in enumerate(triangle):
                A[dof_i][dof_j] += A_sub[i][j]
            integrand = f((x + fine_vertices[v0][0],
                           y + fine_vertices[v0][1])) * basis[i]
            b[dof_i] += integrand.integrate((x, y, quarter), (y, 0, quarter))
    # Upper triangles in fine mesh
    basis = [1 - 4 * y, 4 * x, 4 * y - 4 * x]
    A_sub = [[(i.diff(x) * j.diff(x) + i.diff(y) * j.diff(y)).integrate((x, 0, y), (y, 0, quarter))
                     for j in basis] for i in basis]
    for v0 in [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18]:
        triangle = [v0, v0 + 6, v0 + 5]
        for i, dof_i in enumerate(triangle):
            for j, dof_j in enumerate(triangle):
                A[dof_i][dof_j] += A_sub[i][j]
            integrand = f((x + fine_vertices[v0][0],
                           y + fine_vertices[v0][1])) * basis[i]
            b[dof_i] += integrand.integrate((x, 0, y), (y, 0, quarter))
    # apply boundary conditions
    for i in fine_boundary:
        for j in range(25):
            A[i][j] = 0
            A[j][i] = 0
        A[i][i] = one
        b[i] = zero
    return A, b


def make_fenics_Ab(n):
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, n, n)
    space = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(space)
    v = ufl.TestFunction(space)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    mesh.topology.create_connectivity(1, 2)
    boundary_facets = np.where(np.array(
        dolfinx.cpp.mesh.compute_boundary_facets(mesh.topology)) == 1)[0]
    boundary_dofs = dolfinx.fem.locate_dofs_topological(
        space, 1, boundary_facets)
    uD = dolfinx.Function(space)
    uD.interpolate(lambda x: 0)
    bc = dolfinx.DirichletBC(uD, boundary_dofs)
    A_dolf = dolfinx.fem.assemble_matrix(a, bcs=[bc])
    A_dolf.assemble()

    x = ufl.SpatialCoordinate(mesh)
    b = ufl.inner(f(x), v) * ufl.dx
    b_dolf = dolfinx.fem.assemble_vector(b)
    dolfinx.fem.apply_lifting(b_dolf, [a], [[bc]])
    dolfinx.fem.set_bc(b_dolf, [bc])

    return space, A_dolf, b_dolf


def make_fenics_coarse_Ab():
    return make_fenics_Ab(2)


def make_fenics_fine_Ab():
    return make_fenics_Ab(4)


def test_coarse_Ab_against_sympy():
    A, b = make_sympy_coarse_Ab()
    space, A_dolf, b_dolf = make_fenics_coarse_Ab()

    # Test coarse matrix and vector
    dolfin_to_sympy = []
    for i, co in enumerate(space.tabulate_dof_coordinates()):
        for j, co2 in enumerate(coarse_vertices):
            if np.allclose(co[:2], [float(k) for k in co2]):
                dolfin_to_sympy.append(j)
        assert len(dolfin_to_sympy) == i + 1

    for i0, i1 in enumerate(dolfin_to_sympy):
        assert np.isclose(float(b[i1]), b_dolf[i0])
        for j0, j1 in enumerate(dolfin_to_sympy):
            assert np.isclose(A_dolf[i0, j0], float(A[i1][j1]))


def test_fine_Ab_against_sympy():
    A, b = make_sympy_fine_Ab()
    space, A_dolf, b_dolf = make_fenics_fine_Ab()

    # Test fine matrix and vector
    dolfin_to_sympy = []
    for i, co in enumerate(space.tabulate_dof_coordinates()):
        for j, co2 in enumerate(fine_vertices):
            if np.allclose(co[:2], [float(k) for k in co2]):
                dolfin_to_sympy.append(j)
        assert len(dolfin_to_sympy) == i + 1

    for i0, i1 in enumerate(dolfin_to_sympy):
        assert np.isclose(float(b[i1]), b_dolf[i0])
        for j0, j1 in enumerate(dolfin_to_sympy):
            assert np.isclose(A_dolf[i0, j0], float(A[i1][j1]))


def run_sympy_multigrid():
    A_coarse, b_coarse = make_sympy_coarse_Ab()
    A_fine, b_fine = make_sympy_fine_Ab()

    initial_guess = [0 for i in range(25)]
    iterations = [initial_guess]
    solution_fine = initial_guess

    for N in range(10):
        for i in range(3):
            solution_fine = jacobi(A_fine, b_fine, solution_fine)
        residual_fine = sympy.Matrix(A_fine) * sympy.Matrix(solution_fine) - sympy.Matrix(b_fine)
        residual_coarse = restrict(residual_fine)
        solution_coarse = solve(A_coarse, residual_coarse)
        correction_fine = interpolate(solution_coarse)
        solution_fine = [i + j for i, j in zip(correction_fine, solution_fine)]
        for i in range(3):
            solution_fine = jacobi(A_fine, b_fine, solution_fine)
        iterations.append(solution_fine)

    return iterations


def run_fenics_multigrid():
    class Parameters:
        def __init__(self):
            self.mesh_dof_list_dict = None
            self.element_size = {1: 0.5, 2: 0.25}
            self.coarsest_level_elements_per_dim = 1
            self.coarsest_level = 1
            self.finest_level = 2
            self.A_sp_dict = None
            self.A_jacobi_sp_dict = None
            self.b_dict = None
            self.mu0 = 1
            self.mu1 = 3
            self.mu2 = 3
            self.omega = 1
            self.residual_per_V_cycle_finest = []
            self.error_per_V_cycle_finest = []
            self.u_exact_fine = None
            self.V_fine_dolfx = None

    space_coarse, A_coarse, b_coarse = make_fenics_coarse_Ab()
    space_fine, A_fine, b_fine = make_fenics_fine_Ab()

    x_fine = ufl.SpatialCoordinate(space_fine.mesh)

    p = Parameters()
    p.mesh_dof_list_dict = {}
    # COARSE
    dof_dict = {}
    dof_coords = space_coarse.tabulate_dof_coordinates()
    for j in range(0, space_coarse.dofmap.index_map.size_local * space_coarse.dofmap.index_map_bs):
        cord_tup = tuple(round(cor, 9) for cor in dof_coords[j])
        dof_dict[j] = cord_tup
        dof_dict[cord_tup] = j
    p.mesh_dof_list_dict[1] = dof_dict
    # FINE
    dof_dict = {}
    dof_coords = space_fine.tabulate_dof_coordinates()
    for j in range(0, space_fine.dofmap.index_map.size_local * space_fine.dofmap.index_map_bs):
        cord_tup = tuple(round(cor, 9) for cor in dof_coords[j])
        dof_dict[j] = cord_tup
        dof_dict[cord_tup] = j
    p.mesh_dof_list_dict[2] = dof_dict

    p.u_exact_fine = x_fine[0] * (1 - x_fine[0]) * x_fine[1] * (1 - x_fine[1])
    p.V_fine_dolfx = space_fine
    p.A_sp_dict = {1: (scipy.sparse.csr_matrix(A_coarse.getValuesCSR()[::-1]), 1),
                   2: (scipy.sparse.csr_matrix(A_fine.getValuesCSR()[::-1]), 2)}
    p.b_dict = {1: np.array(b_coarse.array).reshape(9, 1),
                2: np.array(b_fine.array).reshape(25, 1)}
    p.A_jacobi_sp_dict = {key: getJacobiMatrices(value) for key, value in p.A_sp_dict.items()}

    initialize_problem(p)
    u_FMG = FullMultiGrid(p.A_jacobi_sp_dict[2], p.b_dict[2])

    return space_fine, [0*u_FMG, u_FMG]


def test_multigrid_run():
    space_fine, fenics_data = run_fenics_multigrid()
    sympy_data = run_sympy_multigrid()

    # Test fine matrix and vector
    dolfin_to_sympy = []
    for i, co in enumerate(space_fine.tabulate_dof_coordinates()):
        for j, co2 in enumerate(fine_vertices):
            if np.allclose(co[:2], [float(k) for k in co2]):
                dolfin_to_sympy.append(j)
        assert len(dolfin_to_sympy) == i + 1

    for sympy_v, fenics_v in zip(sympy_data[-1:], fenics_data[-1:]):
        print(fenics_v.T[0])
        print([float(sympy_v[j]) for j in dolfin_to_sympy])
        for i, j in enumerate(dolfin_to_sympy):
            assert np.isclose(float(sympy_v[j]), fenics_v[i, 0])
