import dolfinx
import numpy as np
import ufl
import math
import scipy as scp
from mpi4py import MPI
from petsc4py import PETSc


finest_level = 3
coarsest_level = 0
coarsest_level_elements_per_dim = 8
coarsest_level_nodes_per_dim = coarsest_level_elements_per_dim + 1

# Defining the Parameters of multigrid
mu0 = 10
mu1 = 10
mu2 = 10
omega = float(2/3)  # Parameter for Jacobi Smoother


# Defining the matrices of different levels as dictionaries

mesh_dict = {}  # Format : {level : mesh}
# Format : {level : (A , level)} Second level is used in V_cycle and FMG to iterate over coarse matrices
A_dict = {}
b_dict = {}  # Format : {level : b}
A_dict_jacobi = {}  # Format {level : (R_omega , diag_A_inv , level)}

A_jacobi_store = {}

for i in range(coarsest_level, finest_level + 1):
    mesh = dolfinx.UnitSquareMesh(
        MPI.COMM_WORLD, coarsest_level_elements_per_dim * math.pow(2, (i - coarsest_level)), coarsest_level_elements_per_dim * math.pow(2, (i - coarsest_level)), dolfinx.cpp.mesh.CellType.triangle)
    mesh_dict[i] = (mesh, i)
    V = dolfinx.FunctionSpace(mesh, ("CG", 1))
    uD = dolfinx.Function(V)
    uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
    uD.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)
    fdim = mesh.topology.dim-1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    boundary_facets = np.where(np.array(
        dolfinx.cpp.mesh.compute_boundary_facets(mesh.topology)) == 1)[0]
    boundary_dofs = dolfinx.fem.locate_dofs_topological(
        V, fdim, boundary_facets)
    bc = dolfinx.DirichletBC(uD, boundary_dofs)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = dolfinx.Constant(mesh, -6)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    A = dolfinx.fem.assemble_matrix(a, bcs=[bc])
    A.assemble()
    assert isinstance(A, PETSc.Mat)
    ai, aj, av = A.getValuesCSR()
    A_sp = scp.sparse.csr_matrix((av, aj, ai))
    # Stores the Sparse version of the Stiffness Matrices
    A_dict[i] = (A_sp, i)

    if i == finest_level:                           # Storing the finest level vector in b_dict
        L = f * v * ufl.dx
        b = dolfinx.fem.create_vector(L)
        with b.localForm() as loc_b:
            loc_b.set(0)
        dolfinx.fem.assemble_vector(b, L)
        dolfinx.fem.apply_lifting(b, [a], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(b, [bc])
        b_dict[i] = b

# Defining the matrices for the Jacobi Smoothing


# Takes in a Global Stiffness Matrix and gives the Final Jacobi Iteration Matrix
def getJacobiMatrices(A):
    negative_upper_A = (-1) * scp.sparse.triu(A[0], k=1)
    negative_lower_A = (-1) * scp.sparse.tril(A[0], k=-1)
    diag_A_inv = scp.sparse.csr_matrix(
        np.diag(1 / (A[0].diagonal().reshape(A[0].shape[0], 1))))
    # only a diagonal matrix so space can be saved here
    # diag_A_inverse = [1 / x for x in diag_A]
    Rj = diag_A_inv.dot((negative_upper_A + negative_lower_A))
    R_omega = (
        (1-omega) * scp.sparse.csr_matrix(np.identity(A[0].shape[0]))) + omega * Rj
    return (R_omega, diag_A_inv, A[1])


def jacobiRelaxation(A, v, f, nw):
    for i in range(nw):
        v =  A[0].dot(v) + omega*A[1].dot(f)
    return v


# Next Step : Defining the V-Grid Cycles and FMG Cycles
def V_cycle_scheme(A_h, v_h, f_h):
    v_h = jacobiRelaxation(A_h[0], v_h, f_h, mu1)
    # Check if the space is the coarsest
    if(A_h[2] == coarsest_level):
        v_h = jacobiRelaxation(A_h, v_h, f_h, mu2)
        return v_h
    else:
        f_2h = Restriction2D((f_h - np.matmul(A_h, v_h)))
        v_2h = np.zeros((f_2h.shape[0], 1))

        # Fetch the Smaller discretication matrix
        A_2h = A_dict_jacobi[A_h[2] - 1]
        v_2h = V_cycle_scheme(A_2h[0], v_2h, f_2h)
    v_h = v_h + Interpolation2D(v_2h)
    # v_h[v_h.shape[0] - 1] += 0.5
    v_h = jacobiRelaxation(A_h, v_h, f_h, mu2)
    return v_h


def FullMultiGrid(A_h, f_h):
    # Check if the space is the coarsest
    if(A_h[2] == coarsest_level):
        v_h = np.zeros((f_h.shape[0], 1))
        for i in range(mu0):
            v_h = V_cycle_scheme(A_h, v_h, f_h)
        return v_h
    else:
        f_2h = Restriction2D(f_h)
        # Get the next coarse level of Discretization Matrix
        A_2h = A_dict_jacobi[A_h[2] - 1]
        v_2h = FullMultiGrid(A_2h, f_2h)
    v_h = Interpolation2D(v_2h)
    # v_h[v_h.shape[0] - 1] += 0.5
    for i in range(mu0):
        v_h = V_cycle_scheme(A_h, v_h, f_h)
    return v_h

 # Assemble the matrices in for Jacobi in A_dict_jacobi

for key, value in A_dict:
    A_dict_jacobi[key] = getJacobiMatrices(value)
