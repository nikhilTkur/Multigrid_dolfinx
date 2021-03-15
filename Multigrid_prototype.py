import dolfinx
import numpy as np
import ufl
import math
import scipy as scp
from scipy import sparse
from scipy.sparse.linalg import spsolve
from multigrid import getJacobiMatrices, FullMultiGrid, writing_error_for_mesh_to_csv, writing_residual_for_mesh_to_csv, initialize_problem
from mpi4py import MPI
from petsc4py import PETSc
import matplotlib.pyplot as plt
import csv


class Var_initializer:
    def __init__(self, mesh_dof_list_dict, element_size, coarsest_level_elements_per_dim, coarsest_level, finest_level, A_sp_dict, A_jacobi_sp_dict, b_dict, mu0, mu1, mu2, omega, residual_per_V_cycle_finest, error_per_V_cycle_finest, u_exact_fine, V_fine_dolfx):
        self.mesh_dof_list_dict = mesh_dof_list_dict
        self.element_size = element_size
        self.coarsest_level_elements_per_dim = coarsest_level_elements_per_dim
        self.coarsest_level = coarsest_level
        self.finest_level = finest_level
        self.A_sp_dict = A_sp_dict
        self.A_jacobi_sp_dict = A_jacobi_sp_dict
        self.b_dict = b_dict
        self.mu0 = mu0
        self.mu1 = mu1
        self.mu2 = mu2
        self.omega = omega
        self.residual_per_V_cycle_finest = residual_per_V_cycle_finest
        self.error_per_V_cycle_finest = error_per_V_cycle_finest
        self.u_exact_fine = u_exact_fine
        self.V_fine_dolfx = V_fine_dolfx


finest_level = 3
coarsest_level = finest_level - 1  # For 3 Level V-Cycle
coarsest_level_elements_per_dim = 8
residual_per_V_cycle_finest = []
error_per_V_cycle_finest = []

# Defining the Parameters of multigrid
mu0 = 2  # No of V-Cycles / Level
mu1 = 2  # Pre Relax Counts
mu2 = 2  # Post Relax Counts

omega = 2/3  # Parameter for Jacobi Smoother
mesh_fine = None
mesh_dof_list_dict = {}  # Stores dof and coord for all levels
element_size = {}  # Stores basic elmt size for all levels
A_sp_dict = {}  # Stores A mat sparse for all levels
b_dict = {}  # Stores the RHS of Au=b for all levels
A_jacobi_sp_dict = {}  # Stores the matrices for Jacobi smoother for all levels

# Defining the operators for finest level solution using Dolfinx
a_fine_dolfx = None  # Stores the bilinear operator for finest level
L_fine_dolfx = None  # Stores the linear operator for finest level
bcs_fine_dolfinx = None  # Stores the BCs for finest level
b_fine_dolfx = None  # Stores the b object for finest level
V_fine_dolfx = None  # Stores the CG1 Function space for finest level

# Assembling the dictionaries and operators defined above using Dolfinx Functions
for i in range(coarsest_level, finest_level + 1):
    num_elems_i = coarsest_level_elements_per_dim * 2**i
    element_size[i] = 1 / num_elems_i
    mesh_i = dolfinx.UnitSquareMesh(
        MPI.COMM_WORLD, num_elems_i, num_elems_i, dolfinx.cpp.mesh.CellType.triangle)
    V_i = dolfinx.FunctionSpace(mesh_i, ("CG", 1))
    V_dof_coord_i = V_i.tabulate_dof_coordinates()
    dof_dict_i = {}
    for j in range(0, V_i.dofmap.index_map.size_local * V_i.dofmap.index_map_bs):
        cord_tup = tuple(round(cor, 9) for cor in V_dof_coord_i[j])
        dof_dict_i[j] = cord_tup
        dof_dict_i[cord_tup] = j
    mesh_dof_list_dict[i] = dof_dict_i  # Stores the dofs at a level in dict
    del dof_dict_i

    uD_i = dolfinx.Function(V_i)
    uD_i.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
    uD_i.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                            mode=PETSc.ScatterMode.FORWARD)
    fdim_i = mesh_i.topology.dim-1
    mesh_i.topology.create_connectivity(fdim_i, mesh_i.topology.dim)
    boundary_facets_i = np.where(np.array(
        dolfinx.cpp.mesh.compute_boundary_facets(mesh_i.topology)) == 1)[0]
    boundary_dofs_i = dolfinx.fem.locate_dofs_topological(
        V_i, fdim_i, boundary_facets_i)
    bc_i = dolfinx.DirichletBC(uD_i, boundary_dofs_i)
    u_i = ufl.TrialFunction(V_i)
    v_i = ufl.TestFunction(V_i)
    f_i = dolfinx.Constant(mesh_i, 0)
    a_i = ufl.dot(ufl.grad(u_i), ufl.grad(v_i)) * ufl.dx
    A_i = dolfinx.fem.assemble_matrix(a_i, bcs=[bc_i])
    A_i.assemble()
    assert isinstance(A_i, PETSc.Mat)
    ai, aj, av = A_i.getValuesCSR()
    A_sp_i = scp.sparse.csr_matrix((av, aj, ai))
    del A_i, av, ai, aj
    # Stores the Sparse version of the Stiffness Matrices
    A_sp_dict[i] = (A_sp_i, i)
    L_i = f_i * v_i * ufl.dx
    b_i = dolfinx.fem.create_vector(L_i)
    with b_i.localForm() as loc_b:
        loc_b.set(0)
    dolfinx.fem.assemble_vector(b_i, L_i)
    dolfinx.fem.apply_lifting(b_i, [a_i], [[bc_i]])
    b_i.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                    mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b_i, [bc_i])

    b_dict[i] = np.array(b_i.array).reshape((num_elems_i + 1) ** 2, 1)

    if i == finest_level:                           # Storing the finest level vector in b_dict
        mesh_fine = mesh_i
        L_fine_dolfx = L_i
        a_fine_dolfx = a_i
        bcs_fine_dolfinx = bc_i
        V_fine_dolfx = V_i
        b_fine_dolfx = b_i

# Solving the Dolfinx CG1 solution
problem_dolfx_CG1 = dolfinx.fem.LinearProblem(a_fine_dolfx, L_fine_dolfx, bcs=[bcs_fine_dolfinx], petsc_options={
    "ksp_type": "preonly", "pc_type": "lu"})
uh_dolfx_CG1 = problem_dolfx_CG1.solve()

# Generating the exact CG2 solution
V2_fine = dolfinx.FunctionSpace(mesh_fine, ("CG", 2))
u_exact_fine = dolfinx.Function(V2_fine)
u_exact_fine.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
u_exact_fine.vector.ghostUpdate(
    addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
L2_error_dolfx = ufl.inner(uh_dolfx_CG1 - u_exact_fine,
                           uh_dolfx_CG1 - u_exact_fine) * ufl.dx
error_L2_dolfx_norm = np.sqrt(dolfinx.fem.assemble_scalar(L2_error_dolfx))

for key, value in A_sp_dict.items():
    A_jacobi_sp_dict[key] = getJacobiMatrices(value)

parameter = Var_initializer(mesh_dof_list_dict, element_size, coarsest_level_elements_per_dim, coarsest_level, finest_level, A_sp_dict,
                            A_jacobi_sp_dict, b_dict, mu0, mu1, mu2, omega, residual_per_V_cycle_finest, error_per_V_cycle_finest, u_exact_fine, V_fine_dolfx)
initialize_problem(parameter)

u_FMG = FullMultiGrid(A_jacobi_sp_dict[finest_level], b_dict[finest_level])
writing_residual_for_mesh_to_csv(residual_per_V_cycle_finest)
writing_error_for_mesh_to_csv(error_per_V_cycle_finest)

# Writing the final_error of CG1 dolfinx for comparison
with open(f'error_for_{coarsest_level_elements_per_dim * 2**finest_level}_{finest_level-coarsest_level +1}_levels.csv', mode='a') as file:
    error_writer_dolfx = csv.writer(file, delimiter=',')
    error_writer_dolfx.writerow(['Dolf', error_L2_dolfx_norm])
#
