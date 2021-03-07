import dolfinx
import numpy as np
import ufl
import math
import scipy as scp
from scipy import sparse
from scipy.sparse.linalg import spsolve
from mpi4py import MPI
from petsc4py import PETSc
import matplotlib.pyplot as plt


finest_level = 3
coarsest_level = finest_level - 2
coarsest_level_elements_per_dim = 4
coarsest_level_nodes_per_dim = coarsest_level_elements_per_dim + 1
residual_per_iteration = []
# Defining the Parameters of multigrid
mu0 = 10
mu1 = 30
mu2 = 30
omega = 2/3  # Parameter for Jacobi Smoother
residual_values_v_cycles = []

# Plotting Variables
L2_error_list = []
residual_list = []
L2_error_list_dolfinx = []

# mesh_dict = {}  # Format : {level : mesh}
mesh_fine = None
element_size = {}
mesh_dof_list_dict = {}  # Contains the mesh dictionaries type :: level : mesh_dict
# Format : {level : (A-sparse , level)} level is used in V_cycle and FMG to iterate over coarse matrices
A_dict = {}
b_dict = {}  # Format : {level : b}
A_dict_jacobi = {}  # Format {level : (R_omega , diag_A_inv , level)}
a_dolfinx = None
L_dolfinx = None
bcs_dolfinx = None
b_dolfinx = None
V_dolfinx = None


for i in range(coarsest_level, finest_level + 1):
    element_size[i] = 1 / (coarsest_level_elements_per_dim * 2**i)
    mesh = dolfinx.UnitSquareMesh(
        MPI.COMM_WORLD, coarsest_level_elements_per_dim * (2**(i)), coarsest_level_elements_per_dim * (2**(i)), dolfinx.cpp.mesh.CellType.triangle)
    #mesh_dict[i] = (mesh, i)
    V = dolfinx.FunctionSpace(mesh, ("CG", 1))
    V_dof_coord = V.tabulate_dof_coordinates()
    dof_dict = {}
    for j in range(0, V.dofmap.index_map.size_local * V.dofmap.index_map_bs):
        # dof_dict[j] = tuple([round(k, 3) for k in V_dof_coord[j]])
        # dof_dict[tuple([round(k, 3) for k in V_dof_coord[j]])] = j
        #cord_tup = tuple(cor if cor > 1E-7 else 0 for cor in V_dof_coord[j])
        cord_tup = tuple(round(cor, 9) for cor in V_dof_coord[j])
        dof_dict[j] = cord_tup
        dof_dict[cord_tup] = j
    mesh_dof_list_dict[i] = dof_dict
    del dof_dict

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
    f = dolfinx.Constant(mesh, 6)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    A = dolfinx.fem.assemble_matrix(a, bcs=[bc])
    A.assemble()
    assert isinstance(A, PETSc.Mat)
    ai, aj, av = A.getValuesCSR()
    A_sp = scp.sparse.csr_matrix((av, aj, ai))
    del A, av, ai, aj
    # Stores the Sparse version of the Stiffness Matrices
    A_dict[i] = (A_sp, i)
    #print('Level {} matrix assembled and its shape is {}'.format(i, A_sp.shape))

    if i == finest_level:                           # Storing the finest level vector in b_dict
        mesh_fine = mesh
        L = f * v * ufl.dx
        L_dolfinx = L
        a_dolfinx = a
        bcs_dolfinx = bc
        V_dolfinx = V
        b = dolfinx.fem.create_vector(L)
        with b.localForm() as loc_b:
            loc_b.set(0)
        dolfinx.fem.assemble_vector(b, L)
        dolfinx.fem.apply_lifting(b, [a], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(b, [bc])
        b_dolfinx = b
        b_dict[i] = np.array(b.array).reshape(
            (coarsest_level_elements_per_dim * 2**finest_level + 1) ** 2, 1)

# Defining the matrices for the Jacobi Smoothing
# print(b_dict[finest_level].shape)

# Takes in a Global Stiffness Matrix and gives the Final Jacobi Iteration Matrix


def getJacobiMatrices(A):
    A_mat = A[0]
    negative_upper_A = (-1) * scp.sparse.triu(A_mat, k=1)
    negative_lower_A = (-1) * scp.sparse.tril(A_mat, k=-1)
    #diag_A_inv = scp.sparse.csr_matrix(np.diag(1 / (A_mat.diagonal())))
    diag_A_inv = scp.sparse.diags((1 / A_mat.diagonal()), 0)
    # only a diagonal matrix so space can be saved here
    # diag_A_inverse = [1 / x for x in diag_A]
    Rj = diag_A_inv.dot((negative_upper_A + negative_lower_A))
    #R_omega = ((1-omega) * scp.sparse.csr_matrix(np.identity(A_mat.shape[0]))) + omega * Rj
    R_omega = (1-omega) * \
        scp.sparse.diags(np.ones(A_mat.shape[0]), 0) + omega * Rj
    del negative_lower_A, negative_upper_A,
    return (R_omega, diag_A_inv, A[1])


def jacobiRelaxation(A, v, f, nw):
    for i in range(nw):
        v = A[0].dot(v) + omega*A[1].dot(f)
    return v

# With one coarsening, the number of elements in a dimension gets reduces by a factor of 2. No of elements in any dim = no of nodes in any dim -1. No of


def Interpolation2D(vec_2h, level_coarse):
    mesh_dict_coarse = mesh_dof_list_dict[level_coarse]
    mesh_dict_fine = mesh_dof_list_dict[level_coarse + 1]
    element_size_coarse = element_size[level_coarse]
    element_size_fine = element_size[level_coarse + 1]

    # target_dof_size[0] * target_dof_size[1]
    vec_h_dim = (coarsest_level_elements_per_dim *
                 2**(level_coarse + 1) + 1) ** 2
    vec_h = np.zeros((vec_h_dim, 1))

    # Iterating over the dofs
    for i in range(0, vec_h_dim):
        fine_coord = mesh_dict_fine[i]  # Getting the finer level coordinates
        coarse_dof = mesh_dict_coarse.get(fine_coord, -1)
        if coarse_dof != -1:                # Cannot directly inject the boundary values not present in coarse grid
            # Directly injecting the coarse dof into the fine dof value
            vec_h[i] = vec_2h[coarse_dof]

        else:
            i_h = int(fine_coord[0] / element_size_fine)
            j_h = int(fine_coord[1] / element_size_fine)
            # interpolation needs to be carried out for the dof which are not the part of the coarse vector

            # check if first dimension is odd and second is even
            if i_h % 2 == 1 and j_h % 2 == 0:
                i_2h = (i_h - 1) / 2
                j_2h = j_h / 2
                coarse_dof_1 = mesh_dict_coarse[(i_2h * element_size_coarse,
                                                 j_2h * element_size_coarse, 0)]
                coarse_dof_2 = mesh_dict_coarse[((i_2h + 1) * element_size_coarse,
                                                 j_2h * element_size_coarse, 0)]
                vec_h[i] = 0.5 * (vec_2h[coarse_dof_1] +
                                  vec_2h[coarse_dof_2])

            # check if first dimension is even and second is odd
            elif i_h % 2 == 0 and j_h % 2 == 1:
                i_2h = i_h / 2
                j_2h = (j_h - 1) / 2
                coarse_dof_1 = mesh_dict_coarse[(i_2h * element_size_coarse,
                                                 j_2h * element_size_coarse, 0)]
                coarse_dof_2 = mesh_dict_coarse[(i_2h * element_size_coarse,
                                                 (j_2h + 1) * element_size_coarse, 0)]
                vec_h[i] = 0.5 * (vec_2h[coarse_dof_1] +
                                  vec_2h[coarse_dof_2])

                # Check if both are odd
            else:
                i_2h = (i_h - 1) / 2
                j_2h = (j_h - 1) / 2

                coarse_dof_1 = mesh_dict_coarse[(i_2h * element_size_coarse,
                                                 j_2h * element_size_coarse, 0)]
                coarse_dof_2 = mesh_dict_coarse[((i_2h + 1) * element_size_coarse,
                                                 j_2h * element_size_coarse, 0)]
                coarse_dof_3 = mesh_dict_coarse[(i_2h * element_size_coarse,
                                                 (j_2h + 1) * element_size_coarse, 0)]
                coarse_dof_4 = mesh_dict_coarse[((i_2h + 1) * element_size_coarse,
                                                 (j_2h + 1) * element_size_coarse, 0)]
                vec_h[i] = 0.25 * (vec_2h[coarse_dof_1] + vec_2h[coarse_dof_2] +
                                   vec_2h[coarse_dof_3] + vec_2h[coarse_dof_4])
            # TODO Boundary dof case  locate_entities_boundary

    # Return the interpolated vector
    return vec_h


def Restriction2D(vec_h, level_fine):
    mesh_dict_coarse = mesh_dof_list_dict[level_fine - 1]
    mesh_dict_fine = mesh_dof_list_dict[level_fine]
    element_size_coarse = element_size[level_fine - 1]
    element_size_fine = element_size[level_fine]

    vec_2h_dim = (coarsest_level_elements_per_dim *
                  2**(level_fine - 1) + 1) ** 2
    vec_2h = np.zeros((vec_2h_dim, 1))

    # Iterating over the dofs
    for i in range(0, vec_2h_dim):
        coarse_coord = mesh_dict_coarse[i]
        #fine_dof = mesh_dict_fine.get(coarse_coord, -1)
        # if the dof is a boundary dof, inject it directly into the coarser grid

        # else
        i_2h = coarse_coord[0] / element_size_coarse
        j_2h = coarse_coord[1] / element_size_coarse

        fine_dof_1 = mesh_dict_fine.get(
            ((2 * i_2h - 1) * element_size_fine, (2 * j_2h - 1) * element_size_fine, 0), -1)
        fine_dof_2 = mesh_dict_fine.get(
            ((2 * i_2h - 1) * element_size_fine, (2 * j_2h + 1) * element_size_fine, 0), -1)
        fine_dof_3 = mesh_dict_fine.get(
            ((2 * i_2h + 1) * element_size_fine, (2 * j_2h - 1) * element_size_fine, 0), -1)
        fine_dof_4 = mesh_dict_fine.get(
            ((2 * i_2h + 1) * element_size_fine, (2 * j_2h + 1) * element_size_fine, 0), -1)
        fine_dof_5 = mesh_dict_fine.get(
            (2 * i_2h * element_size_fine, (2 * j_2h - 1) * element_size_fine, 0), -1)
        fine_dof_6 = mesh_dict_fine.get(
            (2 * i_2h * element_size_fine, (2 * j_2h + 1) * element_size_fine, 0), -1)
        fine_dof_7 = mesh_dict_fine.get(
            ((2 * i_2h - 1) * element_size_fine, 2 * j_2h * element_size_fine, 0), -1)
        fine_dof_8 = mesh_dict_fine.get(
            ((2 * i_2h + 1) * element_size_fine, 2 * j_2h * element_size_fine, 0), -1)
        fine_dof_9 = mesh_dict_fine.get(
            (2 * i_2h * element_size_fine, 2 * j_2h * element_size_fine, 0), -1)
        sum_1 = 0
        sum_2 = 0

        if fine_dof_1 != -1:
            sum_1 += vec_h[fine_dof_1]

        if fine_dof_2 != -1:
            sum_1 += vec_h[fine_dof_2]

        if fine_dof_3 != -1:
            sum_1 += vec_h[fine_dof_3]

        if fine_dof_4 != -1:
            sum_1 += vec_h[fine_dof_4]

        if fine_dof_5 != -1:
            sum_2 += vec_h[fine_dof_5]

        if fine_dof_6 != -1:
            sum_2 += vec_h[fine_dof_6]

        if fine_dof_7 != -1:
            sum_2 += vec_h[fine_dof_7]

        if fine_dof_8 != -1:
            sum_2 += vec_h[fine_dof_8]

        vec_2h[i] = (1/16) * (sum_1 + 2 * sum_2 + 4 * vec_h[fine_dof_9])

    return vec_2h


def V_cycle_scheme(A_h, v_h, f_h):
    #v_h = jacobiRelaxation(A_h, v_h, f_h, mu1)
    # Check if the space is the coarsest
    if(A_h[2] == coarsest_level):
        v_h = spsolve(A_dict[coarsest_level][0], f_h)
        #v_h = jacobiRelaxation(A_h, v_h, f_h, mu2)
        return v_h
    else:
        v_h = jacobiRelaxation(A_h, v_h, f_h, mu1)
        # f_2h = Restriction2D((f_h - np.matmul(A_dict[A_h[2]], v_h)), A_h[2])
        f_2h = Restriction2D((f_h - A_dict[A_h[2]][0].dot(v_h)), A_h[2])
        v_2h = np.zeros((f_2h.shape[0], 1))
        # Fetch the Smaller discretication matrix
        A_2h = A_dict_jacobi[A_h[2] - 1]
        v_2h = V_cycle_scheme(A_2h, v_2h, f_2h)
    v_h = v_h + Interpolation2D(v_2h, A_h[2] - 1)
    # v_h[v_h.shape[0] - 1] += 0.5
    v_h = jacobiRelaxation(A_h, v_h, f_h, mu2)
    return v_h


def FullMultiGrid(A_h, f_h, iter_V):
    # Check if the space is the coarsest
    if(A_h[2] == coarsest_level):
        """v_h = np.zeros((f_h.shape[0], 1))
        for i in range(mu0):
            v_h = V_cycle_scheme(A_h, v_h, f_h)"""
        v_h = spsolve(A_dict[coarsest_level][0], f_h)
        return v_h
    else:
        f_2h = Restriction2D(f_h, A_h[2])
        # Get the next coarse level of Discretization Matrix
        A_2h = A_dict_jacobi[A_h[2] - 1]
        v_2h = FullMultiGrid(A_2h, f_2h, iter_V)
    v_h = Interpolation2D(v_2h, A_h[2] - 1)
    # v_h[v_h.shape[0] - 1] += 0.5
    for i in range(iter_V):
        v_h = V_cycle_scheme(A_h, v_h, f_h)
    return v_h

# Assemble the matrices in for Jacobi in A_dict_jacobi


for key, value in A_dict.items():
    A_dict_jacobi[key] = getJacobiMatrices(value)

# Computing the dolfinx solution
problem = dolfinx.fem.LinearProblem(a_dolfinx, L_dolfinx, bcs=[bcs_dolfinx], petsc_options={
                                    "ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
V2 = dolfinx.FunctionSpace(mesh_fine, ("CG", 2))
u_exact = dolfinx.Function(V2)
u_exact.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
u_exact.vector.ghostUpdate(
    addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
u_exact_vertex_values = u_exact.compute_point_values()
L2_error_dolfinx = ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx
error_L2_dolfinx = np.sqrt(dolfinx.fem.assemble_scalar(L2_error_dolfinx))

# Computing the FMG solution over range of V-cycles for fixed pre and post smooths
for iter_V in range(1, mu0):
    u = FullMultiGrid(A_dict_jacobi[finest_level],
                      b_dict[finest_level], iter_V)
    u_FMG = dolfinx.Function(V_dolfinx)
    u_FMG.vector[:] = u
    residual = b_dict[finest_level] - A_dict[finest_level][0].dot(u)
    residual_list.append(np.linalg.norm(residual))
    L2_error_FMG = ufl.inner(u_FMG - u_exact, u_FMG - u_exact) * ufl.dx
    error_L2_FMG = np.sqrt(dolfinx.fem.assemble_scalar(L2_error_FMG))
    L2_error_list.append(error_L2_FMG)
    L2_error_list_dolfinx.append(error_L2_dolfinx)

x_axis = np.arange(1, mu0)
plt.figure(1)
axis1 = plt.axes()
axis1.grid(True)
axis1.semilogy(x_axis, residual_list, linewidth=0.8, color='k')
plt.title("FMG Residual v/s no. of V-cycles")
plt.ylabel("Residual")
text_str = f'Smoother = Jacobi\nPre-smooths = {mu1}\nPost-Smooths = {mu2}\nFinest Level = {coarsest_level_elements_per_dim * 2**finest_level}\nV-cycle levels = {finest_level - coarsest_level + 1}'
plt.xlabel("V-cycles")
plt.figtext(0.7, 0.75, text_str, fontsize=8, ha='left')
plt.savefig(
    f'Residual {coarsest_level_elements_per_dim * 2**finest_level} elements.png')

plt.figure(2, figsize=(9, 5))
axis2 = plt.axes()
axis2.grid(True)
axis2.semilogy(x_axis, L2_error_list, linewidth=0.8, color='k', label="FMG")
axis2.semilogy(x_axis, L2_error_list_dolfinx, linewidth=0.5,
               color='g', label="Std. Solver")
plt.legend()
plt.title("L2 error v/s no. of V-cycles")
plt.ylabel("L2 Error")
text_str = f'Smoother = Jacobi\nPre-smooths = {mu1}\nPost-Smooths = {mu2}\nFinest Level = {coarsest_level_elements_per_dim * 2**finest_level}\nV-cycle levels = {finest_level - coarsest_level + 1}'
plt.xlabel("V-cycles")
plt.figtext(0.75, 0.6, text_str, fontsize=8, ha='left')
plt.savefig(
    f'L2-Error {coarsest_level_elements_per_dim * 2**finest_level} elements.png')
