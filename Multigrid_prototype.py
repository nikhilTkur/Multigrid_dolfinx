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
import csv


finest_level = 3
coarsest_level = finest_level - 2  # For 3 Level V-Cycle
coarsest_level_elements_per_dim = 32
residual_per_V_cycle_finest = []
error_per_V_cycle_finest = []

# Defining the Parameters of multigrid
mu0 = 15  # No of V-Cycles / Level
mu1 = 5  # Pre Relax Counts
mu2 = 5  # Post Relax Counts

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
    f_i = dolfinx.Constant(mesh_i, -6)
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


def getJacobiMatrices(A):
    # Takes in a Global Stiffness Matrix and gives the Final Jacobi Iteration Matrices
    A_mat = A[0]
    A_mat_diag = A_mat.diagonal()
    R_mat = A_mat - scp.sparse.diags(A_mat_diag, 0)
    A_mat_dig_inv = 1 / A_mat_diag
    diag_A_inv = scp.sparse.diags(A_mat_dig_inv, 0)
    R_omega_mat = diag_A_inv.dot(R_mat)
    return (R_omega_mat, diag_A_inv, A[1])


def jacobiRelaxation(A, v, f, nw):
    sol = v
    for k in range(0, nw):
        sol = (1-omega) * v + omega * A[1].dot(f) - omega*A[0].dot(v)
        v = sol
    return v


def Interpolation2D(vec_2h, level_coarse):
    mesh_dict_coarse = mesh_dof_list_dict[level_coarse]
    mesh_dict_fine = mesh_dof_list_dict[level_coarse + 1]
    element_size_coarse = element_size[level_coarse]
    element_size_fine = element_size[level_coarse + 1]

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

# Function to calculate the residual


def res_calculator(res, V_space):
    res_v_cyc = dolfinx.Function(V_space)
    res_v_cyc.vector[:] = res
    L2_res_v_cyc = ufl.inner(res_v_cyc, res_v_cyc)*ufl.dx
    res_L2_v_cyc = np.sqrt(dolfinx.fem.assemble_scalar(L2_res_v_cyc))
    return res_L2_v_cyc

# Function to calculate the error L2 norm


def err_calculator(u, u_exact, V_space):
    u_V_cyc = dolfinx.Function(V_space)
    u_V_cyc.vector[:] = u
    L2_error = ufl.inner(u_V_cyc - u_exact, u_V_cyc - u_exact) * ufl.dx
    error_L2_norm = np.sqrt(dolfinx.fem.assemble_scalar(L2_error))
    return error_L2_norm


#  fn for Writing the residual for a particular num_elements for finest level into a csv file


def writing_residual_for_mesh_to_csv(residual):
    with open(f'residual_for_{coarsest_level_elements_per_dim * 2**finest_level}.csv', mode='w') as file:
        residual_writer = csv.writer(file, delimiter=',')
        for i in range(0, len(residual)):
            residual_writer.writerow([i, residual[i]])


def writing_error_for_mesh_to_csv(error):
    with open(f'error_for_{coarsest_level_elements_per_dim * 2**finest_level}.csv', mode='w') as file:
        error_writer = csv.writer(file, delimiter=',')
        for i in range(0, len(error)):
            error_writer.writerow([i, error[i]])


def V_cycle_scheme(A_h, v_h, f_h):
    # Check if the space is the coarsest and solve exactly
    if(A_h[2] == coarsest_level):
        u_h = spsolve(A_sp_dict[coarsest_level][0], f_h)
        u_h_vec = np.array(u_h).reshape(len(u_h), 1)
        return u_h_vec
    else:
        v_h = jacobiRelaxation(A_h, v_h, f_h, mu1)
        r_h = f_h - A_sp_dict[A_h[2]][0].dot(v_h)
        f_2h = Restriction2D(r_h, A_h[2])
        v_2h = np.zeros((f_2h.shape[0], 1))
        # Fetch the Smaller discretication matrix
        A_2h = A_jacobi_sp_dict[A_h[2] - 1]
        v_2h = V_cycle_scheme(A_2h, v_2h, f_2h)
    v_h = v_h + Interpolation2D(v_2h, A_h[2] - 1)
    v_h = jacobiRelaxation(A_h, v_h, f_h, mu2)
    return v_h


def FullMultiGrid(A_h, f_h):
    # Check if the space is the coarsest and solve exactly
    if(A_h[2] == coarsest_level):
        u_h = spsolve(A_sp_dict[coarsest_level][0], f_h)
        u_h_vec = np.array(u_h).reshape(len(u_h), 1)
        return u_h_vec
    else:
        f_2h = b_dict[A_h[2]-1]  # Fetching the exact RHS from the dict
        # Get the next coarse level of Discretization Matrix
        A_2h = A_jacobi_sp_dict[A_h[2] - 1]
        v_2h = FullMultiGrid(A_2h, f_2h)
    v_h = Interpolation2D(v_2h, A_h[2] - 1)
    if (A_h[2] == finest_level):
        # If at finest level , run until the residual is below E-11
        finest_level_V_cycle_count = 0
        while True:
            v_h = V_cycle_scheme(A_h, v_h, f_h)
            finest_level_V_cycle_count += 1
            res_h = f_h - A_sp_dict[finest_level][0].dot(v_h)
            error_norm = err_calculator(v_h, u_exact_fine, V_fine_dolfx)
            error_per_V_cycle_finest.append(error_norm)
            res_norm = res_calculator(res_h, V_fine_dolfx)
            residual_per_V_cycle_finest.append(res_norm)
            if res_norm <= 1E-11:
                # Write the iter count for finest_elements_to CSV file and return
                with open('iter_count_for_diff_num_elems.csv', mode='a') as file1:
                    iter_writer = csv.writer(file1, delimiter=',')
                    iter_writer.writerow(
                        [coarsest_level_elements_per_dim * 2**finest_level, finest_level_V_cycle_count])
                return v_h

    else:
        # Else run for a specified number of V-Cycles
        for i in range(mu0):
            v_h = V_cycle_scheme(A_h, v_h, f_h)
        return v_h


for key, value in A_sp_dict.items():
    A_jacobi_sp_dict[key] = getJacobiMatrices(value)


u_FMG = FullMultiGrid(A_jacobi_sp_dict[finest_level], b_dict[finest_level])
writing_residual_for_mesh_to_csv(residual_per_V_cycle_finest)
writing_error_for_mesh_to_csv(error_per_V_cycle_finest)

# Writing the final_error of CG1 dolfinx for comparison
with open(f'error_for_{coarsest_level_elements_per_dim * 2**finest_level}.csv', mode='a') as file:
    error_writer_dolfx = csv.writer(file, delimiter=',')
    error_writer_dolfx.writerow(['Dolf', error_L2_dolfx_norm])
