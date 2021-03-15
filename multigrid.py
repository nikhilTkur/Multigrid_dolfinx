import scipy as scp
import dolfinx
import ufl
import csv
from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np


mesh_dof_list_dict = None
element_size = None
coarsest_level_elements_per_dim = None
coarsest_level = None
finest_level = None
A_sp_dict = None
A_jacobi_sp_dict = None
b_dict = None
mu0 = None
mu1 = None
mu2 = None
omega = None
residual_per_V_cycle_finest = None
error_per_V_cycle_finest = None
u_exact_fine = None
V_fine_dolfx = None


def initialize_problem(obj):
    global mesh_dof_list_dict, element_size, coarsest_level_elements_per_dim, coarsest_level, finest_level, A_sp_dict, A_jacobi_sp_dict, b_dict, mu0, mu1, mu2, omega, residual_per_V_cycle_finest, error_per_V_cycle_finest, u_exact_fine, V_fine_dolfx
    mesh_dof_list_dict = obj.mesh_dof_list_dict
    element_size = obj.element_size
    coarsest_level_elements_per_dim = obj.coarsest_level_elements_per_dim
    coarsest_level = obj.coarsest_level
    finest_level = obj.finest_level
    A_sp_dict = obj.A_sp_dict
    A_jacobi_sp_dict = obj.A_jacobi_sp_dict
    b_dict = obj.b_dict
    mu0 = obj.mu0
    mu1 = obj.mu1
    mu2 = obj.mu2
    omega = obj.omega
    residual_per_V_cycle_finest = obj.residual_per_V_cycle_finest
    error_per_V_cycle_finest = obj.error_per_V_cycle_finest
    u_exact_fine = obj.u_exact_fine
    V_fine_dolfx = obj.V_fine_dolfx


def getJacobiMatrices(A):
    # Takes in a Global Stiffness Matrix in scipy sparse and gives the Final Jacobi Iteration Matrices
    A_mat = A[0]
    A_mat_diag = A_mat.diagonal()
    R_mat = A_mat - scp.sparse.diags(A_mat_diag, 0)
    A_mat_dig_inv = 1 / A_mat_diag
    diag_A_inv = scp.sparse.diags(A_mat_dig_inv, 0)
    R_omega_mat = diag_A_inv.dot(R_mat)
    return (R_omega_mat, diag_A_inv, A[1])


def Interpolation2D(vec_2h, mesh_dict_coarse, mesh_dict_fine, element_size_coarse, element_size_fine, vec_h_dim):
    """mesh_dict_coarse = mesh_dof_list_dict[level_coarse]
    mesh_dict_fine = mesh_dof_list_dict[level_coarse + 1]
    element_size_coarse = element_size[level_coarse]
    element_size_fine = element_size[level_coarse + 1]

    vec_h_dim = (coarsest_level_elements_per_dim *
                 2**(level_coarse + 1) + 1) ** 2"""
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


def Restriction2D_direct(vec_h, mesh_dict_coarse, mesh_dict_fine, vec_2h_dim):
    """mesh_dict_coarse = mesh_dof_list_dict[level_fine - 1]
    mesh_dict_fine = mesh_dof_list_dict[level_fine]
    vec_2h_dim = (coarsest_level_elements_per_dim * 2**(level_fine-1) + 1) ** 2"""
    vec_2h = np.zeros((vec_2h_dim, 1))
    for i in range(0, vec_2h_dim):
        coord = mesh_dict_coarse[i]
        fine_dof = mesh_dict_fine[coord]
        vec_2h[i] = vec_h[fine_dof]
    return vec_2h


def Restriction2D(vec_h, mesh_dict_coarse, mesh_dict_fine, element_size_coarse, element_size_fine, vec_2h_dim):
    """mesh_dict_coarse = mesh_dof_list_dict[level_fine - 1]
    mesh_dict_fine = mesh_dof_list_dict[level_fine]
    element_size_coarse = element_size[level_fine - 1]
    element_size_fine = element_size[level_fine]

    vec_2h_dim = (coarsest_level_elements_per_dim *
                  2**(level_fine - 1) + 1) ** 2"""
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

# Jacobi Relaxation


def jacobiRelaxation(A, v, f, nw):
    sol = v
    for k in range(0, nw):
        sol = (1-omega) * v + omega * A[1].dot(f) - omega*A[0].dot(v)
        v = sol
    return v


def V_cycle_scheme(A_h, v_h, f_h):
    # Check if the space is the coarsest and solve exactly
    current_level = A_h[2]
    if(current_level == coarsest_level):
        u_h = spsolve(A_sp_dict[coarsest_level][0], f_h)
        u_h_vec = np.array(u_h).reshape(len(u_h), 1)
        return u_h_vec
    else:
        v_h = jacobiRelaxation(A_h, v_h, f_h, mu1)
        r_h = f_h - A_sp_dict[A_h[2]][0].dot(v_h)
        #f_2h = Restriction2D_direct(r_h, A_h[2])

        f_2h_dim = (coarsest_level_elements_per_dim *
                    2**(current_level - 1) + 1) ** 2
        """f_2h = Restriction2D_(r_h, mesh_dof_list_dict[current_level-1], mesh_dof_list_dict[current_level],
                             element_size[current_level-1], element_size[current_level], f_2h_dim)"""
        f_2h = Restriction2D_direct(
            r_h, mesh_dof_list_dict[current_level-1], mesh_dof_list_dict[current_level], f_2h_dim)
        v_2h = np.zeros((f_2h.shape[0], 1))
        # Fetch the Smaller discretication matrix
        A_2h = A_jacobi_sp_dict[A_h[2] - 1]
        v_2h = V_cycle_scheme(A_2h, v_2h, f_2h)
    #v_h = v_h + Interpolation2D(v_2h, A_h[2] - 1)
    v_h = v_h + Interpolation2D(v_2h, mesh_dof_list_dict[current_level-1], mesh_dof_list_dict[current_level],
                                element_size[current_level-1], element_size[current_level], f_h.shape[0])
    v_h = jacobiRelaxation(A_h, v_h, f_h, mu2)
    return v_h


def FullMultiGrid(A_h, f_h):
    current_level = A_h[2]
    # Check if the space is the coarsest and solve exactly
    if(current_level == coarsest_level):
        u_h = spsolve(A_sp_dict[coarsest_level][0], f_h)
        u_h_vec = np.array(u_h).reshape(len(u_h), 1)
        return u_h_vec
    else:
        f_2h = b_dict[A_h[2]-1]  # Fetching the exact RHS from the dict
        # Get the next coarse level of Discretization Matrix
        A_2h = A_jacobi_sp_dict[current_level - 1]
        v_2h = FullMultiGrid(A_2h, f_2h)
    v_h = Interpolation2D(v_2h, mesh_dof_list_dict[current_level-1], mesh_dof_list_dict[current_level],
                          element_size[current_level-1], element_size[current_level], f_h.shape[0])
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
                with open(f'iter_count_for_diff_num_elems_{finest_level-coarsest_level +1}_levels.csv', mode='a') as file1:
                    iter_writer = csv.writer(file1, delimiter=',')
                    iter_writer.writerow(
                        [coarsest_level_elements_per_dim * 2**finest_level, finest_level_V_cycle_count])
                return v_h
    else:
        # Else run for a specified number of V-Cycles
        for i in range(mu0):
            v_h = V_cycle_scheme(A_h, v_h, f_h)
        return v_h

#  fn for Writing the residual for a particular num_elements for finest level into a csv file


def writing_residual_for_mesh_to_csv(residual):
    with open(f'residual_for_{coarsest_level_elements_per_dim * 2**finest_level}_{finest_level-coarsest_level +1}_levels.csv', mode='w') as file:
        residual_writer = csv.writer(file, delimiter=',')
        for i in range(0, len(residual)):
            residual_writer.writerow([i, residual[i]])


def writing_error_for_mesh_to_csv(error):
    with open(f'error_for_{coarsest_level_elements_per_dim * 2**finest_level}_{finest_level-coarsest_level +1}_levels.csv', mode='w') as file:
        error_writer = csv.writer(file, delimiter=',')
        for i in range(0, len(error)):
            error_writer.writerow([i, error[i]])
#
