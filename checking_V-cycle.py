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

num_elems_fine = 512
num_elems_coarse = num_elems_fine // 2

num_V_cycles = 30
num_jacobi_iter = 2
omega = 2/3

mesh_fine = dolfinx.UnitSquareMesh(
    MPI.COMM_WORLD, num_elems_fine, num_elems_fine, dolfinx.cpp.mesh.CellType.triangle)
V_fine = dolfinx.FunctionSpace(mesh_fine, ("CG", 1))

mesh_coarse = dolfinx.UnitSquareMesh(
    MPI.COMM_WORLD, num_elems_coarse, num_elems_coarse, dolfinx.cpp.mesh.CellType.triangle)
V_coarse = dolfinx.FunctionSpace(mesh_coarse, ("CG", 1))

mesh_dof_dict_fine = {}
mesh_dof_dict_coarse = {}

V_dof_coord_fine = V_fine.tabulate_dof_coordinates()
V_dof_coord_coarse = V_coarse.tabulate_dof_coordinates()

for j in range(0, V_fine.dofmap.index_map.size_local * V_fine.dofmap.index_map_bs):
    cord_tup = tuple(round(cor, 9) for cor in V_dof_coord_fine[j])
    mesh_dof_dict_fine[j] = cord_tup
    mesh_dof_dict_fine[cord_tup] = j

for j in range(0, V_coarse.dofmap.index_map.size_local * V_coarse.dofmap.index_map_bs):
    cord_tup = tuple(round(cor, 9) for cor in V_dof_coord_coarse[j])
    mesh_dof_dict_coarse[j] = cord_tup
    mesh_dof_dict_coarse[cord_tup] = j

u_fine = dolfinx.Function(V_fine)
u_fine.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
u_fine.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)

u_coarse = dolfinx.Function(V_coarse)
u_coarse.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
u_coarse.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                            mode=PETSc.ScatterMode.FORWARD)

fdim_fine = mesh_fine.topology.dim-1
fdim_coarse = mesh_coarse.topology.dim-1

mesh_fine.topology.create_connectivity(fdim_fine, mesh_fine.topology.dim)
mesh_coarse.topology.create_connectivity(fdim_coarse, mesh_coarse.topology.dim)

boundary_facets_fine = np.where(np.array(
    dolfinx.cpp.mesh.compute_boundary_facets(mesh_fine.topology)) == 1)[0]
boundary_facets_coarse = np.where(np.array(
    dolfinx.cpp.mesh.compute_boundary_facets(mesh_coarse.topology)) == 1)[0]

boundary_dofs_fine = dolfinx.fem.locate_dofs_topological(
    V_fine, fdim_fine, boundary_facets_fine)
boundary_dofs_coarse = dolfinx.fem.locate_dofs_topological(
    V_coarse, fdim_coarse, boundary_facets_coarse)

bc_fine = dolfinx.DirichletBC(u_fine, boundary_dofs_fine)
u_D_fine = ufl.TrialFunction(V_fine)
v_D_fine = ufl.TestFunction(V_fine)
f_fine = dolfinx.Constant(mesh_fine, -6)
a_fine = ufl.dot(ufl.grad(u_D_fine), ufl.grad(v_D_fine)) * ufl.dx
A_fine = dolfinx.fem.assemble_matrix(a_fine, bcs=[bc_fine])
A_fine.assemble()

bc_coarse = dolfinx.DirichletBC(u_coarse, boundary_dofs_coarse)
u_D_coarse = ufl.TrialFunction(V_coarse)
v_D_coarse = ufl.TestFunction(V_coarse)
f_coarse = dolfinx.Constant(mesh_coarse, -6)
a_coarse = ufl.dot(ufl.grad(u_D_coarse), ufl.grad(v_D_coarse)) * ufl.dx
A_coarse = dolfinx.fem.assemble_matrix(a_coarse, bcs=[bc_coarse])
A_coarse.assemble()

ai1, aj1, av1 = A_fine.getValuesCSR()
A_sp_fine = scp.sparse.csr_matrix((av1, aj1, ai1))

ai2, aj2, av2 = A_coarse.getValuesCSR()
A_sp_coarse = scp.sparse.csr_matrix((av2, aj2, ai2))

L_fine = f_fine * v_D_fine * ufl.dx
L_coarse = f_coarse * v_D_coarse * ufl.dx

b_fine = dolfinx.fem.create_vector(L_fine)
with b_fine.localForm() as loc_b_fine:
    loc_b_fine.set(0)
dolfinx.fem.assemble_vector(b_fine, L_fine)
dolfinx.fem.apply_lifting(b_fine, [a_fine], [[bc_fine]])
b_fine.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                   mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.set_bc(b_fine, [bc_fine])

b_coarse = dolfinx.fem.create_vector(L_coarse)
with b_coarse.localForm() as loc_b_coarse:
    loc_b_coarse.set(0)
dolfinx.fem.assemble_vector(b_coarse, L_coarse)
dolfinx.fem.apply_lifting(b_coarse, [a_coarse], [[bc_coarse]])
b_coarse.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                     mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.set_bc(b_coarse, [bc_coarse])

b_vec_fine = np.array(b_fine.array).reshape((num_elems_fine+1)**2, 1)
b_vec_coarse = np.array(b_coarse.array).reshape((num_elems_coarse+1)**2, 1)

# Creating A matrices for 2 Levels
A_mat_diag_fine = A_sp_fine.diagonal()
R_fine = A_sp_fine - scp.sparse.diags(A_mat_diag_fine, 0)
A_mat_diag_inv_fine = 1 / A_mat_diag_fine
diag_inv_fine = scp.sparse.diags(A_mat_diag_inv_fine, 0)  # Jacobi_matrix1
R_omega_fine = diag_inv_fine.dot(R_fine)

A_mat_diag_coarse = A_sp_coarse.diagonal()
R_coarse = A_sp_coarse - scp.sparse.diags(A_mat_diag_coarse, 0)
A_mat_diag_inv_coarse = 1 / A_mat_diag_coarse
diag_inv_coarse = scp.sparse.diags(A_mat_diag_inv_coarse, 0)  # Jacobi_matrix1
R_omega_coarse = diag_inv_coarse.dot(R_coarse)

# Solving the CG1 DOlfinx Solution

problem_CG1_fine = dolfinx.fem.LinearProblem(a_fine, L_fine, bcs=[bc_fine], petsc_options={
    "ksp_type": "preonly", "pc_type": "lu"})
uh_fine = problem_CG1_fine.solve()

# Generating Exact CG2 Dolfinx SOlution
V2 = dolfinx.FunctionSpace(mesh_fine, ("CG", 2))
u_exact = dolfinx.Function(V2)
u_exact.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
u_exact.vector.ghostUpdate(
    addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
u_exact_vertex_values = u_exact.compute_point_values()
L2_error_dolfinx = ufl.inner(uh_fine - u_exact, uh_fine - u_exact) * ufl.dx
error_L2_dolfinx = np.sqrt(dolfinx.fem.assemble_scalar(L2_error_dolfinx))

# Interpolation and restriction Operators


def Interpolation2D(vec_2h):
    mesh_dict_coarse = mesh_dof_dict_coarse
    mesh_dict_fine = mesh_dof_dict_fine
    element_size_coarse = 1/num_elems_coarse
    element_size_fine = 1/num_elems_fine

    # target_dof_size[0] * target_dof_size[1]
    vec_h = np.zeros(b_vec_fine.shape)

    # Iterating over the dofs
    for i in range(0, vec_h.shape[0]):
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


def Restriction2D(vec_h):
    mesh_dict_coarse = mesh_dof_dict_coarse
    mesh_dict_fine = mesh_dof_dict_fine
    element_size_coarse = 1/num_elems_coarse
    element_size_fine = 1/num_elems_fine

    vec_2h = np.zeros(b_vec_coarse.shape)

    # Iterating over the dofs
    for i in range(0, vec_2h.shape[0]):
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

# Jacobi smoother


def jacobi_smoothing(R_omega, diag_inv, vec, f, nu):
    sol = vec

    for k in range(0, nu):
        sol = (1-omega) * vec + omega * \
            diag_inv.dot(f) - omega*R_omega.dot(vec)
        vec = sol
    return vec

# Function to calculate Errors


def err_calculation(u_vec):
    u_V_cyc = dolfinx.Function(V_fine)
    u_V_cyc.vector[:] = u_vec
    L2_error_V_cyc = ufl.inner(u_V_cyc - u_exact, u_V_cyc - u_exact) * ufl.dx
    error_L2_V_cyc = np.sqrt(dolfinx.fem.assemble_scalar(L2_error_V_cyc))
    return error_L2_V_cyc

# Function to calculate Residual after each V-cycle


def res_calculation(res):
    res_v_cyc = dolfinx.Function(V_fine)
    res_v_cyc.vector[:] = res
    L2_res_v_cyc = ufl.inner(res_v_cyc, res_v_cyc)*ufl.dx
    res_L2_v_cyc = np.sqrt(dolfinx.fem.assemble_scalar(L2_res_v_cyc))
    return res_L2_v_cyc


# Calculating the exact coarse solution
uh_coarse = spsolve(A_sp_coarse, b_vec_coarse)
uh_V_cycle_soln = Interpolation2D(uh_coarse)
error_before_V_cycle = []
error_after_pre_smooth = []
error_after_correction = []
error_after_V_cycle = []
residual_V_cycle = []
#residual_after_V_cycle = []
for i in range(0, num_V_cycles):
    res = b_vec_fine - A_sp_fine.dot(uh_V_cycle_soln)
    residual_V_cycle.append(res_calculation(res))
    error_before_V_cycle.append(err_calculation(uh_V_cycle_soln))
    # Starting at fine level Pre Smoothing
    vec_h = jacobi_smoothing(R_omega_fine, diag_inv_fine,
                             uh_V_cycle_soln, b_vec_fine, num_jacobi_iter)
    error_after_pre_smooth.append(err_calculation(vec_h))
    # Calcualting the residual of the finer level
    res_h = b_vec_fine - A_sp_fine.dot(vec_h)
    res_2h = Restriction2D(res_h)
    er_2h = spsolve(A_sp_coarse, res_2h)
    er_h = Interpolation2D(er_2h)
    # Correction
    vec_h = vec_h + er_h
    error_after_correction.append(err_calculation(vec_h))
    # Post Smoothing
    vec_h = jacobi_smoothing(R_omega_fine, diag_inv_fine,
                             vec_h, b_vec_fine, num_jacobi_iter)
    uh_V_cycle_soln = vec_h
    error_after_V_cycle.append(err_calculation(uh_V_cycle_soln))

# Tabulating the error results
print("Iter \t err_bfr_V \t err_aft_pre \t err_aft_corr \t err_aft_V \t err_dolfx")
for i in range(0, num_V_cycles):
    print(
        f'{i} \t {error_before_V_cycle[i]:.4E} \t {error_after_pre_smooth[i]:.4E} \t {error_after_correction[i]:.4E} \t {error_after_V_cycle[i]:.4E} \t {error_L2_dolfinx:.4E}')

plt.figure(1)
x_axis = np.arange(0, num_V_cycles)
axis = plt.axes()
axis.grid(True)
axis.semilogy(x_axis, residual_V_cycle)
plt.ylabel("Residual")
plt.xlabel("Num V-cycles")
plt.title("Residual VS V-Cycles")
plt.savefig("Residual-VS-V_Cycles.png")
