# TESTS THE INTERPOLATION AND RESTRICTION OPERATORS FOR 2 PARTICULAR LEVELS OF A MESH HEIRARCHY COMPARING THEIR RHS I.E., B_VECTOR

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
from multigrid import Interpolation2D, Restriction2D_direct

num_elems_fine = 8
num_elems_coarse = num_elems_fine // 2
element_size_coarse = 1/num_elems_coarse
element_size_fine = 1/num_elems_fine

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
u_fine.interpolate(lambda x: 1 + 2*x[0]**2 + x[1]**2)
u_fine.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)

u_coarse = dolfinx.Function(V_coarse)
u_coarse.interpolate(lambda x: 1 + 2*x[0]**2 + x[1]**2)
u_coarse.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                            mode=PETSc.ScatterMode.FORWARD)
#plane_2h = np.array(u_coarse.vector.array).reshape((num_elems_coarse+1)**2, 1)
#plane_h = np.array(u_fine.vector.array).reshape((num_elems_fine+1)**2, 1)

fdim_fine = mesh_fine.topology.dim-1
fdim_coarse = mesh_coarse.topology.dim-1

mesh_fine.topology.create_connectivity(fdim_fine, mesh_fine.topology.dim)
mesh_coarse.topology.create_connectivity(fdim_coarse, mesh_coarse.topology.dim)

boundary_facets_fine = np.where(
    np.array(dolfinx.cpp.mesh.compute_boundary_facets(mesh_fine.topology)) == 1)[0]
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
#A_fine = dolfinx.fem.assemble_matrix(a_fine, bcs=[bc_fine])
# A_fine.assemble()

bc_coarse = dolfinx.DirichletBC(u_coarse, boundary_dofs_coarse)
u_D_coarse = ufl.TrialFunction(V_coarse)
v_D_coarse = ufl.TestFunction(V_coarse)
f_coarse = dolfinx.Constant(mesh_coarse, -6)
a_coarse = ufl.dot(ufl.grad(u_D_coarse), ufl.grad(v_D_coarse)) * ufl.dx
#A_coarse = dolfinx.fem.assemble_matrix(a_coarse, bcs=[bc_coarse])
# A_coarse.assemble()

"""ai1, aj1, av1 = A_fine.getValuesCSR()
A_sp_fine = scp.sparse.csr_matrix((av1, aj1, ai1))

ai2, aj2, av2 = A_coarse.getValuesCSR()
A_sp_coarse = scp.sparse.csr_matrix((av2, aj2, ai2))"""

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

b_vec_restricted = Restriction2D_direct(
    b_vec_fine, mesh_dof_dict_coarse, mesh_dof_dict_fine, b_vec_coarse.shape[0])
b_vec_interpolated = Interpolation2D(
    b_vec_coarse, mesh_dof_dict_coarse, mesh_dof_dict_fine, element_size_coarse, element_size_fine, b_vec_fine.shape[0])
diff_interpolation = b_vec_interpolated - b_vec_fine
diff_restricted = b_vec_restricted - b_vec_coarse
assert abs(diff_restricted) < 1E-2
assert abs(diff_interpolation) < 1E-2
