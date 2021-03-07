import dolfinx
import numpy as np
import ufl
import math
import scipy as scp
from scipy import sparse
from mpi4py import MPI
from petsc4py import PETSc
from numpy.linalg import inv

mesh1 = dolfinx.UnitSquareMesh(
    MPI.COMM_WORLD, 4, 4, dolfinx.cpp.mesh.CellType.triangle)
mesh2 = dolfinx.UnitSquareMesh(
    MPI.COMM_WORLD, 8, 8, dolfinx.cpp.mesh.CellType.triangle)

V1 = dolfinx.FunctionSpace(mesh1, ("CG", 1))
V2 = dolfinx.FunctionSpace(mesh2, ("CG", 1))
V1_dof_cord = V1.tabulate_dof_coordinates()
V2_dof_cord = V2.tabulate_dof_coordinates()

mesh1_dict = {}
mesh2_dict = {}

# Storing the coordinates of the dofs for mapping between the various grids of problem

for i in range(0, V1.dofmap.index_map.size_local * V1.dofmap.index_map_bs):
    mesh1_dict[i] = tuple([round(i, 3) for i in V1_dof_cord[i]])
    mesh1_dict[tuple([round(i, 3) for i in V1_dof_cord[i]])] = i

for i in range(0, V2.dofmap.index_map.size_local * V2.dofmap.index_map_bs):
    mesh2_dict[i] = tuple([round(i, 3) for i in V2_dof_cord[i]])
    mesh2_dict[tuple([round(i, 3) for i in V2_dof_cord[i]])] = i

# print(V1_dof_cord[1][1])
# print(mesh1_dict)
print(mesh1_dict[2])
print(mesh2_dict[5])
print(V1_dof_cord)
"""
uD = dolfinx.Function(V)
uD.vector.set(0.0)
#uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
uD.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)
fdim = mesh.topology.dim-1
mesh.topology.create_connectivity(fdim, mesh.topology.dim)
boundary_facets = np.where(np.array(
    dolfinx.cpp.mesh.compute_boundary_facets(mesh.topology)) == 1)[0]
boundary_facets_2 = dolfinx.mesh.locate_entities_boundary(
    mesh, fdim, lambda x: np.isclose(x[0], 1.0))
boundary_dofs = dolfinx.fem.locate_dofs_topological(
    V, fdim, boundary_facets_2)
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
A_c = A_sp.toarray()
L = f * v * ufl.dx
problem = dolfinx.fem.LinearProblem(a, L, bcs=[bc], petsc_options={
                                    "ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
print(uh.compute_point_values())
b = dolfinx.fem.create_vector(L)
with b.localForm() as loc_b:
    loc_b.set(0)
    dolfinx.fem.assemble_vector(b, L)
    # print(b.array)
    dolfinx.fem.apply_lifting(b, [a], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, [bc])
b_np = np.array(b.array).reshape(25, 1)
u = np.dot(inv(A_c), b_np)
print(u)
# print(b_np)
# print(A_sp.shape)
"""
