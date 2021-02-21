import dolfinx
from mpi4py import MPI
from petsc4py import PETSc

mesh1 = dolfinx.UnitSquareMesh(
    MPI.COMM_WORLD, 4, 4, dolfinx.cpp.mesh.CellType.triangle)
mesh2 = dolfinx.UnitSquareMesh(
    MPI.COMM_WORLD, 2, 2, dolfinx.cpp.mesh.CellType.triangle)

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

print(mesh1_dict[5])
print(mesh2_dict)
