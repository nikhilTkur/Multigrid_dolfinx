import dolfinx
from mpi4py import MPI

mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 4, 4)
space = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))

# cell_dofs tells you which DOFs are on each cell
print(space.dofmap.cell_dofs(0))
# [10, 15, 11]
# This tells you that DOFs 10, 15 and 11 are on cell 0
# so entries 10, 15, 11 of the vector are associated with this cell.

# entity_dofs tells you which (local) DOFs are associated with each mesh entity

print(space.dofmap.dof_layout.entity_dofs(0, 0))
# [0]
# This tells you that DOF 0 is on the 0th vertex

print(space.dofmap.dof_layout.entity_dofs(0, 1))
# [1]
# This tells you that DOF 1 is on the 1th vertex

print(space.dofmap.dof_layout.entity_dofs(0, 2))
# [2]
# This tells you that DOF 2 is on the 2th vertex

print(space.dofmap.dof_layout.entity_dofs(1, 0))
# []
# This tells you that there are no DOFs on the 0th edge


# mesh.geometry.x gives you the coordinates of the points in the mesh
print(mesh.geometry.x)
# [[0.  , 1.  , 0.  ], [0.  , 0.75, 0.  ], .....]

# mesh.geometry.dofmap.links tells you which points are attached to each cell
print(mesh.geometry.dofmap.links(0))
# [10, 15, 11]
# This tells you that points 10, 15 and 11 are the corners of cell 0

# You can use this to print the corners of cell 0
for vertex in mesh.geometry.dofmap.links(0):
    print(mesh.geometry.x[vertex])
# [0. 0. 0.]
# [0.25 0.   0.  ]
# [0.25 0.25 0.  ]

#
# For Lagrange order 1, mesh.geometry.dofmap.links and space.dofmap.cell_dofs
# will (almost?) always give you the same numbers. But for other spaces, they won't, eg:
space2 = dolfinx.FunctionSpace(mesh, ("Lagrange", 2))
print(space2.dofmap.cell_dofs(0))
# [28, 45, 29, 53, 40, 54]
print(mesh.geometry.dofmap.links(0))
# [10, 15, 11]
