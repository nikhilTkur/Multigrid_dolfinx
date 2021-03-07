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

mesh = dolfinx.UnitSquareMesh(
    MPI.COMM_WORLD, 8, 8, dolfinx.cpp.mesh.CellType.triangle)
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
L = f * v * ufl.dx

b = dolfinx.fem.create_vector(L)
with b.localForm() as loc_b:
    loc_b.set(0)
dolfinx.fem.assemble_vector(b, L)
dolfinx.fem.apply_lifting(b, [a], [[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
              mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.set_bc(b, [bc])
f = np.array(b.array).reshape(81, 1)

A_mat_diag = A_sp.diagonal()
R = A_sp - scp.sparse.diags(A_mat_diag, 0)
A_mat_diag_inv = 1 / A_mat_diag
diag_inv = scp.sparse.diags(A_mat_diag_inv, 0)  # Jacobi_matrix1
R_omega = diag_inv.dot(R)                      # Jacobi Matrix2

jacobi_residual = []

problem = dolfinx.fem.LinearProblem(a, L, bcs=[bc], petsc_options={
                                    "ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
V2 = dolfinx.FunctionSpace(mesh, ("CG", 2))
u_exact = dolfinx.Function(V2)
u_exact.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
u_exact.vector.ghostUpdate(
    addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
u_exact_vertex_values = u_exact.compute_point_values()
L2_error_dolfinx = ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx
error_L2_dolfinx = np.sqrt(dolfinx.fem.assemble_scalar(L2_error_dolfinx))
iter = 100
jacob_relax_soln = None
v = np.zeros((81, 1))
for i in range(0, iter):
    jacob_relax_soln = (1-(2/3)) * v + (2/3) * \
        diag_inv.dot(f) - (2/3) * R_omega.dot(v)
    v = jacob_relax_soln
    jacobi_residual.append(np.linalg.norm(f - A_sp.dot(jacob_relax_soln)))

plt.figure(1)
x_axis = np.arange(0, iter)
axis1 = plt.axes()
axis1.grid(True)
axis1.semilogy(x_axis, jacobi_residual)
plt.savefig("J_Residual.png")

print(f'Dolfinx error is {error_L2_dolfinx}')

u_jac = dolfinx.Function(V)
u_jac.vector[:] = v
L2_error_jac = ufl.inner(u_jac - u_exact, u_jac - u_exact) * ufl.dx
error_L2_jac = np.sqrt(dolfinx.fem.assemble_scalar(L2_error_jac))
print(f'error in jacobi is {error_L2_jac}')
