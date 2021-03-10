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
coarsest_level = finest_level - 2  # For 3 Level V-Cycle
coarsest_level_elements_per_dim = 4
residual_per_V_cycle = []

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
a_fine_dolfx = None  # Stores the bilinear operator for finest level                |
L_fine_dolfx = None  # Stores the linear operator for finest level                  |
# Stores the BCs for finest level                          |------- For Performing the Comparison
bcs_fine_dolfinx = None
b_fine_dolfx = None  # Stores the b object for finest level                          |
V_fine_dolfx = None  # Stores the CG1 Function space for finest level                |
