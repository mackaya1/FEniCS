"""This demo program solves Laplace's equation

    - div grad u(x, y) = 0

on the unit square with source f given by



and boundary conditions given by

    u(x, y) = 0        for y = 0 or y = 1
du/dn(x, y) = 0        for x = 0 or x = 1
du/dn(x, y) = 1        for x = 1
"""

# Copyright (C) 2007-2011 Anders Logg
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2007-08-16
# Last changed: 2012-11-12

# Begin demo

from dolfin import *
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.tri as tri
import matplotlib.mlab as mlab
# MATPLOTLIB CONTOUR FUNCTIONS
def mesh2triang(mesh):
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells()) # Mesh Diagram

def mplot(obj):                     # Function Plot
    plt.gca().set_aspect('equal')
    if isinstance(obj, Function):
        mesh = obj.function_space().mesh()
        if (mesh.geometry().dim() != 2):
            raise(AttributeError)
        if obj.vector().size() == mesh.num_cells():
            C = obj.vector().array()
            plt.tripcolor(mesh2triang(mesh), C)
        else:
            C = obj.compute_vertex_values(mesh)
            plt.tripcolor(mesh2triang(mesh), C, shading='gouraud')
    elif isinstance(obj, Mesh):
        if (obj.geometry().dim() != 2):
            raise(AttributeError)
        plt.triplot(mesh2triang(obj), color='k')


# Create mesh and define function space
mesh = UnitSquareMesh(25, 25)
V = FunctionSpace(mesh, "Lagrange", 1)


# Define Dirichlet boundary (x = 0 or x = 1)

class UpDownBoundary(SubDomain):
      def inside(self, x, on_boundary):
          return True if x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS and on_boundary else False 
boundary = UpDownBoundary()

class Left_Boundary(SubDomain):
      def inside(self, x, on_boundary):
          return True if x[0] < DOLFIN_EPS and on_boundary else False 
left_boundary = Left_Boundary() 

class Right_Boundary(SubDomain):
      def inside(self, x, on_boundary):
          return True if x[0] > 1.0 - DOLFIN_EPS and on_boundary else False 
right_boundary = Right_Boundary() 

# Subdomains
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
sub_domains.set_all(3)
boundary.mark(sub_domains, 0)
left_boundary.mark(sub_domains, 1)
right_boundary.mark(sub_domains, 2)


plot(sub_domains, interactive = False)        # DO NOT USE WITH RAVEN


boundary_parts = FacetFunction("size_t", mesh)
left_boundary.mark(boundary_parts,0)
right_boundary.mark(boundary_parts,1)
ds = ds(subdomain_data = boundary_parts) #Measure("ds")[boundary_parts] = 

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
#f = Constant(0.0) f*v*dx + 
g = Expression("x[0]", degree=1)
a = -inner(grad(u), grad(v))*dx
L = g*v*ds(0) + g*v*ds(1)

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution in VTK format
#file = File("poisson.pvd")
#file << u

# Plot solution
plot(u, interactive=True)
mplot(u)
plt.colorbar()
plt.show()

