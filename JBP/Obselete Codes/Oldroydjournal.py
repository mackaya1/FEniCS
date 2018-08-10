from __future__ import print_function
""" Journal Bearing Lubrication"""
"""Solution to the Momentum and Energy equation for an Oldroyd-B Fluid """
"""ADAPTED CHORINS PROJECTION METHOD"""
"""Uniform Mesh Spacing. Unrefined at the Journal"""


from decimal import *
from dolfin import *
from mshr import *
from math import pi, sin, cos, sqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

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

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;

# Construct hollow cylinder mesh
r_a=0.08#Journal Radius
r_b=0.1 #Bearing Radius

# Create geometry 1
x1=-0.015
y1=0.0
x2=0.0
y2=0.0

c1=Circle(Point(x1,y1), r_a)
c2=Circle(Point(x2,y2), r_b)

e=x2-x1
c=r_b-r_a

if c < 0:
   print("ERROR! Journal radius greater than bearing radius")
   quit()

# Create mesh
cyl=c2-c1

mesh = generate_mesh(cyl, 45)
print('Number of Cells:', mesh.num_cells())
print('Number of Vertices:', mesh.num_vertices())

# Define function spaces (P2-P1)
V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 2)
W = TensorFunctionSpace(mesh, "CG", 1)

# Define trial and test functions
u = TrialFunction(V)
p = TrialFunction(Q)
T = TrialFunction(Q)
S = TrialFunction(W)
R = TestFunction(W)
v = TestFunction(V)
q = TestFunction(Q)

# Set parameter values
h = mesh.hmin()
dt = 0.00025
Tf = 1.0
rho = 820.0
cp = 4000.0
mu1 = 5.0*10E-0
mu2 = 5.0*10E-2
lambda1 = 1.0*10E-2
w_j = 10.0
eps = 10E-10
kappa = 2.0
heatt= 2.0

#Define Boundaries                                                                          
class Omega0(SubDomain):
      def inside(self, x, on_boundary):
          return True if ((x[0]-x1)*(x[0]-x1))+((x[1]-y1)*(x[1]-y1)) < r_a*r_a + DOLFIN_EPS else False

class Omega1(SubDomain):
      def inside(self, x, on_boundary):
          return True if ((x[0]-x2)*(x[0]-x2))+((x[1]-y2)*(x[1]-y2)) > (1.0-10E-3)*r_b*r_b  else False

omega0= Omega0()
omega1= Omega1()


# Create mesh functions over the cell facets
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
sub_domains.set_all(3)
omega0.mark(sub_domains, 0)
omega1.mark(sub_domains, 1)


#Define inner cylinder velocity 
        
w = Expression(('w_j*(x[1]-y1)' , '-w_j*(x[0]-x1)' ), w_j=w_j , r_a=r_a, x1=x1, y1=y1 )


# Define unit Normal Vector at inner and outer Boundary

n0 =  Expression(('(x[0]-x1)/r_a' , '(x[1]-y1)/r_a' ), r_a=r_a, x1=x1, y1=y1)

n1 =  Expression(('(x[0]-x2)/r_b' , '(x[1]-y2)/r_b' ), r_b=r_b, x2=x2, y2=y2)

# Define boundary conditions
spin =  DirichletBC(V, w, omega0)  #The inner cylinder will be rotated with constant angular velocity w_a
noslip  = DirichletBC(V, (0, 0), omega1) #The outer cylinder remains fixed with zero velocity 
inflow  = DirichletBC(Q, 0, omega0)
outflow = DirichletBC(Q, 0, omega1) #Zero pressure condition at time t=0
temp0 =  DirichletBC(Q, 100.0, omega0)    #Dirichlet Boundary Condition on the inner bearing 


#Collect Boundary Conditions
bcu = [noslip, spin]
bcp = []
bcT = [temp0]
bcS = []


# Create functions
u0 = Function(V)
u1 = Function(V)
p1 = Function(Q)
T0 = Function(Q)
T1 = Function(Q)
S0 = Function(W)
S1 = Function(W)

# Define coefficients 
k = Constant(dt)
alpha = 1.0/(rho*cp)
eta = lambda1/k
ecc = (e)/(c)

#Define Strain Rate and other tensors
sr = 0.5*(grad(u0) + transpose(grad(u0)))
gamdots = inner(sr,grad(u0))
gamdotp = inner(S1, grad(u0))
F = (grad(u0)*S0 + S0*transpose(grad(u0)))
F1 = (grad(u1)*S + S*transpose(grad(u1)))

#CHORIN'S PROJECTION METHOD

# Tentative velocity step
a1 = inner(u, v)*dx #+  k*(mu1/rho)*inner(grad(u), grad(v))*dx 
L1 = inner(u0,v)*dx - k*inner(grad(u0)*u0, v)*dx - (k/rho)*inner(S0,grad(v))*dx

# Pressure update 
a2 = k*inner(grad(p), grad(q))*dx 
L2 = -div(u1)*q*dx

# Velocity update
a3 = inner(u, v)*dx 
L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx

# Stress Update
a4 = (eta + 1.0)*inner(S,R)*dx + lambda1*inner(dot(u1,grad(S)),R)*dx - lambda1*inner(F1, R)*dx
L4 = + eta*inner(S0,R)*dx + 2*mu2*inner(sr,R)*dx #+ lambda1*inner(F, R)*dx 

# Temperature Update
a5 = inner(T,q)*dx + k*kappa*alpha*inner(grad(T),grad(q))*dx + k*inner(inner(u1,grad(T)),q)*dx
L5 = inner(T0,q)*dx + k*alpha*(heatt*inner(T0,q)*ds(1) + mu1*inner(gamdots,q)*dx + inner(gamdotp,q)*dx)  #Neumann Condition on the outer Bearing is encoded in the weak formulation


# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)
A4 = assemble(a4)
A5 = assemble(a5)

# Define unit Normal Vector at inner and outer Boundary (Method 2)

n0 =  Expression(('(x[0]-x1)/r_a' , '(x[1]-y1)/r_a' ), r_a=r_a, x1=x1, y1=y1)
n1 =  Expression(('(x[0]-x2)/r_b' , '(x[1]-y2)/r_b' ), r_b=r_b, x2=x2, y2=y2)
t0 =  Expression(('(x[1]-y1)/r_a' , '-(x[0]-x1)/r_a' ), r_a=r_a, x1=x1, y1=y1)
t1 =  Expression(('(x[1]-y2)/r_b' , '-(x[0]-x2)/r_b' ), r_b=r_b, x2=x2, y2=y2)

n0v = interpolate(n0, V)


I = Expression((('1.0','0.0'),
                ('0.0','1.0')), degree=2)


boundary_parts = FacetFunction("size_t", mesh)
omega0.mark(boundary_parts,0)
omega1.mark(boundary_parts,1)
ds = Measure("ds")[boundary_parts]

sigma0 = (p1*I+2*mu1*sr + S0)*t0
sigma1 = (p1*I+2*mu1*sr + S0)*t1

omegaf0 = p1*I*n0  #Nomral component of the stress 
omegaf1 = p1*I*n1


innerforcex = -inner(Constant((1.0, 0.0)), omegaf0)*ds(0)
innerforcey = -inner(Constant((0.0, 1.0)), omegaf0)*ds(0)
outerforcex = -inner(Constant((1.0, 0.0)), omegaf1)*ds(1)
outerforcey = -inner(Constant((0.0, 1.0)), omegaf1)*ds(1)
innertorque = -inner(n0, sigma0)*ds(0)
outertorque = -inner(n1, sigma1)*ds(1)


# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Use nonzero guesses - essential for CG with non-symmetric BC
parameters['krylov_solver']['nonzero_initial_guess'] = True


print("eccentricity ratio", ecc)

plot(mesh, interactive=True)
# Time-stepping
t = dt

# Time/Torque Plot
x=list()
y=list()
z=list()

# Time-stepping
iter = 0            # iteration counter
maxiter = 1000 
t = dt
while t < Tf + DOLFIN_EPS and iter < maxiter:
    print("t =", t)
    print("iteration", iter)
    iter+=1
    
    # Compute tentative velocity step
    #begin("Computing tentative velocity")
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1, "bicgstab", "default")
    end()

    # Pressure correction
    #begin("Computing pressure correction")
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcp]
    [bc.apply(p1.vector()) for bc in bcp]
    solve(A2, p1.vector(), b2, "bicgstab", prec)
    end()

    # Velocity correction
    #begin("Computing velocity correction")
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u1.vector(), b3, "bicgstab", "default")
    end()
  
    # Stress correction
   # begin("Computing velocity correction")
    b4 = assemble(L4)
    [bc.apply(A4, b4) for bc in bcS]
    solve(A4, S1.vector(), b4, "bicgstab", "default")
    end()

    # Temperature correction
   # begin("Computing temperature correction")
    b5 = assemble(L5)
    [bc.apply(A5, b5) for bc in bcT]
    solve(A5, T1.vector(), b5, "bicgstab", "default")
    end()

    # Calcultate Force on the bearing
    print("Normal Force on inner bearing: ",  (assemble(innerforcex), assemble(innerforcey)))
    print("Torque on inner bearing: ", assemble(innertorque))

    # Record Torque Data 
    x.append(t)
    y.append(assemble(innertorque))
    z.append(assemble(innerforcey))
    
    # Plot solution
    plot(p1, title="Pressure", rescale=True)
    plot(u1, title="Velocity", rescale=True, mode = "auto")
    #plot(T1, title="Temperature", rescale=True)

    # Move to next time step
    u0.assign(u1)
    T0.assign(T1)
    S0.assign(S1)
    t += dt

mplot(p1)
plt.show()
mplot(T1)
plt.show()


# Plot Torque Data
plt.plot(x, y, 'r-', label='Torque')
plt.show()

plt.plot(x, z, 'b-', label='Vertical Force')
plt.show()


# Hold plot
mplot(mesh)
plt.show()

