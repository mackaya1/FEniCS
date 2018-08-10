"""Solution to the Momentum and Energy equation for a NEWTONIAN VISCOUS Fluid """

from __future__ import print_function
from decimal import *
from dolfin import *
from mshr import *
from math import pi, sin, cos, sqrt
import numpy as np

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;


# Construct hollow cylinder mesh
r_a=0.055
r_b=0.095

# Create geometry 1
x1=-0.03
y1=-0.01
x2=0.0
y2=0.0

c1=Circle(Point(x1,y1), r_a)
c2=Circle(Point(x2,y2), r_b)

cyl=c2-c1

# Create mesh
mesh = generate_mesh(cyl, 30)

plot(mesh, interactive=True)


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
dt = 0.25*h/1.
Tf = 20.0
rho = 1000.0
cp = 1000.0
mu1 = 1*0.05
mu2 = 1*1.0
lambda1 = 0.1
w_j = 5.0
eps = 10E-10
kappa = 50.0
heatt= 2.0

#Define Boundaries (Method 1)

def c1_boundary(x):
    tol = 1E-10
    return abs(((x[0]-x1)*(x[0]-x1))+((x[1]-y1)*(x[1]-y1))) < r_a*r_a + DOLFIN_EPS
           
           
def c2_boundary(x):
    tol = 1E-10
    return abs(((x[0]-x2)*(x[0]-x2))+((x[1]-y2)*(x[1]-y2)))  > (1.0-10E-8)*r_b*r_b #This method labels all points within a tolerence distance of the radius of journal b as outer boundary points  

#Define Boundaries (Method 2)   
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
        
w = Expression(('-w_j*(x[1]-y1)' , 'w_j*(x[0]-x1)' ), w_j=w_j , r_a=r_a, x1=x1, y1=y1 )


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


#Define Surface Integral for Torque Calculation

boundary_parts = FacetFunction("size_t", mesh)
omega0.mark(boundary_parts,0)
omega1.mark(boundary_parts,1)
ds = Measure("ds")[boundary_parts]



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
eta = k/lambda1

#Define Strain Rate and other tensors
sr = 0.5*(grad(u0) + transpose(grad(u0)))
gamdot = inner((2*mu1*sr+S0),grad(u0))

#CHORIN'S PROJECTION METHOD

# Tentative velocity step
a1 = inner(u, v)*dx +  k*(mu1/rho)*inner(grad(u), grad(v))*dx 
L1 = inner(u0,v)*dx - k*inner(grad(u0)*u0, v)*dx - (k/rho)*inner(S0,grad(v))*dx


# Pressure update 
a2 = k*inner(grad(p), grad(q))*dx 
L2 = -div(u1)*q*dx

# Velocity update
a3 = inner(u, v)*dx
L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx

# Stress Update
a4 = inner(S,R)*dx
L4 = (1/(eta + 1.0))*(eta*inner(S0,R)*dx + 2*mu2*inner(sr,R)*dx)


# Temperature Update
a5 = inner(T,q)*dx + k*kappa*alpha*inner(grad(T),grad(q))*dx + k*inner(inner(u1,grad(T)),q)*dx
L5 = inner(T0,q)*dx + k*alpha*(heatt*inner(T0,q)*ds(1) + inner(gamdot,q)*dx)  #Neumann Condition on the outer Bearing is encoded in the weak formulation


# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)
A4 = assemble(a4)
A5 = assemble(a5)


#Define Surface Integral for Torque Calculation

I = Expression((('1.0','0.0'),
                ('0.0','1.0')), degree=2)

omegaf0 = p1*I*n0  #Nomral component of the stress 
omegaf1 = p1*I*n1
torque0 = S0*n0
torque1 = S0*n1

innerforcex = -inner(Constant((1.0, 0.0)), omegaf0)*ds(0)
innerforcey = -inner(Constant((0.0, 1.0)), omegaf0)*ds(0)
outerforcex = -inner(Constant((1.0, 0.0)), omegaf1)*ds(1)
outerforcey = -inner(Constant((0.0, 1.0)), omegaf1)*ds(1)
innertorque = -inner(Constant((1.0, 1.0)), torque0)*ds(0)
outertorque = -inner(Constant((1.0, 1.0)), torque1)*ds(1)



# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Use nonzero guesses - essential for CG with non-symmetric BC
parameters['krylov_solver']['nonzero_initial_guess'] = True

# Create files for storing solution
ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")

# Time-stepping
t = dt
while t < Tf + DOLFIN_EPS:
    
    # Compute tentative velocity step
    begin("Computing tentative velocity")
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1, "bicgstab", "default")
    end()

    # Pressure correction
    begin("Computing pressure correction")
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcp]
    [bc.apply(p1.vector()) for bc in bcp]
    solve(A2, p1.vector(), b2, "bicgstab", prec)
    end()

    # Velocity correction
    begin("Computing velocity correction")
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u1.vector(), b3, "bicgstab", "default")
    end()
  
    # Stress correction
    begin("Computing velocity correction")
    b4 = assemble(L4)
    [bc.apply(A4, b4) for bc in bcS]
    solve(A4, S1.vector(), b4, "bicgstab", "default")
    end()


    # Temperature correction
    begin("Computing temperature correction")
    b5 = assemble(L5)
    [bc.apply(A5, b5) for bc in bcT]
    solve(A5, T1.vector(), b5, "bicgstab", "default")
    end()

    # Calcultate Force on the bearing
    print("Normal Force on inner bearing: ",  (assemble(innerforcex), assemble(innerforcey)))
    print("Torque on inner bearing: ", assemble(innertorque))
    

    # Plot solution
    plot(p1, title="Pressure", rescale=True)
    plot(u1, title="Velocity", rescale=True, mode = "auto")
    plot(T1, title="Temperature", rescale=True)


    """# Save to file
    ufile << u1
    pfile << p1"""

    # Move to next time step
    u0.assign(u1)
    T0.assign(T1)
    S0.assign(S1)
    t += dt
    print("t =", t)

# Hold plot
plot(mesh)
interactive()

