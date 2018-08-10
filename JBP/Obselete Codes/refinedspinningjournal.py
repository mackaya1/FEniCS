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
r_a=0.031
r_b=0.035

# Create geometry 1
x1=0.0
y1=0.00
x2=0.002
y2=0.0


c1=Circle(Point(x1,y1), r_a)
c2=Circle(Point(x2,y2), r_b)

cyl=c2-c1

# Create mesh
mesh = generate_mesh(cyl, 35)

print('Number of Cells:', mesh.num_cells())
print('Number of Vertices:', mesh.num_vertices())
plot(mesh, interactive=True)

tol = 0.005
iter = 0         
maxiter = 2

# Refinement Code 1
iter = 0         
maxiter = 2
while iter < maxiter:
      iter+=1
      gam = 0.68
      class Omega0(SubDomain):
            def inside(self, x, on_boundary):
                r = sqrt((x[0]-x1)**2+(x[1]-y1)**2)
                return True if r <= gam*r_a + (1-gam)*r_b  else False
      omega0= Omega0()
      class Omega1(SubDomain):
            def inside(self, x, on_boundary):
                p = sqrt((x[0]-x2)**2+(x[1]-y2)**2)
                return True if p > (1-gam)*r_a + gam*r_b  else False
      omega1= Omega1()
      cell_domains = CellFunction("bool", mesh)
      cell_domains.set_all(False)
      
      #omega1.mark(cell_domains, True)
      omega0.mark(cell_domains, True)
      mesh = refine(mesh, cell_domains, redistribute=True)
      print('Number of Cells:', mesh.num_cells())
      print('Number of Vertices:', mesh.num_vertices())

# Plot cell_domains
plot(cell_domains, interactive=True)

# Set parameter values
h = mesh.hmin()
dt = 0.0005      #0.025*h/1.
Tf = 20.0
rho = 820.0
cp = 4000.0
mu = 1.0*10E-3
w_j = 5.0
eps = 10E-6
kappa = 25.0
heatt= 2.0



# Define function spaces 
V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 2)
W = TensorFunctionSpace(mesh, "CG", 1)

# Define trial and test functions
u = TrialFunction(V)
p = TrialFunction(Q)
T = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)


#Define inner cylinder velocity 
        
w = Expression(('-w_j*(x[1]-y1)' , 'w_j*(x[0]-x1)' ), w_j=w_j , r_a=r_a, x1=x1, y1=y1 )


#Define Boundaries (Method 2)   
class Omega0(SubDomain):
      def inside(self, x, on_boundary):
          return True if ((x[0]-x1)*(x[0]-x1))+((x[1]-y1)*(x[1]-y1)) < r_a*r_a + DOLFIN_EPS else False

class Omega1(SubDomain):
      def inside(self, x, on_boundary):
          return True if ((x[0]-x2)*(x[0]-x2))+((x[1]-y2)*(x[1]-y2)) > (1.0-6*10E-3)*r_b*r_b  else False

omega0= Omega0()
omega1= Omega1()



# Define boundary conditions
spin =  DirichletBC(V, w, omega0)  #The inner cylinder will be rotated with constant angular velocity w_j
noslip  = DirichletBC(V, (0.0, 0.0), omega1) #The outer cylinder remains fixed with zero velocity 
temp0 =  DirichletBC(Q, 100.0, omega0)    #Dirichlet Boundary Condition on the inner bearing 


#Collect Boundary Conditions
bcu = [noslip, spin]
bcp = []
bcT = [temp0]

# Create functions
u0 = Function(V)
u1 = Function(V)
p1 = Function(Q)
T0 = Function(Q)
T1 = Function(Q)

# Define coefficients 
k = Constant(dt)
alpha = 1.0/(rho*cp)
ecc = (x2-x1)/(r_b-r_a)

#Define Strain Rate and other tensors
sr = 0.5*(grad(u0) + transpose(grad(u0)))
gamdot = inner(sr,grad(u0))

#CHORIN'S PROJECTION METHOD

# Tentative velocity step
a1 = inner(u, v)*dx +  k*(mu/rho)*inner(grad(u), grad(v))*dx 
L1 = inner(u0,v)*dx - k*inner(grad(u0)*u0, v)*dx 

# Pressure update 
a2 = k*inner(grad(p), grad(q))*dx #+ eps*inner(p,q)*dx
L2 = -div(u1)*q*dx

# Velocity update
a3 = inner(u, v)*dx
L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx


# Temperature Update
a4 = inner(T,q)*dx + k*kappa*alpha*inner(grad(T),grad(q))*dx + k*inner(inner(u1,grad(T)),q)*dx
L4 = inner(T0,q)*dx + k*alpha*(heatt*inner(T0,q)*ds(1) + mu*inner(gamdot,q)*dx)  #Neumann Condition on the outer bearing is encoded in the weak formulation

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)
A4 = assemble(a4)


#Define Surface Integral for Torque Calculation

# Define unit Normal Vector at inner and outer Boundary (Method 1)
#n0 = FacetNormal(omega0)
#n1 = FacetNormal(omega1)

# Define unit Normal Vector at inner and outer Boundary (Method 2)

n0 =  Expression(('(x[0]-x1)/r_a' , '(x[1]-y1)/r_a' ), r_a=r_a, x1=x1, y1=y1)
n1 =  Expression(('(x[0]-x2)/r_b' , '(x[1]-y2)/r_b' ), r_b=r_b, x2=x2, y2=y2)
t0 =  Expression(('-(x[1]-y1)/r_a' , '(x[0]-x1)/r_a' ), r_a=r_a, x1=x1, y1=y1)
t1 =  Expression(('-(x[1]-y2)/r_b' , '(x[0]-x2)/r_b' ), r_b=r_b, x2=x2, y2=y2)

n0v = interpolate(n0, V)


I = Expression((('1.0','0.0'),
                ('0.0','1.0')), degree=2)


boundary_parts = FacetFunction("size_t", mesh)
omega0.mark(boundary_parts,0)
omega1.mark(boundary_parts,1)
ds = Measure("ds")[boundary_parts]

sigma0 = (p1*I+2*mu*sr)*t0
sigma1 = (p1*I+2*mu*sr)*t1

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

# Time-stepping
iter = 0            # iteration counter
maxiter = 1150  
t = dt
while t < Tf + DOLFIN_EPS and iter < maxiter:
    iter += 1

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

    # Temperature correction
    begin("Computing temperature correction")
    b4 = assemble(L4)
    [bc.apply(A4, b4) for bc in bcT]
    solve(A4, T1.vector(), b4, "bicgstab", "default")
    end()

    # Calcultate Force on the bearing
    print("Normal Force on inner bearing: ",  (assemble(innerforcex), assemble(innerforcey)))
    print("Torque on inner bearing: ", assemble(innertorque))
    

    # Plot solution
    plot(p1, title="Pressure", rescale=True)
    plot(u1, title="Velocity", rescale=True, mode = "auto")
    plot(T1, title="Temperature", rescale=True)

    # Move to next time step
    u0.assign(u1)
    T0.assign(T1)
    t += dt
    print("t =", t)

# Hold plot
plot(mesh)
interactive()

