from __future__ import print_function
""" Journal Bearing Lubrication"""
"""Nondimensionalised Navier Stokes Equations Using Reynolds Number"""
"""CHORINS PROJECTION METHOD with Galerkin Finite Element Method
      (1st Order)                      (1st Order)                               """



from decimal import *
from dolfin import *
from mshr import *
from math import pi, sin, cos, sqrt
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
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



# Construct hollow cylinder mesh
r_a=0.03125
r_b=0.04125

# Create geometry 1
x1=0.0
y1=0.00
x2=0.005
y2=0.00

c1=Circle(Point(x1,y1), r_a, 256)
c2=Circle(Point(x2,y2), r_b, 256)

c3=Circle(Point(x1,y1), 0.9*r_a, 256)


e=x2-x1                   # Eccentricity 
c=r_b-r_a                 # Lenth Scale of the Flow
ecc = (x2-x1)/(r_b-r_a)   # Eccentricity Ratio 

if c < 0:
   print("ERROR! Journal radius greater than bearing radius")
   quit()


cyl=c2-c1

# Create mesh
mesh = generate_mesh(cyl, 15)

print('Number of Cells:', mesh.num_cells())
print('Number of Vertices:', mesh.num_vertices())

#Define Boundaries 
                                                                          
class Omega0(SubDomain):
      def inside(self, x, on_boundary):
          return True if (x[0]-x1)**2+(x[1]-y1)**2 < (0.75*r_a**2+0.25*r_b**2) and on_boundary  else False  # and 
omega0= Omega0()

class Omega1(SubDomain):
      def inside(self, x, on_boundary):
          return True if (x[0]-x2)**2 + (x[1]-y2)**2 > (0.25*r_a**2+0.75*r_b**2) and on_boundary else False  #
omega1= Omega1()



# Create mesh functions over the cell facets
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
sub_domains.set_all(3)
omega0.mark(sub_domains, 0)
omega1.mark(sub_domains, 1)

#plot(sub_domains, interactive=False)

# Mesh Refine Code
for i in range(1):
      g=1.0/(2*(i+1))
      print(g)
      cell_domains = CellFunction("bool", mesh)
      cell_domains.set_all(False)
      for cell in cells(mesh):
          x = cell.midpoint()
          if  (x[0]-x1)**2+(x[1]-y1)**2 < ((1-g)*r_a**2+g*r_b**2): 
              cell_domains[cell]=True
      #plot(cell_domains, interactive=True)
      mesh = refine(mesh, cell_domains, redistribute=True)
      print('Number of Cells:', mesh.num_cells())
      print('Number of Vertices:', mesh.num_vertices())



#plot(cell_domains)
#plot(mesh, interactive=False)



# Define function spaces (P2-P1)
d=2                                    # Degree of interpolation

V = VectorFunctionSpace(mesh, "CG", d) # Vector Function Space V
Q = FunctionSpace(mesh, "CG", d)       # Scalar Function Space Q
W = TensorFunctionSpace(mesh, "CG", 1) # Tensor Function Space W

# Define trial and test functions
u = TrialFunction(V)          # Velocity Test Function
p = TrialFunction(Q)          # Pressure Test Function
T = TrialFunction(Q)
S = TrialFunction(W)
v = TestFunction(V)
q = TestFunction(Q)
R = TestFunction(W)


# Set parameter values
h = mesh.hmin()             # Minimum Cell diameter
dt = 0.006                   # Timestepping
Tf = 20.0                   # Stopping Time
rho = 1000.0                # Density 
cp = 4000.0                 # Heat Capacity 
f = Expression(('0','-10')) # Body Force (Gravity)
mu = (10.0/100)*3.125*10E-1 # Viscosity
w_j = 100.0                 # Journal angular velocity             
kappa = 2.5                 # Heat Conduction coeficient 
heatt= 5.0                  # Heat Transfer (External )
C = 250.0                   #Sutherland's Constant


#Nondimensional Parameters

Re= rho*w_j*r_a*c/mu     #Reynolds Number 
tstar = r_a*w_j/c 

print('Re=', Re, 'h =', h, 'nondimensional second=', tstar)
print('eccentricity', ecc) 
#quit()



#Define inner cylinder velocity (TIME DEPENDENT)
td= Constant('5')
e = Constant('6')
w = Expression(('(1/w_j)*(0.5*(1.0+tanh(e*t-2.5)))*w_j*(x[1]-y1)' , '-(1/w_j)*(0.5*(1.0+tanh(e*t-2.5)))*w_j*(x[0]-x1)' ), w_j=w_j , r_a=r_a, x1=x1, y1=y1, e=e , t=0.0)
"""NONDIMENSIONALISE THE BOUNDARY CONDITIONS"""

# Define boundary conditions
spin =  DirichletBC(V, w, omega0)  #The inner cylinder will be rotated with constant angular velocity w_j
noslip  = DirichletBC(V, (0.0, 0.0), omega1) #The outer cylinder remains fixed with zero velocity 
temp0 =  DirichletBC(Q, 0.0, omega0)    #Dirichlet Boundary Condition on the inner bearing 


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


#Define Strain Rate and other tensors
sr = 0.5*(grad(u0) + transpose(grad(u0)))
gamdot = inner(sr,grad(u0))



#CHORIN'S PROJECTION METHOD
"""Semi Implicit Euler Discretisation"""
"""Nondimensionalised Equations used"""

# Tentative velocity step
a1 = inner(u, v)*dx 
L1 = inner(u0,v)*dx - k*inner(grad(u0)*u0, v)*dx 

# Pressure update 
a2 = (k/Re)*inner(grad(p), grad(q))*dx 
L2 = -div(u1)*q*dx

# Velocity update
a3 = inner(u, v)*dx + (k/Re)*inner(grad(u), grad(v))*dx 
L3 = inner(u1, v)*dx + (k/Re)*inner(grad(p1), v)*dx

# Temperature Update
a5 = inner(T,q)*dx + k*kappa*alpha*inner(grad(T),grad(q))*dx + k*inner(dot(u1,grad(T)),q)*dx
L5 = inner(T0,q)*dx + k*alpha*(heatt*inner(T0,q)*ds(1) + mu*inner(gamdot,q)*dx)  


# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)
A5 = assemble(a5)

#Define Surface Integral for Torque Calculation

# Define unit Normal Vector at inner and outer Boundary 

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


dimenp=(mu*r_a*w_j)/(c)
dimenu=1/(r_a*w_j)

dimensr=mu*r_a*w_j/(c)

print(dimenp, dimensr)
#quit()

sigma0 = dimenp*(p1*I+2*sr)*t0   # Use Dimensional Variables to compute Torque
sigma1 = dimenp*(p1*I+2*sr)*t1

omegaf0 = dimenp*p1*I*n0  #Nomral component of the stress 
omegaf1 = dimenp*p1*I*n1  # in Dimensional form


innerforcex = -inner(Constant((1.0, 0.0)), omegaf0)*ds(0) # Force values in 
innerforcey = -inner(Constant((0.0, 1.0)), omegaf0)*ds(0) # DIMENSIONAL FORM
outerforcex = -inner(Constant((1.0, 0.0)), omegaf1)*ds(1)
outerforcey = -inner(Constant((0.0, 1.0)), omegaf1)*ds(1)
innertorque = -r_a*inner(n0, sigma0)*ds(0)
outertorque = -r_a*inner(n1, sigma1)*ds(1)



# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Use nonzero guesses - essential for CG with non-symmetric BC
parameters['krylov_solver']['nonzero_initial_guess'] = True

# Time-stepping
t = 0.0

# Time/Torque Plot
x=list()
y=list()
z=list()
zz=list()
m=list()
n=list()
l=list()

# Time-stepping
iter = 0            # iteration counter
maxiter = 100
while t < Tf + DOLFIN_EPS and iter < maxiter:
    
    print("t =", t)
    print("iteration", iter)
    iter += 1

    if norm(p1.vector(),'linf') > 10E10:
       print('FEM Solution diverging')
       quit()
    # Compute tentative velocity step
    #begin("Computing tentative velocity")
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1, "bicgstab", "default")
    end()

    # Pressure correction
    # begin("Computing pressure correction")
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcp]
    [bc.apply(p1.vector()) for bc in bcp]
    solve(A2, p1.vector(), b2, "bicgstab", prec)
    end()

    # Velocity correction
    # begin("Computing velocity correction")
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u1.vector(), b3, "bicgstab", "default")
    end()


    # Temperature correction
    #begin("Computing temperature correction")
    b5 = assemble(L5)
    [bc.apply(A5, b5) for bc in bcT]
    solve(A5, T1.vector(), b5, "bicgstab", "default")
    end()

    # Calcultate Force on the bearing
    print("Normal Force on inner bearing: ",  (assemble(innerforcex), assemble(innerforcey)))
    print("Torque on inner bearing: ", assemble(innertorque))

    # Evaluate Pressure at a random point
    print("Pressure", p1.vector().array()[50])
 
    # Record Torque Data 
    x.append(t)
    y.append(assemble(innertorque))
    z.append(assemble(innerforcey))
    zz.append(assemble(innerforcex))
    l.append(norm(u1.vector()-u0.vector()))

    #Update Velocity Boundary Condition
    w.t = t

    # Plot solution
    #plot(p1, title="Pressure", rescale=True)
    #plot(u1, title="Velocity", rescale=True, mode = "auto")
    #plot(T1, title="Temperature", rescale=True)

    # Move to next time step
    u0.assign(u1)
    T0.assign(T1)
    t += dt

# Dimensionalise pressure for plotting

# Matlab Plot of the Solution
mplot(p1)
plt.show()
mplot(T1)
plt.show()

"""# Plot Torque Data
plt.plot(x, y, 'r-', label='Torque')
plt.show()

plt.plot(x, z, 'b-', label='Vertical Force')
plt.show()"""

#quit()

#Plot PRESSURE Contours USING MATPLOTLIB
# Scalar Function code

#Set Values for inner domain as -infty

x = Expression('x[0]', degree=d)  #GET X-COORDINATES LIST
y = Expression('x[1]', degree=d)  #GET Y-COORDINATES LIST
meshc= generate_mesh(c3, 35)
Q1=FunctionSpace(meshc, "CG", 2)
pj=Expression('0') #Expression for the 'pressure' in the domian
pjq=interpolate(pj, Q1)
pjvals=pjq.vector().array()

xyvals=meshc.coordinates()
xqalsv = interpolate(x, Q1)
yqalsv= interpolate(y, Q1)

xqals= xqalsv.vector().array()
yqals= yqalsv.vector().array()

pvals = p1.vector().array() # GET SOLUTION u= u(x,y) list
xyvals = mesh.coordinates()     # CLEAN THIS UP!!!!!!!!!!!!!!!!!!!!!!!!!!!!
xvalsv = interpolate(x, Q)#xyvals[:,0]
yvalsv= interpolate(y, Q)#xyvals[:,1]

xvals = xvalsv.vector().array()
yvals = yvalsv.vector().array()

pvals = np.concatenate([pvals, pjvals])  #Merge two arrays for pressure values
xvals = np.concatenate([xvals, xqals])   #Merge two arrays for x-coordinate values
yvals = np.concatenate([yvals, yqals])   #Merge two arrays for y-coordinate values

xx = np.linspace(-1.5*r_b,1.5*r_b, num=250)
yy = np.linspace(-1.5*r_b,1.5*r_b, num=250)
XX, YY = np.meshgrid(xx,yy)   # (x,y) coordinate data formatted so that it can be used by plt.contour()
uu = mlab.griddata(xvals, yvals, pvals, xx, yy, interp='nn') # u(x,y) data so that it can be used by 

plt.contour(XX, YY, uu, 50)
plt.show()


#Plot TEMPERATURE Contours USING MATPLOTLIB
# Scalar Function code

#Set Values for inner domain as ZERO


Tj=Expression('0') #Expression for the 'pressure' in the domian
Tjq=interpolate(Tj, Q1)
Tjvals=Tjq.vector().array()

Tvals = T1.vector().array() # GET SOLUTION T= T(x,y) list
Tvals = np.concatenate([Tvals, Tjvals])  #Merge two arrays for Temperature values

TT = mlab.griddata(xvals, yvals, Tvals, xx, yy, interp='nn') # u(x,y) data so that it can be used by 

plt.contour(XX, YY, TT, 25)
plt.show()

quit()

# Plot Velocity Field Contours (MATPLOTLIB)

#Set Velocity in the Journal (inner circle) to -1000000 and get in 
V1=VectorFunctionSpace(meshc, "CG", d)
vj=Expression(('0','0')) #Expression for the 'pressure' in the domian
vjq=interpolate(vj, V1)
uvjvals=vjq.vector().array()
n1= meshc.num_vertices()
xy = Expression(('x[0]','x[1]'))  #GET MESH COORDINATES LIST
gg=list()
hh=list()
for i in range(len(u1.vector().array())/2):  
    gg.append(uvjvals[2*i+1])
    hh.append(uvjvals[2*i])
ujvals=np.asarray(hh)
vjvals=np.asarray(gg)

xyvalsv = interpolate(xy, V1)

qq=list()
rr=list()
print(xyvalsv.vector().array())
for i in range(len(u1.vector().array())/2):  
    qq.append(xyvalsv.vector().array()[2*i+1])
    rr.append(xyvalsv.vector().array()[2*i])

xvalsj = np.asarray(rr)
yvalsj = np.asarray(qq)






g=list()
h=list()
n= mesh.num_vertices()
print(u1.vector().array())               # u is the FEM SOLUTION VECTOR IN FUNCTION SPACE 
for i in range(len(u1.vector().array())/2):                     # Length of vector Depends on Degree of the Elements 
    g.append(u1.vector().array()[2*i+1])
    h.append(u1.vector().array()[2*i])

uvals = np.asarray(h)                    # GET SOLUTION (u,v) -> u= u(x,y) list
vvals = np.asarray(g)                    # GET SOLUTION (u,v) -> v= v(x,y) list



xy = Expression(('x[0]','x[1]'))  #GET MESH COORDINATES LIST
xyvalsv = interpolate(xy, V)

q=list()
r=list()

for i in range(len(u1.vector().array())/2):
   q.append(xyvalsv.vector().array()[2*i+1])
   r.append(xyvalsv.vector().array()[2*i])

xvals = np.asarray(r)
yvals = np.asarray(q)

#Determine Speed 



#Merge arrays
uvals = np.concatenate([uvals, ujvals])  #Merge two arrays for velocity values
vvals = np.concatenate([vvals, vjvals])  #Merge two arrays for velocity values
xvals = np.concatenate([xvals, xvalsj])   #Merge two arrays for x-coordinate values
yvals = np.concatenate([yvals, yvalsj])   #Merge two arrays for y-coordinate values


uu = mlab.griddata(xvals, yvals, uvals, xx, yy, interp='nn') 
vv = mlab.griddata(xvals, yvals, vvals, xx, yy, interp='nn') 

plot3 = plt.figure()
plt.streamplot(XX, YY, uu, vv,                
               cmap=cm.cool,        # colour map
               linewidth=2,         # line thickness
               arrowstyle='->',     # arrow style
               arrowsize=0.1)       # arrow size

#plt.colorbar()                      # add colour bar on the right

plt.title('Stream Plot, Dynamic Colour')

plt.show(plot3)                                                                      # display the plot


# Hold plot
plot(mesh, interactive=True)


