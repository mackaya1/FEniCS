"""Base Code for the Numerical Scheme Test"""

from decimal import *
from dolfin import *
from mshr import *
from math import pi, sin, cos, sqrt, fabs
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

# Define Geometry
B=1
L=1
x_0 = 0
y_0 = 0
x_1 = B
y_1 = L

# Mesh refinement comparison Loop

""" mesh refinemment prescribed in code"""
mm = 50
 
nx=mm*B
ny=mm*L

c = min(x_1-x_0,y_1-y_0)
base_mesh= RectangleMesh(Point(x_0,y_0), Point(x_1, y_1), nx, ny)


# Create Unstructured mesh

u_rec=Rectangle(Point(0.0,0.0),Point(1.0,1.0))
mesh0=generate_mesh(u_rec, mm)



#SKEW MESH FUNCTION

# MESH CONSTRUCTION CODE

nv= base_mesh.num_vertices()
nc= base_mesh.num_cells()
coorX = base_mesh.coordinates()[:,0]
coorY = base_mesh.coordinates()[:,1]
cells0 = base_mesh.cells()[:,0]
cells1 = base_mesh.cells()[:,1]
cells2 = base_mesh.cells()[:,2]


# Skew Mapping EXPONENTIAL
N=4.0
def expskewcavity(x,y):
    xi = 0.5*(1.0+np.tanh(2*N*(x-0.5)))
    ups= 0.5*(1.0+np.tanh(2*N*(y-0.5)))
    return(xi,ups)

# Skew Mapping
pi=3.14159265359

def skewcavity(x,y):
    xi = 0.5*(1.0-np.cos(x*pi))**1
    ups =0.5*(1.0-np.cos(y*pi))**1
    return(xi,ups)

# OLD MESH COORDINATES -> NEW MESH COORDINATES
r=list()
l=list()
for i in range(nv):
    r.append(skewcavity(coorX[i], coorY[i])[0])
    l.append(skewcavity(coorX[i], coorY[i])[1])

r=np.asarray(r)
l=np.asarray(l)

# MESH GENERATION (Using Mesheditor)
mesh1 = Mesh()
editor = MeshEditor()
editor.open(mesh1,2,2)
editor.init_vertices(nv)
editor.init_cells(nc)
for i in range(nv):
    editor.add_vertex(i, r[i], l[i])
for i in range(nc):
    editor.add_cell(i, cells0[i], cells1[i], cells2[i])
editor.close()

# Mesh Refine Code (UNSTRUCTURED MESH)

for i in range(2):
      g = (max(x_1,y_1)-max(x_0,y_0))*0.05/(i+1)
      cell_domains = CellFunction("bool", mesh0)
      cell_domains.set_all(False)
      for cell in cells(mesh0):
          x = cell.midpoint()
          if  (x[0] < x_0+g or x[1] < y_0+g) or (x[0] > x_1-g or x[1] > y_1-g): # or (x[0] < x0+g and x[1] < y0+g)  or (x[0] > x1-g and x[1] < g): 
              cell_domains[cell]=True
      #plot(cell_domains, interactive=True)
      mesh0 = refine(mesh0, cell_domains, redistribute=True)

#plot(mesh0)
#plot(mesh)
#plot(mesh1,interactive=True)

#mplot(mesh0)
#plt.savefig("fine_unstructured_grid.png")
#plt.clf() 
#mplot(mesh)
#plt.savefig("fine_structured_grid.png")
#plt.clf() 
#mplot(mesh1)
#plt.savefig("fine_skewed_grid.png")
#plt.clf()
#quit()

# Choose Mesh to Use

mesh = mesh1

#Define Boundaries 

top_bound = 0.5*(1+tanh(N)) 

class No_slip(SubDomain):
      def inside(self, x, on_boundary):
          return True if on_boundary else False 
                                                                          
class Lid(SubDomain):
      def inside(self, x, on_boundary):
          return True if x[1] > L*(top_bound - DOLFIN_EPS) and on_boundary  else False   

no_slip = No_slip()
lid = Lid()


# MARK SUBDOMAINS (Create mesh functions over the cell facets)
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
sub_domains.set_all(5)
no_slip.mark(sub_domains, 0)
lid.mark(sub_domains, 2)


plot(sub_domains, interactive=False)        # DO NOT USE WITH RAVEN
#quit()

#Define Boundary Parts
boundary_parts = FacetFunction("size_t", mesh)
no_slip.mark(boundary_parts,0)
lid.mark(boundary_parts,1)
ds = Measure("ds")[boundary_parts]

# Define function spaces (P2-P1)

# Discretization  parameters
family = "CG"; dfamily = "DG"; rich = "Bubble"
shape = "triangle"; order = 2

#mesh.ufl_cell()

V_s = VectorElement(family, mesh.ufl_cell(), order)       # Elements
Z_c = VectorElement(family, mesh.ufl_cell(),  order, 3)
Z_s = VectorElement(dfamily, mesh.ufl_cell(),  order-1, 3)
Z_se = VectorElement(rich, mesh.ufl_cell(),  order+1, 3)
Q_s = FiniteElement(family, mesh.ufl_cell(), order-1)
Q_p = FiniteElement(rich, mesh.ufl_cell(), order+1, 3)
Z_e = EnrichedElement(Z_c,Z_se)                 # Enriched Elements
Q_rich = EnrichedElement(Q_s,Q_p)


W = FunctionSpace(mesh,V_s*Z_s)             # F.E. Spaces 
V = FunctionSpace(mesh,V_s)

Z = FunctionSpace(mesh,Z_s)
Ze = FunctionSpace(mesh,Z_e)
Zc = FunctionSpace(mesh,Z_c)
Q = FunctionSpace(mesh,Q_s)
Qt = FunctionSpace(mesh, "DG", order-2)
Qr = FunctionSpace(mesh,Q_s)

# Define trial and test functions [TAYLOR GALERKIN Method]
rho=TrialFunction(Q)
p = TrialFunction(Q)
T = TrialFunction(Q)
q = TestFunction(Q)
r = TestFunction(Q)

p0=Function(Q)       # Pressure Field t=t^n
p1=Function(Q)       # Pressure Field t=t^n+1
rho0=Function(Q)
rho1=Function(Q)
T0=Function(Qt)       # Temperature Field t=t^n
T1=Function(Qt)       # Temperature Field t=t^n+1


(v, R_vec) = TestFunctions(W)
(u, D_vec) = TrialFunctions(W)

tau_vec = TrialFunction(Zc)
Rt_vec = TestFunction(Zc)


tau0_vec=Function(Zc)     # Stress Field (Vector) t=t^n
tau12_vec=Function(Zc)    # Stress Field (Vector) t=t^n+1/2
tau1_vec=Function(Zc)     # Stress Field (Vector) t=t^n+1

w0= Function(W)
w12= Function(W)
ws= Function(W)
w1= Function(W)

(u0, D0_vec) = w0.split()
(u12, D12_vec) = w12.split()
(u1, D1_vec) = w1.split()
(us, Ds_vec) = ws.split()


I = Expression((('1.0','0.0'),
                ('0.0','1.0')), degree=2)


# Some Useful Functions
def  tgrad (w):
    """ Returns  transpose  gradient """
    return  transpose(grad(w))
def Dincomp (w):
    """ Returns 2* the  rate of  strain  tensor """
    return (grad(w) + tgrad(w))/2
def Dcomp (w):
    """ Returns 2* the  rate of  strain  tensor """
    return ((grad(w) + tgrad(w))-(2.0/3)*div(w)*I)/2

def normalize_solution(u):
    "Normalize u: return u divided by max(u)"
    u_array = u.vector().array()
    u_max = np.max(np.abs(u_array))
    u_array /= u_max
    u.vector()[:] = u_array
    #u.vector().set_local(u_array)  # alternative
    return u

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

# The  projected  rate -of-strain
D_proj_vec = Function(Ze)
D_proj = as_matrix([[D_proj_vec[0], D_proj_vec[1]],
                    [D_proj_vec[1], D_proj_vec[2]]])

"""I_vec = Expression(('1.0','0.0','1.0'), degree=2)
I_vec = interpolate(I_vec, Z)

I_matrix = as_matrix([[I_vec[0], I_vec[1]],
                      [I_vec[1], I_vec[2]]])"""



# Project Vector Trial Functions of Stress onto SYMMETRIC Tensor Space

D =  as_matrix([[D_vec[0], D_vec[1]],
                [D_vec[1], D_vec[2]]])

tau = as_matrix([[tau_vec[0], tau_vec[1]],
                 [tau_vec[1], tau_vec[2]]])  

# Project Vector Test Functions of Stress onto SYMMETRIC Tensor Space

Rt = as_matrix([[Rt_vec[0], Rt_vec[1]],
                 [Rt_vec[1], Rt_vec[2]]])        # DEVSS Space

R = as_matrix([[R_vec[0], R_vec[1]],
                 [R_vec[1], R_vec[2]]])

# Project Vector Functions of Stress onto SYMMETRIC Tensor Space

D0 = as_matrix([[D0_vec[0], D0_vec[1]],
                [D0_vec[1], D0_vec[2]]])        #DEVSS STABILISATION

D12 = as_matrix([[D12_vec[0], D12_vec[1]],
                [D12_vec[1], D12_vec[2]]])

Ds = as_matrix([[Ds_vec[0], Ds_vec[1]],
                [Ds_vec[1], Ds_vec[2]]])


D1 = as_matrix([[D1_vec[0], D1_vec[1]],
                [D1_vec[1], D1_vec[2]]]) 


tau0 = as_matrix([[tau0_vec[0], tau0_vec[1]],
                  [tau0_vec[1], tau0_vec[2]]])        # Stress 

tau12 = as_matrix([[tau12_vec[0], tau12_vec[1]],
                   [tau12_vec[1], tau12_vec[2]]]) 

tau1 = as_matrix([[tau1_vec[0], tau1_vec[1]],
                  [tau1_vec[1], tau1_vec[2]]])   





# Default Nondimensional Parameters
conv=0
U=1.0
betav = 0.5     
Re = 1                             #Reynolds Number
We = 0.5                           #Weisenberg NUmber
Di = 0.005                         #Diffusion Number
Vh = 0.005
T_0 = 300
T_h = 350
Bi = 0.2
c0 = 1500
Ma = U/c0 
rho_0 = 1.0




# Define boundary/stabilisation FUNCTIONS
td= Constant('5')
e = Constant('6')
ulidreg=Expression(('8*(1.0+tanh(8*t-4.0))*(x[0]*(L-x[0]))*(x[0]*(L-x[0]))','0'), degree=2, t=0.0, L=L, e=e, T_0=T_0, T_h=T_h) # Lid Speed 
ulid=Expression(('0.5*(1.0+tanh(8*t-4.0))','0'), degree=2, t=0.0, T_0=T_0, T_h=T_h) # Lid Speed 
T_bl = Expression('T_0+(T_h-T_0)*sin((x[0]+x[1])*2*pi/L)', degree=2, T_0=T_0, T_h=T_h, pi=pi, L=L, B=B)
T_bb = Expression('T_0+(T_h-T_0)*sin((x[0]+x[1])*2*pi/B)', degree=2, T_0=T_0, T_h=T_h, pi=pi, L=L, B=B)
h_sk = Expression('cos(pi*x[0])-cos(pi*(x[0]+1/mm))','cos(pi*x[1])-cos(pi*(x[1]+1/mm))', degree=2, pi=pi, mm=mm, L=L, B=B)             # Mesh size function
h_k = Expression(('1/mm','1/mm'), degree=2, mm=mm, L=L, B=B)
h_m = Expression('0.5*h', degree=2, h=mesh.hmin())
h_ka = Expression('0.5*1/mm', degree=2, mm=mm, L=L, B=B)
h_ska= Expression('0.5*(cos(pi*x[0])-cos(pi*(x[0]+1/mm))+cos(0.5*pi*x[1])-cos(0.5*pi*(x[1]+1/mm)))', degree=2, pi=pi, mm=mm, L=L, B=B)
rampd=Expression('0.5*(1+tanh(8*(2.0-t)))', degree=2, t=0.0)
rampu=Expression('0.5*(1+tanh(16*(t-2.0)))', degree=2, t=0.0)


# Interpolate Stabilisation Functions
h = CellSize(mesh)
h_k = project(h/mesh.hmax(), Qt)


# Default Stabilisation Parameters
alph = 0
th = 0                  # DEVSS
c1 = 0*h_ska            # SUPG / SU
c2 = 0                  # Artificial Diffusion

# Define unit Normal/tangent Vector at inner and outer Boundary (Method 2)
n0 =  Expression(('-1' , '0'), degree=2)
n1 =  Expression(('0' , '1' ), degree=2)
n2 =  Expression(('1' , '0' ), degree=2)
n3 =  Expression(('0' , '-1'), degree=2)



# Dirichlet Boundary Conditions  (LID DRIVEN CAVITY)
noslip  = DirichletBC(W.sub(0), Constant((0.0, 0.0)), no_slip)  # No Slip boundary conditions on the left wall
drive1  =  DirichletBC(W.sub(0), ulidreg, lid)  # No Slip boundary conditions on the upper wall
#slip  = DirichletBC(V, sl, omega0)  # Slip boundary conditions on the second part of the flow wall 
#temp0 =  DirichletBC(Qt, T_0, omega0)    #Temperature on Omega0 
#temp2 =  DirichletBC(Qt, T_0, omega2)    #Temperature on Omega2 
#temp3 =  DirichletBC(Qt, T_0, omega3)    #Temperature on Omega3 
#Collect Boundary Conditions
bcu = [noslip, drive1]
bcp = []
bcT = []    #temp0, temp2
bctau = []

# Log Conformation Tensor 




