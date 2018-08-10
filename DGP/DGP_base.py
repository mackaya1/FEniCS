"""Base Code for the Finite Element solution of the Lid Driven Cavity Flow"""

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
import time 


# Some Useful Functions
def mesh2triang(mesh):
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells()) # Mesh Diagram

def mplot(obj): 
    """Plots DOLFIN functions/meshes using matplotlib"""                    # Function Plot
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


def  tgrad (w):
    """ Returns  transpose  gradient """
    return  transpose(grad(w))
def Dincomp (w):
    """ Returns 2* the  rate of  strain  tensor """
    return (grad(w) + tgrad(w))/2
def Dcomp (w):
    """ Returns 2* the  rate of  strain  tensor """
    return ((grad(w) + tgrad(w))-(2.0/3)*div(w)*Identity(len(u)))/2

def sigma(u, p, Tau):
    return 2*Pr*betav*Dincomp(u) - p*Identity(len(u)) + Pr*((1-betav)/(We))*(Tau-Identity(len(u)))

def sigmacom(u, p, Tau):
    return 2*Pr*betav*Dcomp(u) - p*Identity(len(u)) + Pr*((1-betav)/(We))*(Tau-Identity(len(u)))

def Fdef(u, Tau):
    return dot(u,grad(Tau)) - dot(grad(u),Tau) - dot(Tau,tgrad(u))

def Fdefcom(u, Tau):
    return dot(u,grad(Tau)) - dot(grad(u),Tau) - dot(Tau,tgrad(u)) + div(u)*Tau 

def normalize_solution(u):
    "Normalize u: return u divided by max(u)"
    u_array = u.vector().array()
    u_max = np.max(np.abs(u_array))
    u_array /= u_max
    u.vector()[:] = u_array
    #u.vector().set_local(u_array)  # alternative
    return u

def magnitude(u):
    return np.power((u[0]*u[0]+u[1]*u[1]), 0.5)

def absolute(u):
    u_array = np.absolute(u.vector().get_local())
    u.vector()[:] = u_array
    return u


def stream_function(u):
    '''Compute stream function of given 2-d velocity vector.'''
    V = u.function_space().sub(0).collapse()

    if V.mesh().topology().dim() != 2:
        raise ValueError("Only stream function in 2D can be computed.")

    psi = TrialFunction(V)
    phi = TestFunction(V)

    a = inner(grad(psi), grad(phi))*dx
    L = inner(u[1].dx(0) - u[0].dx(1), phi)*dx
    bc = DirichletBC(V, Constant(0.), DomainBoundary())

    A, b = assemble_system(a, L, bc)
    psi = Function(V)
    solve(A, psi.vector(), b)

    return psi

def comp_stream_function(rho, u):
    '''Compute stream function of given 2-d velocity vector.'''
    V = u.function_space().sub(0).collapse()

    if V.mesh().topology().dim() != 2:
        raise ValueError("Only stream function in 2D can be computed.")

    psi = TrialFunction(V)
    phi = TestFunction(V)

    a = inner(grad(psi), grad(phi))*dx
    L = inner(rho*u[1].dx(0) - rho*u[0].dx(1), phi)*dx
    bc = DirichletBC(V, Constant(0.), DomainBoundary())

    A, b = assemble_system(a, L, bc)
    psi = Function(V)
    solve(A, psi.vector(), b)

    return psi


def min_location(u):

    V = u.function_space()

    if V.mesh().topology().dim() != 2:
       raise ValueError("Only minimum of scalar function in 2D can be computed.")

    dofs_x = V.tabulate_dof_coordinates().reshape((-1, gdim))

    function_array = u.vector().get_local()
    minimum = min(u.vector().get_local())

    min_index = np.where(function_array == minimum)
    min_loc = dofs_x[min_index]

    return min_loc

def l2norm_solution(u):
    u_array = u.vector().array()
    u_l2 = norm(u, 'L2')
    u_array /= u_l2
    u.vector()[:] = u_array
    return u
    

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]




def DGP_unstructured_mesh(mm):
    u_rec=Rectangle(Point(0.0,0.0),Point(1.0,1.0))
    mesh0=generate_mesh(u_rec, mm)

    return mesh0




def expskewcavity(x,y,N):
    """exponential Skew Mapping"""
    xi = 0.5*(1+np.tanh(2*N*(x-0.5)))
    ups= 0.5*(1+np.tanh(2*N*(y-0.5)))
    return(xi,ups)

# Skew Mapping
pi=3.14159265359

def skewcavity(x,y):
    xi = 0.5*(1-np.cos(x*pi))**1
    ups =0.5*(1-np.cos(y*pi))**1
    return(xi,ups)

def xskewcavity(x,y):
    xi = 0.5*(1-np.cos(x*pi))**1
    ups = y
    return(xi,ups)


def DGP_structured_mesh(mm):
    nx=mm*B
    ny=mm*L
    base_mesh= RectangleMesh(Point(x_0,y_0), Point(x_1, y_1), nx, ny)

    nv= base_mesh.num_vertices()
    nc= base_mesh.num_cells()
    coorX = base_mesh.coordinates()[:,0]
    coorY = base_mesh.coordinates()[:,1]
    cells0 = base_mesh.cells()[:,0]
    cells1 = base_mesh.cells()[:,1]
    cells2 = base_mesh.cells()[:,2]

    # OLD MESH COORDINATES -> NEW MESH COORDINATES
    r=list()
    l=list()
    for i in range(nv):
      r.append(xskewcavity(coorX[i], coorY[i])[0])
      l.append(xskewcavity(coorX[i], coorY[i])[1])

      r=np.asarray(r)
      l=np.asarray(l)

    # MESH GENERATION (Using Mesheditor)
    mesh1 = Mesh()
    editor = MeshEditor()
    editor.open(mesh1, "triangle", 2,2)
    editor.init_vertices(nv)
    editor.init_cells(nc)
    for i in range(nv):
        editor.add_vertex(i, r[i], l[i])
    for i in range(nc):
        editor.add_cell(i, cells0[i], cells1[i], cells2[i])
    editor.close()
    
    return mesh1

# Mesh Refine Code (UNSTRUCTURED MESH)

def refine_boundary(mesh, times):
    for i in range(times):
          g = (max(x_1,y_1)-max(x_0,y_0))*0.025/(i+1)
          cell_domains = CellFunction("bool", mesh)
          cell_domains.set_all(False)
          for cell in cells(mesh):
              x = cell.midpoint()
              if  (x[0] < x_0+g or x[1] < y_0+g) or (x[0] > x_1-g or x[1] > y_1-g): 
                  cell_domains[cell]=True

          mesh = refine(mesh, cell_domains, redistribute=True)
    return mesh

def refine_top(mesh, times):
    for i in range(times):
          g = (max(x_1,y_1)-max(x_0,y_0))*0.025/(i+1)
          cell_domains = CellFunction("bool", mesh)
          cell_domains.set_all(False)
          for cell in cells(mesh):
              x = cell.midpoint()
              if  x[1] > y_1-g:
                  cell_domains[cell]=True
          mesh_refine = refine(mesh, cell_domains, redistribute=True)
    return mesh_refine

def refine_walls(mesh, times):
    for i in range(times):
          g = (max(x_1,y_1)-max(x_0,y_0))*0.05/(i+1)
          cell_domains = CellFunction("bool", mesh)
          cell_domains.set_all(False)
          for cell in cells(mesh):
              x = cell.midpoint()
              if  x[0] > x_1 - g or x[0] < x_0 + g:
                  cell_domains[cell]=True 
          mesh_refine = refine(mesh, cell_domains, redistribute=True)
    return mesh_refine


# Define Geometry/Mesh
B=1
L=1
x_0 = 0
y_0 = 0
x_1 = 1
y_1 = 1

# Mesh refinement comparison Loop
mm = 48
 


# Choose Mesh to Use

mesh = DGP_structured_mesh(mm)
gdim = mesh.geometry().dim() # Mesh Geometry

#mesh = refine_walls(mesh, 2)

mplot(mesh)
plt.savefig("fine_skewed_grid.png")
plt.clf()
plt.close()
#quit()

#Define Boundaries 

bottom_bound = 0.5*(1+tanh(-N)) 
top_bound = 0.5*(1+tanh(N)) 

class No_slip(SubDomain):
      def inside(self, x, on_boundary):
          return True if on_boundary else False 
                                                                          
class Left(SubDomain):
      def inside(self, x, on_boundary):
          return True if x[0] < bottom_bound + DOLFIN_EPS and on_boundary  else False  

class Right(SubDomain):
      def inside(self, x, on_boundary):
          return True if x[0] > top_bound - DOLFIN_EPS and on_boundary  else False   

class Top(SubDomain):
      def inside(self, x, on_boundary):
          return True if x[1] > top_bound - DOLFIN_EPS and on_boundary  else False  

no_slip = No_slip()
left = Left()
right = Right()
top = Top()


# MARK SUBDOMAINS (Create mesh functions over the cell facets)
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
sub_domains.set_all(5)
no_slip.mark(sub_domains, 0)
left.mark(sub_domains, 2)
right.mark(sub_domains, 3)
top.mark(sub_domains, 4)


plot(sub_domains, interactive=False)        # DO NOT USE WITH RAVEN
#quit()

#Define Boundary Parts

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1) #FacetFunction("size_t", mesh)
no_slip.mark(boundary_parts,0)
left.mark(boundary_parts,1)
right.mark(boundary_parts,2)
top.mark(boundary_parts,3)
ds = Measure("ds")[boundary_parts]

# Define function spaces (P2-P1)

# Discretization  parameters
family = "CG"; dfamily = "DG"; rich = "Bubble"
shape = "triangle"; order = 2

#mesh.ufl_cell()

V_s = VectorElement(family, mesh.ufl_cell(), order)       # Elements
Z_c = VectorElement(family, mesh.ufl_cell(),  order, 3)
Z_s = VectorElement(dfamily, mesh.ufl_cell(),  order-1, 3)
Z_d = VectorElement(dfamily, mesh.ufl_cell(),  order-2, 3)
Z_se = VectorElement(rich, mesh.ufl_cell(),  order+1, 3)
Q_s = FiniteElement(family, mesh.ufl_cell(), order-1)
Q_p = FiniteElement(rich, mesh.ufl_cell(), order+1, 3)
Z_e = EnrichedElement(Z_c,Z_se)                 # Enriched Elements
Q_rich = EnrichedElement(Q_s,Q_p)


W = FunctionSpace(mesh,V_s*Z_s)             # F.E. Spaces 
V = FunctionSpace(mesh,V_s)

Z = FunctionSpace(mesh,Z_s)
Zd = FunctionSpace(mesh,Z_d)
#Ze = FunctionSpace(mesh,Z_e)
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
T0=Function(Q)       # Temperature Field t=t^n
T1=Function(Q)       # Temperature Field t=t^n+1


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


# Initial Conformation Tensor
I_vec = Expression(('1.0','0.0','1.0'), degree=2)
initial_guess_conform = project(I_vec, Zc)




# The  projected  rate -of-strain
D_proj_vec = Function(Zc)
D_proj = as_matrix([[D_proj_vec[0], D_proj_vec[1]],
                    [D_proj_vec[1], D_proj_vec[2]]])





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
U = 1
betav = 0.5     
Ra = 1000                           #Rayleigh Number
Pr = 1
We = 0.5                           #Weisenberg NUmber
Di = 0.005                         #Diffusion Number
Vh = 0.005
T_0 = 300
T_h = 350
Bi = 0.2
c0 = 0
Ma = 0 




# Define boundary/stabilisation FUNCTIONS
td= Constant('5')
e = Constant('6')
T_bl = Expression('T_0+(T_h-T_0)*sin((x[0]+x[1])*2*pi/L)', degree=2, T_0=T_0, T_h=T_h, pi=pi, L=L, B=B)
T_bb = Expression('T_0+(T_h-T_0)*sin((x[0]+x[1])*2*pi/B)', degree=2, T_0=T_0, T_h=T_h, pi=pi, L=L, B=B)
rampd = Expression('0.5*(1+tanh(8*(2.0-t)))', degree=2, t=0.0)
rampu = Expression('0.5*(1+tanh(16*(t-2.0)))', degree=2, t=0.0)
ramped_T = Expression('0.5*(1+tanh(8*(t-0.5)))*(T_h-T_0)+T_0', degree=2, t=0.0, T_0=T_0, T_h=T_h)
f = Expression(('0','-1'), degree=2)


# Interpolate Stabilisation Functions
h = CellDiameter(mesh)
n = FacetNormal(mesh)


# Default Stabilisation Parameters
alph = 0
th = 0                  # DEVSS
c1 = 0*h                # SUPG / SU
c2 = 0                  # Artificial Diffusion

# Define unit Normal/tangent Vector at inner and outer Boundary (Method 2)
n0 =  Expression(('-1' , '0'), degree=2)
n1 =  Expression(('0' , '1' ), degree=2)
n2 =  Expression(('1' , '0' ), degree=2)
n3 =  Expression(('0' , '-1'), degree=2)



# Dirichlet Boundary Conditions  (LID DRIVEN CAVITY)
noslip0  = DirichletBC(W.sub(0), Constant((0.0, 0.0)), no_slip)  # No Slip boundary conditions on the left wall
noslip1 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), left)  # No Slip boundary conditions on the left wall
noslip2 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), right)  # No Slip boundary conditions on the left wall
noslip3 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), top)  # No Slip boundary conditions on the left wall
temp_left =  DirichletBC(Q, ramped_T, left)    #Temperature on Omega0 
temp_right =  DirichletBC(Q, T_0, right)    #Temperature on Omega2 

#Collect Boundary Conditions
bcu = [noslip0, noslip1, noslip2, noslip3]
bcp = []
bcT = [temp_left, temp_right]    #temp0, temp2
bctau = []

