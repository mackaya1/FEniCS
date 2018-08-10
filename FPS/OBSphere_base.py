"""Base Code for the Finite Element solution of the FLOW PAST A Sphere of an OLDROD B FLUID"""

"""Code Note Complete"""


from decimal import *
from dolfin import *
from mshr import *
from math import pi, sin, cos, sqrt, fabs
import numpy as np
import matplotlib
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
        plt.axis('on')
    elif isinstance(obj, Mesh):
        if (obj.geometry().dim() != 2):
            raise(AttributeError)
        plt.triplot(mesh2triang(obj), color='k', linewidth = 0.2)
        plt.axis('off')

def sphere_mesh(mm, r_a):

    c0 = Circle(Point(r_x,r_y), r_a, 256) # Create Circle

    box0 = Rectangle(Point(x_0, y_0), Point(x_1, y_1)) #Create Box

    dist = min(x_1-x_0,y_1-y_0)
    c = dist - r_a

    if c <= 0.0:
       print("ERROR! SPHERE radius greater than box Diameter")
       quit()



    # Create Geometry
    geom = box0-c0

    mesh = generate_mesh(geom, mm)
    return mesh

def restricted_sphere_mesh(mm, r_a):

    c0 = Circle(Point(r_x,r_y), r_a, 256) # Create Circle

    box0 = Rectangle(Point(-3.0, y_0), Point(4.0, y_1)) #Create Box

    dist = min(x_1-x_0,y_1-y_0)
    c = dist - r_a

    if c <= 0.0:
       print("ERROR! SPHERE radius greater than box Diameter")
       quit()

    # Create Geometry
    geom = box0-c0

    mesh = generate_mesh(geom, mm)
    return mesh

# Mesh Refinement 
def refine_bottom(mesh, times):
    for i in range(times):
          g = (max(x_1,y_1)-max(x_0,y_0))/(i+1)
          cell_domains = MeshFunction("bool", mesh, 2)
          cell_domains.set_all(False)
          for cell in cells(mesh):
              x = cell.midpoint()
              if  ((x[0]-r_x)**2+(x[1]-r_y)**2) < r_a**2+2.0*g or x[1] < y_0 + 0.025*g or x[0] < x_0 + 0.025*g:
                  cell_domains[cell]=True

          mesh = refine(mesh, cell_domains, redistribute=True)
    return mesh

# Adaptive Mesh Refinement 
def adaptive_refinement(mesh, kapp, ratio):
    kapp_array = kapp.vector().get_local()
    kapp_level = np.percentile(kapp_array, (1-ratio)*100)

    cell_domains = MeshFunction("bool", mesh, 2)
    cell_domains.set_all(False)
    for cell in cells(mesh):
        x = cell.midpoint()
        if  kapp([x[0], x[1]]) > kapp_level:
            cell_domains[cell]=True

    mesh = refine(mesh, cell_domains, redistribute=True)
    return mesh

# BOUNDARIES

#Top                                                                              
class No_Slip(SubDomain):
      def inside(self, x, on_boundary):
          return True if on_boundary and near(x[1], y_1) else False  
no_slip = No_Slip()

# Sphere
class Sphere(SubDomain):
      def inside(self, x, on_boundary):
          return True if (x[0]-r_x)**2 + (x[1]-r_y)**2 < r_a**2 + DOLFIN_EPS and on_boundary  else False  
sphere = Sphere()

# Bottom 
class Bottom(SubDomain):
      def inside(self, x, on_boundary):
          return True if x[1] < y_0 + DOLFIN_EPS and   on_boundary  else False  
bottom = Bottom()

# Left Wall of Box 
class Left_Wall(SubDomain):
      def inside(self, x, on_boundary):
          return True if near(x[0],x_0) and on_boundary else False 
left_wall = Left_Wall()


# Right Wall of Box 
class Right_Wall(SubDomain):
      def inside(self, x, on_boundary):
          return True if near(x[0],x_1) and on_boundary else False 
right_wall = Right_Wall()

# Some Useful Functions
def  tgrad (w):
    """ Returns  transpose  gradient """
    return  transpose(grad(w))
def Dincomp (w):
    """ Returns 2* the  rate of  strain  tensor """
    return (grad(w) + tgrad(w))/2
def Dcomp (w):
    """ Returns 2* the  rate of  strain  tensor """
    return ((grad(w) + tgrad(w))-(2.0/3)*div(w)*Identity(len(u)))/2

def Tr(tau):
    return abs(tau[0,0] + tau[1,1])

def fene_func(tau, b):
    f_tau = 1.0/(1.-(Tr(tau)-2.)/(b*b))
    return f_tau

def sigma(u, p, Tau):
    return 2.0*betav*Dincomp(u) - p*Identity(len(u)) + ((1.-betav)/We)*(Tau-Identity(len(u)))

def sigmacom(u, p, Tau):
    return 2.0*betav*Dcomp(u) - p*Identity(len(u)) + ((1.-betav)/We)*(Tau-Identity(len(u)))

def fene_sigma(u, p, Tau, b, lambda_d):
    return 2.0*betav*Dincomp(u) - p*Identity(len(u)) + ((1.-betav)/We)*(phi_def(u, lambda_d)*fene_func(Tau, b)*Tau-Identity(len(u)))

def fene_sigmacom(u, p, Tau, b,lambda_d):
    return 2.0*betav*Dcomp(u) - p*Identity(len(u)) + ((1.-betav)/We)*(phi_def(u, lambda_d)*fene_func(Tau, b)*Tau-Identity(len(u)))

def Fdef(u, Tau):
    return dot(u,grad(Tau)) - dot(grad(u),Tau) - dot(Tau,tgrad(u)) 

def FdefG(u, G, Tau): # DEVSS-G
    return dot(u,grad(Tau)) - dot(G,Tau) - dot(Tau,transpose(G)) 

def Fdefcom(u, Tau):
    return dot(u,grad(Tau)) - dot(grad(u),Tau) - dot(Tau,tgrad(u)) + div(u)*Tau

# Invariants of Tensor 
def I_3(A):
    return A[0,0]*A[1,1]-A[1,0]*A[0,1]



def I_2(A):
    return 0.5*(A[0,0]*A[0,0] + A[1,1]*A[1,1] + 2*A[1,0]*A[0,1]) 

def phi_def(u, lambda_d):
    D_u = as_matrix([[Dincomp(u)[0,0], Dincomp(u)[0,1]],
                     [Dincomp(u)[1,0], Dincomp(u)[1,1]]])
    phi = 1. + (lambda_d*3*I_3(D_u)/(I_2(D_u)+0.0000001))**2  # Dolfin epsillon used to avoid division by zero
    return phi

def psi_def(phi):
    return 0.5*(phi - 1.)
    


def normalize_solution(u):
    "Normalize u: return u divided by max(u)"
    u_array = u.vector().get_local()
    u_max = np.max(np.abs(u_array))
    u_array /= u_max
    u.vector()[:] = u_array
    #u.vector().set_local(u_array)  # alternative
    return u

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

def magnitude(u):
    return np.power((u[0]*u[0]+u[1]*u[1]), 0.5)


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
    u_array = u.vector().get_local()
    u_l2 = norm(u, 'L2')
    u_array /= u_l2
    u.vector()[:] = u_array
    return u
    

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

# CREATE MESH

# Circle 
r_a = 1.0  # Circle Radius
r_x = 0.0
r_y = 0.0

pi = 3.14159265359


# Rectangle 
x_0 = -20.0
y_0 = -0.0
x_1 = 20.0
y_1 = 2.0
nx = 20
ny = 20

rat = (r_a)/(y_1-y_0)



# Artificial Circle inside
c3 = Circle(Point(x_1,y_1), 0.99*r_a, 256)  # Mesh to be used for pressure contour plot
meshc = generate_mesh(c3, 50)




# Create mesh
mm = 64
mesh = sphere_mesh(mm, r_a)
#mesh = refine_bottom(mesh, 1)

res_mesh = restricted_sphere_mesh(mm, r_a)

# Plot/save mesh
mplot(mesh)
plt.savefig("sphere_mesh.png")
plt.clf()
plt.close() 
#quit()




# MARK SUBDOMAINS (Create mesh functions over the cell facets)
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
sub_domains.set_all(5)
no_slip.mark(sub_domains, 0)
bottom.mark(sub_domains, 2)
left_wall.mark(sub_domains, 3)
right_wall.mark(sub_domains, 4)
sphere.mark(sub_domains, 1)

#file = File("subdomains.pvd")
#file << sub_domains


#plot(sub_domains, interactive=False)        # DO NOT USE WITH RAVEN
#quit()

#Define Boundary Parts
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
no_slip.mark(boundary_parts,0)
left_wall.mark(boundary_parts,1)
right_wall.mark(boundary_parts,2)
bottom.mark(boundary_parts,3)
sphere.mark(boundary_parts,4)

ds = Measure("ds")[boundary_parts]



# Define unit Normal/tangent Vector at Sphere Boundary 
n_x =  Expression(('1' , '0'), degree=2)
n_s = Expression(('(x[0]-r_x)/r_a' , '(x[1]-r_y)/r_a'), degree=2, r_a=r_a, r_x=r_x , r_y=r_y)

h = CellDiameter(mesh)
n = FacetNormal(mesh)
n_sphere = FacetNormal(mesh)

# Define function spaces (P2-P1)

# Discretization  parameters
family = "CG"; dfamily = "DG"; rich = "Bubble"
shape = "triangle"; order = 2

# Finite Element Spaces

V_s = VectorElement(family, mesh.ufl_cell(), order)       # Velocity Elements
V_d = VectorElement(dfamily, mesh.ufl_cell(), order-1)
V_se = VectorElement(rich, mesh.ufl_cell(),  order+1)
 
Z_c = VectorElement(family, mesh.ufl_cell(),  order, 3)     # Stress Elements
Z_s = VectorElement(dfamily, mesh.ufl_cell(),  order-1, 3)
Z_se = VectorElement(rich, mesh.ufl_cell(),  order+1, 3)
Z_d = VectorElement(dfamily, mesh.ufl_cell(),  order-2, 3)

Q_s = FiniteElement(family, mesh.ufl_cell(), order-1)   # Pressure/Density Elements
Q_p = FiniteElement(rich, mesh.ufl_cell(), order+1, 3)

# Finite element spaces (Reduced mesh)
Q_sr = FiniteElement(family, res_mesh.ufl_cell(), order-1)   
Q_res = FunctionSpace(res_mesh, Q_sr)
Q_rest = FunctionSpace(res_mesh, "DG", order-2)


Z_e = Z_c + Z_se

#Z_e = EnrichedElement(Z_c,Z_se)                 # Enriched Elements
Z_e = MixedElement(Z_c,Z_se)
V_e = EnrichedElement(V_s,V_se) 
Q_rich = EnrichedElement(Q_s,Q_p)


W = FunctionSpace(mesh,V_s*Z_d)             # F.E. Spaces 
V = FunctionSpace(mesh,V_s)
Vd = FunctionSpace(mesh,V_d)
Z = FunctionSpace(mesh,Z_s)
Ze = FunctionSpace(mesh,Z_e)               #FIX!!!!!!!!!!!!!!!!!!!!!!!!!!!
Zd = FunctionSpace(mesh,Z_d)
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


# Non-mixed Formulation
#u = TestFunction(V)
#v = TrialFunction(V)
#u0 = Function(V)
#u12 = Function(V)
#us = Function(V)
#u1 = Function(V)

#D0_vec = Function(Zd)
#D12_vec = Function(Zd)
#Ds_vec = Function(Zd)
#D1_vec = Function(Zd)

# Mixed Finite Element FOrmulation
#(v, R_vec) = TestFunctions(W)
(u, D_vec) = TrialFunctions(W)


tau_vec = TrialFunction(Zc)
Rt_vec = TestFunction(Zc)

u_test = Function(V)


tau0_vec=Function(Zc)     # Stress Field (Vector) t=t^n
tau12_vec=Function(Zc)    # Stress Field (Vector) t=t^n+1/2
tau1_vec=Function(Zc)     # Stress Field (Vector) t=t^n+1

# Used in conformation tensor only
sig0_vec=Function(Zc)     # Stress Field (Vector) t=t^n
sig12_vec=Function(Zc)    # Stress Field (Vector) t=t^n+1/2
sig1_vec=Function(Zc)     # Stress Field (Vector) t=t^n+1

w0= Function(W)
w12= Function(W)
ws= Function(W)
w1= Function(W)

(u0, D0_vec) = w0.split()
(u12, D12_vec) = w12.split()
(u1, D1_vec) = w1.split()
(us, Ds_vec) = ws.split()



# Initial Conformation Tensor
I_vec = Expression(('1.0','0.0','1.0'), degree=2)
initial_guess_conform = project(I_vec, Zc)


I = Expression((('1.0','0.0'),
                ('0.0','1.0')), degree=2)



# The  projected  rate -of-strain
D_proj_vec = Function(Z)
D_proj = as_matrix([[D_proj_vec[0], D_proj_vec[1]],
                    [D_proj_vec[1], D_proj_vec[2]]])

I_vec = Expression(('1.0','0.0','1.0'), degree=2)
I_vec = interpolate(I_vec, Zc)

I_matrix = as_matrix([[I_vec[0], I_vec[1]],
                      [I_vec[1], I_vec[2]]])



# Project Vector Trial Functions of Stress onto SYMMETRIC Tensor Space

D =  as_matrix([[D_vec[0], D_vec[1]],
                [D_vec[1], D_vec[2]]])

tau = as_matrix([[tau_vec[0], tau_vec[1]],
                 [tau_vec[1], tau_vec[2]]])  

# Project Vector Test Functions of Stress onto SYMMETRIC Tensor Space

Rt = as_matrix([[Rt_vec[0], Rt_vec[1]],
                 [Rt_vec[1], Rt_vec[2]]])        # DEVSS Space

#R = as_matrix([[R_vec[0], R_vec[1]],
#                [R_vec[1], R_vec[2]]])

# SUPG Weighted Test Function
def R_s(Rt, u1):

    h= CellDiameter(mesh)
    alpha_supg = h/(magnitude(u1) + 0.000001)
    
    return Rt + alpha_supg*dot(u1,grad(Rt))

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
conv=1
U=1
L = y_1 - y_0
B = x_1 - x_0
beta_ratio = r_a/(y_1-y_0)
betav = 1.0     
Re = 1                             #Reynolds Number
We = 0.1                           #Weisenberg NUmber
Di = 0.005                         #Diffusion Number
Vh = 0.005
T_0 = 300
T_h = 350
Bi = 0.2
c0 = 1500
Ma = c0/U 




# Define boundary/stabilisation FUNCTIONS

SIN_THETA = Expression('x[1]*x[1]/((x[0]*x[0]+x[1]*x[1])+DOLFIN_EPS)', degree=2) #sin(arctan(y/x)) used in the calculation of the drag 
sin_theta_sq = interpolate(SIN_THETA, Q) # Intepolation of SIN_THETA onto function space
sin_theta = project(np.power(sin_theta_sq, 0.5), Q)


COS_THETA = Expression('x[0]/((x[0]*x[0]+x[1]*x[1])+DOLFIN_EPS)', degree=2) #cos(arctan(y/x)) used in the calculation of the drag 
cos_theta = interpolate(COS_THETA, Q) # Intepolation of COS_THETA onto function space

inflow_profile = ('(16.0-x[1]*x[1])*0.0625','0') #(0.5*(1.0+tanh(8*(t-0.5))))*

in_vel = Expression(('(0.5*(1.0+tanh(8*(t-0.5))))','0'), degree=2, t=0.0, y_1=y_1) # Velocity Boundary Condition (CHANNEL HEIGHT DEPENDENT) #
in_stress = Expression(('(1. + (0.5*(1.0+tanh(8*(t-0.5))))*2.0*(2.0*We*x[1]/(16.0))*(2.0*We*x[1]/(16.0)))'\
                        ,'(0.5*(1.0+tanh(8*(t-0.5))))*(-2.0*We*x[1]/(16.0))','1.'), degree=2, We=We, t=0.0, y_1=y_1) # (CHANNEL HEIGHT DEPENDENT)
rampd=Expression('0.5*(1.0 + tanh(8*(2.0-t)))', degree=2, t=0.0)
rampu=Expression('0.5*(1.0 + tanh(16*(t-2.0)))', degree=2, t=0.0)


#vel_con = interpolate(in_vel, V)
#vel_conx = project(vel_con[0], Q)
#mplot(vel_conx)
#plt.colorbar()
#plt.savefig("velocity.png")
#plt.clf() 
#quit()





# Dirichlet Boundary Conditions  (LID DRIVEN CAVITY)
noslip  = DirichletBC(W.sub(0), in_vel, no_slip)  # Wall moves with the flow
noslip_s  = DirichletBC(W.sub(0), Constant((0, 0)), sphere) 
freeslip = DirichletBC(W.sub(0).sub(1), Constant(0), bottom)

fully_developed = DirichletBC(W.sub(0).sub(1), Constant(0), right_wall)
stress_sphere = DirichletBC(Zc.sub(2), Constant(0), sphere)
stress_top = DirichletBC(Zc.sub(2), Constant(0), no_slip)  
drive  =  DirichletBC(W.sub(0), in_vel, left_wall)  
stress_boundary = DirichletBC(Zc, in_stress, left_wall)
outflow_pressure = DirichletBC(Q, Constant(0), right_wall)
inflow_pressure = DirichletBC(Q, Constant(40.), left_wall)

bcu = [drive, noslip, noslip_s, freeslip, fully_developed]
bcp = [outflow_pressure]
bcT = [] 
bctau = [stress_boundary]

