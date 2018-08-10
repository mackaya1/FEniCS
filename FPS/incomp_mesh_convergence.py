"""Base Code for the Finite Element solution of the FLOW PAST A Sphere"""

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
        plt.axis('off')
    elif isinstance(obj, Mesh):
        if (obj.geometry().dim() != 2):
            raise(AttributeError)
        plt.triplot(mesh2triang(obj), color='k', linewidth = 0.3)
        plt.axis('off')

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

# Artificial Circle inside
c3 = Circle(Point(x_1,y_1), 0.99*r_a, 256)  # Mesh to be used for pressure contour plot
meshc = generate_mesh(c3, 50)

#quit()

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


# Create Mesh




#No-Slip Boundary                                                                              
class No_Slip(SubDomain):
      def inside(self, x, on_boundary):
          return True if on_boundary and near(x[1], y_1) else False  
no_slip = No_Slip()

# Sphere
class Sphere(SubDomain):
      def inside(self, x, on_boundary):
          return True if (x[0]-r_x)**2 + (x[1]-r_y)**2 < r_a**2 + DOLFIN_EPS and on_boundary  else False  
sphere = Sphere()

# Bottom Boundary
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

#plot(mesh)




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


def absolute(u):
    u_array = np.absolute(u.vector().get_local())
    u.vector()[:] = u_array
    return u


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

# Default Nondimensional Parameters
U=1
L = y_1 - y_0
B = x_1 - x_0
beta_ratio = r_a/(y_1-y_0)

Di = 0.005                         #Diffusion Number
Vh = 0.005
T_0 = 300
T_h = 350
Bi = 0.2
c0 = 1500
Ma = c0/U 


  
T_f = 8.0
Tf = T_f
loopend=3
j = 0
jj = 0
jjj = 0
tol = 10E-5
defpar = 1.0

conv = 0                                      # Non-inertial Flow Parameter (Re=0)
We=0.1
Re=1.0
c0 = 1000
Ma = 0.001
betav = 0.5

alph1 = 0.0
c1 = 0.1
c2 = 0.001
th = 0.5              # DEVSS

corr=1

# FEM Solution Convergence/Energy Plot
x1=list()
x2=list()
x3=list()
x4=list()
x5=list()
y=list()
z=list()
zz=list()
zzz=list()
zl=list()
ek1=list()
ek2=list()
ek3=list()
ek4=list()
ee1=list()
ee2=list()
ee3=list()
ee4=list()
ek5=list()
ee5=list()
fd1=list()
fd2=list()
fd3=list()
fd4=list()
fd5=list()
x_axis=list()
y_axis=list()
u_xg = list()
u_yg = list()
sig_xxg = list()
sig_xyg = list()
sig_yyg = list()
while j < loopend:
    j+=1
    t=0.0

  

    if j==1:
        mm=48
        mesh = sphere_mesh(mm, r_a)
    if j==2:
        mm=64
        mesh = sphere_mesh(mm, r_a)
    if j==3:
        mm=80
        mesh = sphere_mesh(mm, r_a)


    mesh = refine_bottom(mesh, 1)

    mplot(mesh)
    plt.savefig("sphere_mesh"+str(mm)+".png")
    plt.clf()
    plt.close() 
    #quit()


    dt = 2.5*mesh.hmin()**2  #Time Stepping

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

    # Define function spaces (P2-P1)

    # Discretization  parameters
    family = "CG"; dfamily = "DG"; rich = "Bubble"
    shape = "triangle"; order = 2

    # Finite ELement Spaces

    V_s = VectorElement(family, mesh.ufl_cell(), order)       # Velocity Elements
    V_d = VectorElement(dfamily, mesh.ufl_cell(), order-1)
    V_se = VectorElement(rich, mesh.ufl_cell(),  order+1)
     
    Z_c = VectorElement(family, mesh.ufl_cell(),  order, 3)     # Stress Elements
    Z_s = VectorElement(dfamily, mesh.ufl_cell(),  order-1, 3)
    Z_se = VectorElement(rich, mesh.ufl_cell(),  order+1, 3)
    Z_d = VectorElement(dfamily, mesh.ufl_cell(),  order-2, 3)

    Q_s = FiniteElement(family, mesh.ufl_cell(), order-1)   # Pressure/Density Elements
    Q_p = FiniteElement(rich, mesh.ufl_cell(), order+1, 3)


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
    (v, R_vec) = TestFunctions(W)
    (u, D_vec) = TrialFunctions(W)

    u_test = Function(V)


    tau_vec = TrialFunction(Zc)
    Rt_vec = TestFunction(Zc)


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




    # Define boundary/stabilisation FUNCTIONS

    SIN_THETA = Expression('x[1]/((x[0]*x[0]+x[1]*x[1])+DOLFIN_EPS)', degree=2) #sin(arctan(y/x)) used in the calculation of the drag   x[1]/((x[0]**2+x[1]**2)+DOLFIN_EPS)
    sin_theta = interpolate(SIN_THETA, Q) # Intepolation of SIN_THETA onto function space

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

    # Define unit Normal/tangent Vector at Sphere Boundary 
    n_x =  Expression(('1' , '0'), degree=2)
    n_s = Expression(('(x[0]-r_x)/r_a' , '(x[1]-r_y)/r_a'), degree=2, r_a=r_a, r_x=r_x , r_y=r_y)

    h = CellDiameter(mesh)
    n = FacetNormal(mesh)
    n_sphere = FacetNormal(mesh)



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




        

    # Continuation in Reynolds/Weissenberg Number Number (Re-->20Re/We-->20We)
    Ret=Expression('Re*(1.0+19.0*0.5*(1.0+tanh(0.7*t-4.0)))', t=0.0, Re=Re, degree=2)
    Rey=Re
    Wet=Expression('(We/100)*(1.0+99.0*0.5*(1.0+tanh(0.7*t-5.0)))', t=0.0, We=We, degree=2)


    print '############# TIME SCALE ############'
    print 'Timestep size (s):', dt
    print 'Finish Time (s):', Tf

    print '############# Scalings & Nondimensional Parameters ############'
    print 'Channel Length (m):', L
    print 'Characteristic Length:', B
    print 'Shpere Radius / Channel Width:', beta_ratio
    print 'Reynolds Number:', Rey
    print 'Non-inertial parameter:', conv
    print 'Weissenberg Number:', We
    print 'Viscosity Ratio:', betav
    print 'Diffusion Number:' ,Di
    print 'Viscous Heating Number:', Vh

    Np= len(p0.vector().get_local())
    Nv= len(w0.vector().get_local())   
    Ntau= len(tau0_vec.vector().get_local())
    #dof= 3*Nv+2*Ntau+Np
    print '############# Discrete Space Characteristics ############'
    print 'Degree of Elements', order
    print 'Mesh: %s x %s' %(mm, mm)
    print('Size of Pressure Space = %d ' % Np)
    print('Size of Velocity/DEVSS Space = %d ' % Nv)
    print('Size of Stress Space = %d ' % Ntau)
    #print('Degrees of Freedom = %d ' % dof)
    print 'Number of Cells:', mesh.num_cells()
    print 'Number of Vertices:', mesh.num_vertices()
    print 'Minimum Cell Diamter:', mesh.hmin()
    print 'Maximum Cell Diamter:', mesh.hmax()
    print '############# Stabilisation Parameters ############'
    print 'DEVSS Momentum Term:', th

    


    # Initial Temperature Field
    T_array = T0.vector().get_local()
    for i in range(len(T_array)):  
        T_array[i] = T_0
    T0.vector()[:] = T_array

     


    # Stabilisation

    # Ernesto Castillo 2016 p.
    """F1R = dot(u1,grad(tau1)) - dot(grad(u1),tau1) - dot(tau1,tgrad(u1)) + div(u1)*tau1  #Compute the residual in the STRESS EQUATION
    F1R_vec = as_vector([F1R[0,0], F1R[1,0], F1R[1,1]])
    Dcomp1_vec = as_vector([Dcomp(u1)[0,0], Dcomp(u1)[1,0], Dcomp(u1)[1,1]])
    restau = We*F1R_vec - 2*(1-betav)*Dcomp1_vec
    res_test = project(restau0, Zd)
    res_orth = project(restau0-res_test, Zc) 
    Fv = dot(u1,grad(Rt)) - dot(grad(u1),Rt) - dot(Rt,tgrad(u1)) + div(u1)*Rt
    Fv_vec = as_vector([Fv[0,0], Fv[1,0], Fv[1,1]])
    Dv_vec =  as_vector([Dcomp(v)[0,0], Dcomp(v)[1,0], Dcomp(v)[1,1]])                              
    osgs_stress = inner(res_orth, We*Fv_vec - 2*(1-betav)*Dv_vec)*dx"""

    # LPS Projection
    """F1R = dot(u1,grad(tau1)) - dot(grad(u1),tau1) - dot(tau1,tgrad(u1)) + div(u1)*tau1  #Compute the residual in the STRESS EQUATION
    F1R_vec = as_vector([F1R[0,0], F1R[1,0], F1R[1,1]])
    Dcomp1_vec = as_vector([Dcomp(u1)[0,0], Dcomp(u1)[1,0], Dcomp(u1)[1,1]])
    restau0 = We/dt*(tau1_vec-tau0_vec) + We*F1R_vec + tau1_vec - 2*(1-betav)*Dcomp1_vec 
    res_test = project(restau0, Zd)
    res_orth = project(restau0-res_test, Zc)                                
    res_orth_norm_sq = project(inner(res_orth,res_orth), Qt)     # Project residual norm onto discontinuous space
    res_orth_norm = np.power(res_orth_norm_sq, 0.5)
    tau_stab = as_matrix([[res_orth[0]*tau_vec[0], res_orth[1]*tau_vec[1]],
                          [res_orth[1]*tau_vec[1], res_orth[2]*tau_vec[2]]])
    tau_stab1 = as_matrix([[res_orth[0]*tau1_vec[0], res_orth[1]*tau1_vec[1]],
                          [res_orth[1]*tau1_vec[1], res_orth[2]*tau1_vec[2]]])
    Rt_stab = as_matrix([[res_orth[0]*Rt_vec[0], res_orth[1]*Rt_vec[1]],
                          [res_orth[1]*Rt_vec[1], res_orth[2]*Rt_vec[2]]]) 
    kapp = project(res_orth_norm, Qt)
    LPSl_stress = inner(kapp*h*c1*grad(tau),grad(Rt))*dx + inner(kapp*h*c2*div(tau),div(Rt))*dx  # Stress Stabilisation"""


    # DEVSS Stabilisation
    
    DEVSSl_u12 = 2*(1-betav)*inner(Dcomp(u),Dincomp(v))*dx    
    DEVSSr_u12 = 2*inner(D0,Dincomp(v))*dx   
    DEVSSl_u1 = 2*(1-betav)*inner(Dcomp(u),Dincomp(v))*dx    
    DEVSSr_u1 = 2*inner(D12,Dincomp(v))*dx 



    # Set up Krylov Solver 

    # Use amg preconditioner if available
    prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

    # Use nonzero guesses - essential for CG with non-symmetric BC
    parameters['krylov_solver']['nonzero_initial_guess'] = True
    parameters['krylov_solver']['monitor_convergence'] = False
    
    solveru = KrylovSolver("bicgstab", "default")
    solvertau = KrylovSolver("bicgstab", "default")
    solverp = KrylovSolver("cg", prec)

    #Folder To Save Plots for Paraview
    #fv=File("Velocity Results Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"/velocity "+str(t)+".pvd")
 

    #Lists for Energy Values
    x=list()
    ee=list()
    ek=list()
    #ftau=File("Incompressible Viscoelastic Flow Results/Paraview/Stress_th"+str(th)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"/stress "+str(t)+".pvd")
    #fv=File("Incompressible Viscoelastic Flow Results/Paraview/Velocity_th"+str(th)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"/velocity "+str(t)+".pvd")

    # Time-stepping
    t = 0.0
    iter = 0            # iteration counter
    maxiter = 1000000
    while t < Tf + DOLFIN_EPS and iter < maxiter:
        iter += 1
        print"t = %s,  Iteration = %d, Convergence Failures = %s, Loop = %s - %s" %(t, iter, jj, jjj, j)

        # Set Function timestep
        in_stress.t = t
        in_vel.t = t


        # Update Stabilisation (Copy and Paste Stabilisation Technique from above)
        F1R = Fdef(u1, tau1)  #Compute the residual in the STRESS EQUATION
        F1R_vec = as_vector([F1R[0,0], F1R[1,0], F1R[1,1]])
        Dincomp1_vec = as_vector([Dincomp(u1)[0,0], Dincomp(u1)[1,0], Dincomp(u1)[1,1]])
        restau0 = We/dt*(tau1_vec-tau0_vec) + We*F1R_vec + tau1_vec - I_vec
        res_test = project(restau0, Zd)
        res_orth = project(restau0-res_test, Zc)                                
        res_orth_norm_sq = project(inner(res_orth,res_orth), Qt)     # Project residual norm onto discontinuous space
        res_orth_norm = np.power(res_orth_norm_sq, 0.5)
        kapp = project(res_orth_norm, Qt)
        kapp = absolute(kapp)
        LPSl_stress = inner(kapp*h*c1*grad(tau),grad(Rt))*dx + inner(kapp*h*c2*div(tau),div(Rt))*dx  # Stress Stabilisation
                
        U12 = 0.5*(u1 + u0)    
        # Update Solutions
        if iter > 1:
            w0.assign(w1)
            T0.assign(T1)
            p0.assign(p1)
            tau0_vec.assign(tau1_vec)


        noslip  = DirichletBC(W.sub(0), in_vel, no_slip) 
        drive  =  DirichletBC(W.sub(0), in_vel, left_wall)  
        stress_boundary = DirichletBC(Zc, in_stress, left_wall)

        #if iter > 10:
            #bcu = [noslip, noslip_s, freeslip, drive, fully_developed]             

        (u0, D0_vec)=w0.split()  

        D0 = as_matrix([[D0_vec[0], D0_vec[1]],
                        [D0_vec[1], D0_vec[2]]])                    #DEVSS STABILISATION
        DEVSSr_u1 = 2.0*(1-betav)*inner(D0,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS


        U = 0.5*(u + u0)     

         
        

        """# VELOCITY HALF STEP (ALTERNATIVE)
        lhsFu12 = Re*(2.0*(u - u0) / dt + conv*dot(u0, nabla_grad(u0)))
        Fu12 = dot(lhsFu12, v)*dx \
               + inner(2.0*betav*Dincomp(u0), Dincomp(v))*dx - ((1.-betav)/We)*inner(div(tau0), v)*dx + inner(grad(p0),v)*dx\
               + inner(D-Dincomp(u),R)*dx 

        a1 = lhs(Fu12)
        L1 = rhs(Fu12)

        #+ dot(p0*n, v)*ds - dot(betav*nabla_grad(U)*n, v)*ds - ((1.0-betav)/We)*dot(tau0*n, v)*ds\

            #DEVSS Stabilisation
        a1+= th*DEVSSl_u12                     
        L1+= th*DEVSSr_u12 

        A1 = assemble(a1)
        b1= assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]
        solve(A1, w12.vector(), b1, "bicgstab", "default")
        end()

        (u12, D12_vec) = w12.split()
        D12 = as_matrix([[D12_vec[0], D12_vec[1]],
                        [D12_vec[1], D12_vec[2]]])
        DEVSSr_u1 = 2*(1-betav)*inner(D12,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS"""

        """# STRESS Half Step
        F12 = dot(u12,grad(tau)) - dot(grad(u12),tau) - dot(tau,tgrad(u12)) # Convection/Deformation Terms
        lhs_tau12 = (We/dt+1.0/We)*tau + F12                             # Left Hand Side
        rhs_tau12= (We/dt)*tau0 + (1/We)*I                     # Right Hand Side

        a3 = inner(lhs_tau12,Rt)*dx                                 # Weak Form
        L3 = inner(rhs_tau12,Rt)*dx

        a3 += SUPGl3             # SUPG Stabilisation LHS
        L3 += SUPGr3             # SUPG / SU Stabilisation RHS
        A3=assemble(a3)
        b3=assemble(L3)
        [bc.apply(A3, b3) for bc in bctau]
        solve(A3, tau12_vec.vector(), b3, "bicgstab", "default")
        end()"""
        
        #Predicted U* Equation
        """lhsFus = Re*((u - u0)/dt + conv*dot(u12, nabla_grad(u12)))
        Fus = dot(lhsFus, v)*dx + \
               + inner(sigma(U, p0, tau0), Dincomp(v))*dx\
               + dot(p0*n, v)*ds - betav*(dot(Dincomp(U)*n, v)*ds) - ((1.0-betav)/We)*dot(tau0*n, v)*ds\
               + inner(D-Dincomp(u),R)*dx   
              
        a2= lhs(Fus)
        L2= rhs(Fus)""" 



        #Predicted U* Equation (ALTERNATIVE)
        lhsFus = Re*((u - u0)/dt + conv*dot(u0, nabla_grad(u0)))
        Fus = dot(lhsFus, v)*dx + \
               + inner(2.0*betav*Dincomp(U), Dincomp(v))*dx - ((1. - betav)/We)*inner(div(tau0), v)*dx + inner(grad(p0),v)*dx\
               + inner(D-Dincomp(u),R)*dx   
              
        a2= lhs(Fus)
        L2= rhs(Fus)

        #+ dot(p0*n, v)*ds - betav*(dot(nabla_grad(U)*n, v)*ds) - ((1.0-betav)/We)*dot(tau0*n, v)*ds\

            # Stabilisation
        a2+= th*DEVSSl_u1   #[th*DEVSSl_u12]                     
        L2+= th*DEVSSr_u1    #[th*DEVSSr_u12]

        A2 = assemble(a2)        
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcu]
        solveru.solve(A2, ws.vector(), b2)
        end()
        (us, Ds_vec) = ws.split()


        #PRESSURE CORRECTION
        a5=inner(grad(p),grad(q))*dx 
        L5=inner(grad(p0),grad(q))*dx - (Re/dt)*div(us)*q*dx
        A5 = assemble(a5)
        b5 = assemble(L5)
        [bc.apply(A5, b5) for bc in bcp]
        #[bc.apply(p1.vector()) for bc in bcp]
        solverp.solve(A5, p1.vector(), b5)
        end()
        
        #Velocity Update
        lhs_u1 = (Re/dt)*u                                          # Left Hand Side
        rhs_u1 = (Re/dt)*us                                         # Right Hand Side

        a7=inner(lhs_u1,v)*dx + inner(D-Dincomp(u),R)*dx                                           # Weak Form
        L7=inner(rhs_u1,v)*dx - 0.5*inner(grad(p1-p0),v)*dx #- 0.5*dot(p1*n, v)*ds

        a7+= 0   #[th*DEVSSl_u1]                                                #DEVSS Stabilisation
        L7+= 0   #[th*DEVSSr_u1] 

        A7 = assemble(a7)
        b7 = assemble(L7)
        [bc.apply(A7, b7) for bc in bcu]
        solve(A7, w1.vector(), b7)
        end()
        (u1, D1_vec) = w1.split()
        D1 = as_matrix([[D1_vec[0], D1_vec[1]],
                        [D1_vec[1], D1_vec[2]]])

        U12 = 0.5*(u1 + u0)   

        # Stress Full Step
        lhs_tau1 = (We/dt + 1.0)*tau  +  We*Fdef(u1,tau)                            # Left Hand Side
        rhs_tau1= (We/dt)*tau0 + Identity(len(u)) 

        A = inner(lhs_tau1,Rt)*dx - inner(rhs_tau1,Rt)*dx
        a4 = lhs(A)
        L4 = rhs(A) 

            # SUPG / SU / LPS Stabilisation (User Choose One)

        a4 += LPSl_stress  # [SUPGl4, SUl4, LPSl_stab, LPSl_stress, diff_stab, 0]
        L4 += 0  # [SUPGr4, SUr4, LPSr_stab, LPS_res_stab, 0]   


        A4=assemble(a4)                                     # Assemble System
        b4=assemble(L4)
        [bc.apply(A4, b4) for bc in bctau]
        solvertau.solve(A4, tau1_vec.vector(), b4)
        end()


        # Temperature Update (FIRST ORDER)
        #lhs_theta1 = (1.0/dt)*thetal + dot(u1,grad(thetal))
        #rhs_theta1 = (1.0/dt)*thetar + dot(u1,grad(thetar)) + (1.0/dt)*theta0 + Vh*gamdots
        #a8 = inner(lhs_theta1,r)*dx + Di*inner(grad(thetal),grad(r))*dx 
        #L8 = inner(rhs_theta1,r)*dx + Di*inner(grad(thetar),grad(r))*dx + Bi*inner(grad(theta0),n1*r)*ds(1) 

        # Energy Calculations
        E_k=assemble(0.5*dot(u1,u1)*dx)
        E_e=assemble((tau1[0,0]+tau1[1,1]-2.)*dx)

        # Drag Calculations
        
        tau_drag = dot(sigma(u1, p1, tau1),Constant((1.0,0.0)))
        drag_correction = 6.0*pi*r_a   # corr
        F_drag = assemble(-2.*pi*r_a*r_a*inner(n,tau_drag)*SIN_THETA*ds(4))/drag_correction

        # Alternative Formulation
        #tau_drag = (sigma(u1, p1, tau1)[0,0]*COS_THETA + sigma(u1, p1, tau1)[1,0]*SIN_THETA)*SIN_THETA
        #drag_correction = 6.0*pi*r_a 
        #F_drag = -2.*pi*r_a*r_a*assemble(tau_drag*ds(4))/drag_correction



        #print tau_drag.vector().arrray().max()     
        #print F_drag       
        # Record Elastic & Kinetic Energy Values (Method 1)
        if j==1:
           x1.append(t)
           ek1.append(E_k)
           ee1.append(E_e)
           fd1.append(F_drag) # Newtonian Reference line (F_drag)
        if j==2:
           x2.append(t)
           ek2.append(E_k)
           ee2.append(E_e)
           fd2.append(F_drag)
        if j==3:
           x3.append(t)
           ek3.append(E_k)
           ee3.append(E_e)
           fd3.append(F_drag)
        if j==4:
           x4.append(t)
           ek4.append(E_k)
           ee4.append(E_e)
           fd4.append(F_drag)
        if j==5:
           x5.append(t)
           ek5.append(E_k)
           ee5.append(E_e)
           fd5.append(F_drag)

        # Record Error Data 


        
        #shear_stress=project(tau1[1,0],Q)
        # Save Plot to Paraview Folder 
        #for i in range(5000):
        #    if iter== (0.01/dt)*i:
        #       ftau << shear_stress


        # Break Loop if code is diverging

        if norm(w1.vector(), 'linf') > 10E6 or np.isnan(sum(w1.vector().get_local())):
            print 'FE Solution Diverging'   #Print message 
            #with open("DEVSS Weissenberg Compressible Stability.txt", "a") as text_file:
                 #text_file.write("Iteration:"+str(j)+"--- Re="+str(Rey)+", We="+str(We)+", t="+str(t)+", dt="+str(dt)+'\n')
            if j==1:           # Clear Lists
               x1=list()
               ek1=list()
               ee1=list()
            if j==2:
               x2=list()
               ek2=list()
               ee2=list()
            if j==3:
               x3=list()
               ek3=list()
               ee3=list()
            if j==4:
               x4=list()
               ek4=list()
               ee4=list()
            if j==5:
               x5=list()
               ek5=list()
               ee5=list() 
            j-=1                            # Extend loop
            jj+= 1                          # Convergence Failures
            if jj>0:
                Tf= (iter-25)*dt
            #alph = alph + 0.05

            # Reset Functions
            p0=Function(Q)       # Pressure Field t=t^n
            p1=Function(Q)       # Pressure Field t=t^n+1
            T0=Function(Qt)       # Temperature Field t=t^n
            T1=Function(Qt)       # Temperature Field t=t^n+1
            tau0_vec=Function(Zc)     # Stress Field (Vector) t=t^n
            tau12_vec=Function(Zc)    # Stress Field (Vector) t=t^n+1/2
            tau1_vec=Function(Zc)     # Stress Field (Vector) t=t^n+1
            w0= Function(W)
            w12= Function(W)
            ws= Function(W)
            w1= Function(W)
            (u0, D0_vec)=w0.split()
            (u12, D12_vec)=w0.split()
            (us, Ds_vec)=w0.split()
            (u1, D1_vec)=w0.split()
            break


        # Plot solution
        #if t>0.1:
            #plot(kapp, title="Stabilisation Coeficient", rescale=True )
            #plot(tau1[1,0], title="Normal Stress", rescale=True)
            #plot(p1, title="Pressure", rescale=True)
            #plot(u1, title="Velocity", rescale=True)
            #plot(T1, title="Temperature", rescale=True)
                

        # Move to next time step
        t += dt

    # Plot Mesh Convergence Data 
    """if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E6 and j==5:
        # Kinetic Energy
        plt.figure(0)
        plt.plot(x1, ek1, 'r-', label=r'M1')
        plt.plot(x2, ek2, 'b-', label=r'M2')
        plt.plot(x3, ek3, 'c-', label=r'M3')
        plt.plot(x4, ek4, 'm-', label=r'M4')
        plt.plot(x5, ek5, 'g-', label=r'M5')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_k')
        plt.savefig("Incompressible Viscoelastic Flow Results/Stability-Convergence/Mesh_KineticEnergyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
        # Elastic Energy
        plt.figure(1)
        plt.plot(x1, ee1, 'r-', label=r'M1')
        plt.plot(x2, ee2, 'b-', label=r'M2')
        plt.plot(x3, ee3, 'c-', label=r'M3')
        plt.plot(x4, ee4, 'm-', label=r'M4')
        plt.plot(x5, ee5, 'g-', label=r'M5')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_e')
        plt.savefig("Incompressible Viscoelastic Flow Results/Stability-Convergence/Mesh_ElasticEnergyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
        plt.clf()"""

        #Plot Kinetic and elasic Energies for different REYNOLDS numbers at constant Weissenberg Number    
    """if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E6 and j==5:
        # Kinetic Energy
        plt.figure(0)
        plt.plot(x1, ek1, 'r-', label=r'$Re=0$')
        plt.plot(x2, ek2, 'b-', label=r'$Re=5$')
        plt.plot(x3, ek3, 'c-', label=r'$Re=10$')
        plt.plot(x4, ek4, 'm-', label=r'$Re=25$')
        plt.plot(x5, ek5, 'g-', label=r'$Re=50$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_k')
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/Fixed_We_KineticEnergyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
        # Elastic Energy
        plt.figure(1)
        plt.plot(x1, ee1, 'r-', label=r'$Re=0$')
        plt.plot(x2, ee2, 'b-', label=r'$Re=5$')
        plt.plot(x3, ee3, 'c-', label=r'$Re=10$')
        plt.plot(x4, ee4, 'm-', label=r'$Re=25$')
        plt.plot(x5, ee5, 'g-', label=r'$Re=50$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_e')
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/Fixed_We_ElasticEnergyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
        plt.clf()"""



     # Plot Mesh Convergence Data 
    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E10 and j==loopend or j==1:
        # Kinetic Energy
        plt.figure(0)
        plt.plot(x1, ek1, 'r-', label=r'M1')
        plt.plot(x2, ek2, 'b--', label=r'M2')
        plt.plot(x3, ek3, 'c:', label=r'M3')
        #plt.plot(x4, ek4, 'm-', label=r'M4')
        plt.legend(loc='best')
        plt.xlabel('time(s)', fontsize=16)
        plt.ylabel('$E_k$', fontsize=16)
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/Mesh_KineticEnergyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
        plt.clf()
        # Elastic Energy
        plt.figure(1)
        plt.plot(x1, ee1, 'r-', label=r'M1')
        plt.plot(x2, ee2, 'b--', label=r'M2')
        plt.plot(x3, ee3, 'c:', label=r'M3')
        #plt.plot(x4, ee4, 'm-', label=r'M4')
        plt.legend(loc='best')
        plt.xlabel('time(s)', fontsize=16)
        plt.ylabel('$E_e$', fontsize=16)
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/Mesh_ElasticEnergyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
        plt.clf()
        plt.figure(2)
        plt.plot(x1, fd1, 'r-', label=r'M1')
        plt.plot(x2, fd2, 'b--', label=r'M2')
        plt.plot(x3, fd3, 'c:', label=r'M3')
        #plt.plot(x4, ee4, 'm-', label=r'M4')
        plt.legend(loc='best')
        plt.xlabel('time(s)', fontsize=16)
        plt.ylabel('$D^{*}$', fontsize=16)
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/Mesh_DragEnergyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
        plt.clf()
        plt.close()




    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E5 and abs(E_k) < 100 and j==1 or j==loopend:

        # Plot Stress/Normal Stress Difference
        tau_xx=project(tau1[0,0],Q)
        mplot(tau_xx)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/meshtau_xxRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(alph1)+".png")
        plt.clf() 
        tau_xy=project(tau1[1,0],Q)
        mplot(tau_xy)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/meshtau_xyRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(alph1)+".png")
        plt.clf() 
        tau_yy=project(tau1[1,1],Q)
        mplot(tau_yy)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/meshtau_yyRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(alph1)+".png")
        plt.clf() 
        #N1=project(tau1[0,0]-tau1[1,1],Q)
        #mplot(N1)
        #plt.colorbar()
        #plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/FirstNormalStressDifferenceRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
        #plt.clf()

    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E4 and abs(E_k) < 100:
 
       # Plot Velocity Components
        ux=project(u1[0],Q)
        mplot(ux)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/meshu_xRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
        plt.clf()
        uy=project(u1[1],Q)
        mplot(uy)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/meshu_yRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
        plt.clf()
        mplot(p1)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/meshPressureRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
        plt.clf()

    if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==loopend or j==1:
        #Plot Contours USING MATPLOTLIB
        # Scalar Function code


        x = Expression('x[0]', degree=2)     #GET X-COORDINATES LIST
        y = Expression('x[1]', degree=2)     #GET Y-COORDINATES LIST
        pvals = p1.vector().get_local()          # GET SOLUTION p= p(x,y) list
        tauxx = project(tau1_vec[0], Q)
        tauxxvals = tauxx.vector().get_local()
        xyvals = mesh.coordinates()     # CLEAN THIS UP!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        xvalsq = interpolate(x, Q)#xyvals[:,0]
        yvalsq= interpolate(y, Q)#xyvals[:,1]
        xvalsw = interpolate(x, Qt)#xyvals[:,0]
        yvalsw= interpolate(y, Qt)#xyvals[:,1]

        xvals = xvalsq.vector().get_local()
        yvals = yvalsq.vector().get_local()


        xx = np.linspace(x_0,x_1)
        yy = np.linspace(y_0,y_1)
        XX, YY = np.meshgrid(xx,yy)   # (x,y) coordinate data formatted so that it can be used by plt.contour()
        pp = mlab.griddata(xvals, yvals, pvals, xx, yy, interp='nn') # u(x,y) data so that it can be used by 


        plt.contour(XX, YY, pp, 25)
        plt.title('Pressure Contours')   # PRESSURE CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/meshPressureContoursRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
        plt.clf()


        #Plot Velocity Streamlines USING MATPLOTLIB
        u1_q = project(u1[0],Q)
        uvals = u1_q.vector().get_local()
        v1_q = project(u1[1],Q)
        vvals = v1_q.vector().get_local()

            # Interpoltate velocity field data onto matlab grid
        uu = mlab.griddata(xvals, yvals, uvals, xx, yy, interp='nn') 
        vv = mlab.griddata(xvals, yvals, vvals, xx, yy, interp='nn') 


            #Determine Speed 
        speed = np.sqrt(uu*uu+ vv*vv)

        plot3 = plt.figure()
        plt.streamplot(XX, YY, uu, vv,  
                       density=3,              
                       color=speed,  
                       cmap=cm.gnuplot,                         # colour map
                       linewidth=0.8)                           # line thickness
                                                                # arrow size
        plt.colorbar()                                          # add colour bar on the right
        plt.title('Lid Driven Cavity Flow')
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/meshVelocityContoursRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")   
        plt.clf()                                             # display the plot


    plt.close()

    if dt < tol:
       j=loopend+1
       break

    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E5 and abs(E_k) < 100:
        Tf=T_f 
    
    if j==5:
       jjj+=1
       alph1 = alph1/2
       j=0

    if jjj==3:
       quit()    


    # Reset Functions
    rho0 = Function(Q)
    rho1 = Function(Q)
    p0 = Function(Q)       # Pressure Field t=t^n
    p1 = Function(Q)       # Pressure Field t=t^n+1
    T0 = Function(Q)       # Temperature Field t=t^n
    T1 = Function(Q)       # Temperature Field t=t^n+1
    tau0_vec = Function(Zc)     # Stress Field (Vector) t=t^n
    tau12_vec = Function(Zc)    # Stress Field (Vector) t=t^n+1/2
    tau1_vec = Function(Zc)     # Stress Field (Vector) t=t^n+1
    w0 = Function(W)
    w12 = Function(W)
    ws = Function(W)
    w1 = Function(W)
    (u0, D0_vec) = w0.split()
    (u12, D12_vec) = w0.split()
    (us, Ds_vec) = w0.split()
    (u1, D1_vec) = w0.split()
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





