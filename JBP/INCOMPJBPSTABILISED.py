"""Static Journal Bearing Problem For a Compressible Viscous fluid"""
"""REFERENCE PAPER: AN ANISOTHERMAL PIEZOVISCOUS MODEL FOR JOURNAL BEARING LUBRICATION- T.Phillips, P.C. Bollada"""
"""Log Density formulation for compressible Navier Stokes Equations"""
"""VARIABLE VISCOSITY formulation NOT used"""

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

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;

# HOLOLOW CYLINDER MESH

# Parameters
r_a=0.03 #Journal Radius
r_b=0.05 #Bearing Radius
x1=-0.015
y1=0.0
x2=0.0
y2=0.0

c0=Circle(Point(0.0,0.0), r_a, 256) 
c1=Circle(Point(x1,y1), r_a, 256)
c2=Circle(Point(x2,y2), r_b, 256)

c3 = Circle(Point(x1,y1), 0.99*r_a, 256)  # Mesh to be used for pressure contour plot


ex=x2-x1
ey=y2-y1
ec=np.sqrt(ex**2+ey**2)
c=r_b-r_a
ecc = (ec)/(c)

if c <= 0.0:
   print("ERROR! Journal radius greater than bearing radius")
   quit()

# Create mesh
cyl0=c2-c0
cyl=c2-c1

mesh0 = generate_mesh(cyl0, 45) #Initial Meshing (FENICS Mesh generator)
mesh = generate_mesh(cyl, 20)
meshc= generate_mesh(c3, 20)

# ADAPTIVE MESH REFINEMENT (METHOD 2) "MESH"

# Mesh Refine Code

for i in range(1):
      g=1.0/(3*(i+1))
      #print(g)
      cell_domains = CellFunction("bool", mesh)
      cell_domains.set_all(False)
      for cell in cells(mesh):
          x = cell.midpoint()
          if  (x[0]-x1)**2+(x[1]-y1)**2 < ((1-g)*r_a**2+g*r_b**2) or (x[0]-x2)**2+(x[1]-y2)**2 > (g*r_a**2+(1-g)*r_b**2): 
              cell_domains[cell]=True
      #plot(cell_domains, interactive=True)
      mesh = refine(mesh, cell_domains, redistribute=True)




#Jounral Boundary                                                                              
class Omega0(SubDomain):
      def inside(self, x, on_boundary):
          return True if (x[0]-x1)**2+(x[1]-y1)**2 < (0.96*r_a**2+0.04*r_b**2) and on_boundary  else False  # and 
omega0= Omega0()

# Bearing Boundary
class Omega1(SubDomain):
      def inside(self, x, on_boundary):
          return True if (x[0]-x2)**2 + (x[1]-y2)**2 > (0.04*r_a**2+0.96*r_b**2) and on_boundary else False  #
omega1= Omega1()

# Subdomian for the pressure boundary condition at (r_a,0)
class POmega(SubDomain):
      def inside(self, x, on_boundary):
          return True if x[0] < 0.5*(r_a+r_b) and x[0] > 0 and x[1] < r_a*0.02 and x[1] > -r_a*0.05 and on_boundary else False 
POmega=POmega()


# Create mesh functions over the cell facets (Verify Boundary Classes)
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
sub_domains.set_all(0)
omega0.mark(sub_domains, 2)
omega1.mark(sub_domains, 3)
#POmega.mark(sub_domains, 4)

plot(sub_domains, interactive=True, scalarbar = False)
#quit()


# Define function spaces (P2-P1)
d=1

V = VectorFunctionSpace(mesh1, "CG", 2)
Q = FunctionSpace(mesh1, "CG", 1)
Qt = FunctionSpace(mesh1, "CG", 1)
Z = TensorFunctionSpace(mesh1, "CG", 2)
Zc = TensorFunctionSpace(mesh, "CG", d)

Vs=VectorElement("CG", mesh1.ufl_cell(), 2)
Zs=TensorElement("CG", mesh1.ufl_cell(), 2)
W=FunctionSpace(mesh1,Vs*Zs)
Z=FunctionSpace(mesh1, Zs)




# Set Initial Parameters before loop

w_j =  500.0                     # Characteristic Rotation Speed (rotations/s)
lambda1=0.05      # Relaxation Time



theta=1.0*10E-20
thetat = 1.0*10E-3

dt = 0.0001 #Time Stepping  

loopend=7
j=0
jj=0
tol=10E-6

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
while j < loopend:
    j+=1

    # Define trial and test functions
    #u = TrialFunction(V)
    rho=TrialFunction(Q)
    p = TrialFunction(Q)
    T = TrialFunction(Qt)
    #v = TestFunction(V)
    q = TestFunction(Q)
    r = TestFunction(Qt)
    #tau = TrialFunction(Z)
    #R = TestFunction(Z)

    (v, R) = TestFunctions(W)
    (u, D) = TrialFunctions(W)
    
    tau=TrialFunction(Z)
    Rt=TestFunction(Z)


    #Define Discretised Functions
    #u0=Function(V)       # Velocity Field t=t^n
    #us=Function(V)       # Predictor Velocity Field 
    #u12=Function(V)      # Velocity Field t=t^n+1/2
    #u1=Function(V)       # Velocity Field t=t^n+1

    p0=Function(Q)       # Pressure Field t=t^n
    p1=Function(Q)       # Pressure Field t=t^n+1
    mu=Function(Qt)       # Viscosity Field t=t^n
    T00=Function(Qt) 
    T0=Function(Qt)       # Temperature Field t=t^n
    T1=Function(Qt)       # Temperature Field t=t^n+1

    tau0=Function(Z)     # Stress Field t=t^n
    tau12=Function(Z)    # Stress Field t=t^n+1/2
    tau1=Function(Z)     # Stress Field t=t^n+1

    w0= Function(W)     #Mixed Functions
    w12= Function(W)
    ws= Function(W)
    w1= Function(W)

    (u0, D0) = w0.split()       # Split Mixed Functions
    (u12, D12) = w12.split()
    (u1, D1) = w1.split()
    (us, Ds) = ws.split()


    boundary_parts = FacetFunction("size_t", mesh)
    omega0.mark(boundary_parts,0)
    omega1.mark(boundary_parts,1)
    ds = Measure("ds")[boundary_parts]

    # SCALINGS Nondimensional Parameters
    U = w_j*r_a 
    L = r_b-r_a                               # ()



    # DIMENSIONAL PARAMETERS
    h = mesh.hmin()                     # Minimum Cell Diameter
    #print(h) 
    Tf = 7.5                          # Final Time
    Cv = 1000.0                         # Heat Capcity 
    mu_1 = 5.0*(6.0)*10E-3             # Solvent Viscosity 
    mu_2 = 5.0*(6.0)*10E-3             # Polymeric Viscosity
    mu_0 = mu_1+mu_2                    # Total Viscosity
    Rc = 3.33*10E1       
    T_0 = 300.0                         # Cold Reference Temperature
    T_h = 350.0                         # Hot Reference temperature
    C=250.0                             # Sutherland's Constant
    rho_0=1000.0                        # Density
    #lambda1=2.0*10E-2                  # Relaxation Time
    kappa = 2.0                         # Thermal Conductivity
    heatt= 0.01                         # Heat Transfer Coeficient 
    beta = 69*10E-2                     # Thermal Expansion Coefficient
    betav = mu_1/mu_0                   # Viscosity Ratio 
    alpha=kappa/(rho_0*Cv)              # Thermal Diffusivity
    Bi=0.75                             # Biot Number
    ms=1.0                              # Equation of State Parameter
    Bs=20000.0                          # Equation of State Parameter
    #c0c0=ms*(p0+Bs)*irho0              # Speed of Sound Squared (Dynamic)
    c0=1500.0                           # Speed of Sound (Static)
    k = Constant(dt)
 


    # Nondimensional Parameters
    Re = rho_0*U*(L/mu_0)                              # Reynolds Number
    We = lambda1*U/L                                 # Weisenberg NUmber
    al=0.1                                           # Nonisothermal Parameter
    Di=kappa/(rho_0*Cv*U*L)                          # Diffusion Number
    Vh= U*mu_0/(rho_0*Cv*L*(T_h-T_0))               # Viscous Heating Number
    



    # Comparing different WEISSENBERG Numbers (We=0.1,0.2,0.3,0.4,0.5) at Re=__
    conv=10E-8                                     # Non-inertial Flow Parameter (Re=0)
    Re=1.0
    if j==1:
       We=0.1
    elif j==2:
       We=0.2
    elif j==3:
       We=0.3
    elif j==4:
       We=0.4
    elif j==5:
       We=0.5


    # Comparing different REYNOLDS NUMBERS Numbers (Re=0,5,10,25,50) at We=0.5
    """conv=1                                      # Non-inertial Flow Parameter (Re=0)
    We=1.0
    if j==1:
       conv=10E-8
       Re=1
    elif j==2:
       Re=5
    elif j==3:
       Re=10
    elif j==4:
       Re=25
    elif j==5:
       Re=50"""

    # Continuation in Reynolds/Weissenberg Number Number (Re-->20Re/We-->20We)
    Ret=Expression('Re*(1.0+19.0*0.5*(1.0+tanh(0.7*t-4.0)))', t=0.0, Re=Re, d=d, degree=d)
    Rey=Re
    Wet=Expression('(We/100)*(1.0+99.0*0.5*(1.0+tanh(0.7*t-5.0)))', t=0.0, We=We, d=d, degree=d)


    # Stabilisation Parameters
    th=(1.0-betav)           # DEVSS Stabilisation Terms
    c1=0.1
    c2=0.025
    c3=0.01


    # Define boundary Functions
    td= Constant('5')
    e = Constant('6')
    w = Expression(('(0.5*(1.0+tanh(8*(t-0.5))))*(x[1]-y1)/r_a' , '-(0.5*(1.0+tanh(8*(t-0.5))))*(x[0]-x1)/r_a' ), d=d, degree=d, w_j=w_j , r_a=r_a, x1=x1, y1=y1, e=e , t=0.0)
    #T_bl = Expression('T_0+(T_h-T_0)*sin((x[0]+x[1])*2*pi/L)', d=d, degree=d, T_0=T_0, T_h=T_h, pi=pi, L=L, B=B)
    #T_bb = Expression('T_0+(T_h-T_0)*sin((x[0]+x[1])*2*pi/B)', d=d, degree=d, T_0=T_0, T_h=T_h, pi=pi, L=L, B=B)


    # Define unit Normal/tangent Vector at inner and outer Boundary (Method 2)
    n0 =  Expression(('(x[0]-x1)/r_a' , '(x[1]-y1)/r_a' ), d=d, degree=d, r_a=r_a, x1=x1, y1=y1)
    n1 =  Expression(('(x[0]-x2)/r_b' , '(x[1]-y2)/r_b' ), d=d, degree=d, r_b=r_b, x2=x2, y2=y2)
    t0 =  Expression(('(x[1]-y1)/r_a' , '-(x[0]-x1)/r_a' ),d=d, degree=d, r_a=r_a, x1=x1, y1=y1)
    t1 =  Expression(('(x[1]-y2)/r_b' , '-(x[0]-x2)/r_b' ),d=d, degree=d, r_b=r_b, x2=x2, y2=y2)

    n0v = interpolate(n0, V)
    I = Expression((('1.0','0.0'),
                    ('0.0','1.0')), d=d, degree=d)

     # Dirichlet Boundary Conditions  (LID DRIVEN CAVITY)
    spin =  DirichletBC(W.sub(0), w, omega0)  #The inner cylinder will be rotated with constant angular velocity w_a
    noslip  = DirichletBC(W.sub(0), (0.0, 0.0), omega1) #The outer cylinder remains fixed with zero velocity 
    temp0 =  DirichletBC(W, T_0, omega0)    #Temperature on Omega0 
    press2 = DirichletBC(Q, 0.0, POmega)

    #Collect Boundary Conditions
    bcu = [noslip, spin]
    bcp = []
    bcT = [temp0]
    bctau = []

    N= len(p0.vector().array())

    # Print Parameters of flow simulation
    t = 0.0                  #Time
    e=6
    print '############# Journal Bearing Length Ratios ############'
    print'Eccentricity (m):' ,ec
    print'Radius DIfference (m):',c
    print'Eccentricity Ratio:',ecc

    print '############# Fluid Characteristics ############'
    print 'Density', rho_0
    print 'Solvent Viscosity (Pa.s)', mu_1
    print 'Polymeric Viscosity (Pa.s)', mu_2
    print 'Total Viscosity (Pa.s)', mu_0
    print 'Heat Capacity', Cv
    print 'Thermal Conductivity', kappa

    print '############# TIME SCALE ############'
    print 'Timestep size (s):', dt
    print 'Finish Time (s):', Tf

    print '############# Scalings & Nondimensional Parameters ############'
    print 'Characteristic Length (m):', L
    print 'Characteristic Velocity (m/s):', U
    print 'Speed of sound (m/s):', c0
    print 'Cylinder Speed (t=0) (m/s):', w_j*r_a*(1.0+tanh(e*t-3.0))
    print 'Nondimensionalised Speed of Sound', c0/U
    print 'Nondimensionalised Cylinder Speed (t=0) (m/s):', (1.0+tanh(e*t-3.0))
    print 'Reynolds Number:', Rey
    print 'Weissenberg Number:', We
    print 'Viscosity Ratio:', betav
    print 'Diffusion Number:' ,Di
    print 'Viscous Heating Number:', Vh

    print '############# Discrete Space Characteristics ############'
    print 'Degree of Elements', d
    print('Size of FE Space = %d x d' % N)
    print 'Number of Cells:', mesh.num_cells()
    print 'Number of Vertices:', mesh.num_vertices()
    print '############# Stabilisation Parameters ############'
    print 'DEVSS Momentum Term:', theta
    print 'DEVSS Temperature Term:', thetat
    #quit()

    # Update Nondimensional Variables
    c0=c0/U
    # Equations of State & Viscosity Relations
    c0c0=c0*c0*(1.0+al*T0/T_0)          # Nonisothermal Speed of Sound
    c0c0=project(c0c0,Q)

    # Initial Density Field
    rho_array = rho0.vector().array()
    for i in range(len(rho_array)):  
        rho_array[i] = 1.0
    rho0.vector()[:] = rho_array 

    # Initial Reciprocal of Density Field
    irho_array = irho0.vector().array()
    for i in range(len(irho_array)):  
        irho_array[i] = 1.0/rho_array[i]
    irho0.vector()[:] = irho_array 

    # Initial Temperature Field
    #T_array = T0.vector().array()
    #for i in range(len(T_array)):  
        #T_array[i] = T_0
    #T0.vector()[:] = T_array

      
    I = Expression((('1.0','0.0'),
                    ('0.0','1.0')), d=d, degree=d)


    #Define Variable Parameters, Strain Rate and other tensors
    sr0 = 0.5*(grad(u0) + transpose(grad(u0)))
    sr1 = 0.5*(grad(u1) + transpose(grad(u1)))
    sr12 = 0.5*(grad(u12) + transpose(grad(u12)))
    sr = 0.5*(grad(u) + transpose(grad(u)))
    F0 = (grad(u0)*tau0 + tau0*transpose(grad(u0)))
    F12 = dot(grad(u12),tau) + dot(tau,transpose(grad(u12)))
    Fr12 = dot(grad(u12),tau12) + dot(tau12,transpose(grad(u12)))
    F1 = (grad(u1)*tau + tau*transpose(grad(u1)))
    gamdots = inner(sr0,grad(u0))
    gamdots12 = inner(sr12,grad(u12))
    gamdotp = inner(tau0,grad(u0))
    gamdotp12 = inner(tau12,grad(u12))
    thetal = (T)/(T_h-T_0)
    thetar = (T_0)/(T_h-T_0)
    thetar = project(thetar,W)
    theta0 = (T0-T_0)/(T_h-T_0)
    theta12 = (T12-T_0)/(T_h-T_0)
    alpha = 1.0/(rho*Cv)

    weta = We/dt                                                  #Ratio of Weissenberg number to time step

    # Artificial Diffusion Term
    o= tau1.vector()-tau0.vector()                         # Stress Difference per timestep
    o12= tau12.vector()-tau0.vector()                         # Stress Difference per timestep
    h= p1.vector()-p0.vector()
    m=u1.vector()-u0.vector()                              # Velocity Difference per timestep
    l=T1.vector()-T0.vector()
    alt=norm(o)/(norm(tau1.vector())+10E-10)
    alt=norm(o12)/(norm(tau1.vector())+10E-10)
    alp=norm(h)/(norm(p1.vector())+10E-10)
    alu=norm(m)/(norm(u1.vector())+10E-10)
    alT=norm(l, 'linf')/(norm(T1.vector(),'linf')+10E-10)
    epstau12 = alt*betav+10E-8                                    #Stabilisation Parameter (Stress)
    epstau = alt*betav+10E-8                                    #Stabilisation Parameter (Stress)
    epsp = alp*betav+10E-8                                      #Stabilisation Parameter (Pressure)
    epsu = alu*betav+10E-8                                      #Stabilisation Parameter (Stress)
    epsT = 0.1*alT*kappa+10E-8                                  #Stabilisation Parameter (Temperature)

    # TAYLOR GALERKIN METHOD (COMPRESSIBLE VISCOELASTIC)

    # Weak Formulation (DEVSS in weak formulation)

    """DEVSS Stabilisation used in 'Half step' and 'Velocity Update' stages of the Taylor Galerkin Scheme"""





    # TAYLOR GALERKIN METHOD (INCOMPRESSIBLE VISCOELASTIC)


    #Half Step
    a1=(Re/(dt/2.0))*inner(u,v)*dx+betav*(inner(grad(u),grad(v))*dx)+th*(inner(grad(u),grad(v))*dx-inner(D,grad(v))*dx)+(inner(D,R)*dx-inner(sr,R)*dx)
    L1=(Re/(dt/2.0))*inner(u0,v)*dx-Re*conv*inner(grad(u0)*u0,v)*dx+inner(p0,div(v))*dx \
        -inner(tau0,grad(v))*dx 


    # Stress Half Step
    a3 = (2.0*We/dt)*inner(tau,R)*dx + We*(inner(dot(u12,grad(tau)),R)*dx - inner(F12, R)*dx)
    L3 = (2.0*We/dt-1.0)*inner(tau0,R)*dx + 2.0*(1.0-betav)*inner(D12,R)*dx

    #Predicted U* Equation
    a2=(Re/dt)*inner(u,v)*dx +th*(inner(grad(u),grad(v))*dx-inner(D,grad(v))*dx)+(inner(D,R)*dx-inner(sr,R)*dx)
    L2=(Re/dt)*inner(u0,v)*dx-0.5*betav*(inner(grad(u12),grad(v))*dx) \
        -Re*conv*inner(grad(u12)*u12,v)*dx+inner(p0,div(v))*dx-inner(tau0,grad(v))*dx #\
        #+ theta*inner(D,grad(v))*dx


    #Continuity Equation 1
    a5=inner(grad(p),grad(q))*dx   #Using Dynamic Speed of Sound (c=c(x,t))
    L5=inner(grad(p0),grad(q))*dx+(Re/dt)*inner(us,grad(q))*dx

 
    #Velocity Update
    a7=(Re/dt)*inner(u,v)*dx+0.5*betav*(inner(grad(u),grad(v))*dx)+(inner(D,R)*dx-inner(sr,R)*dx)
    L7=(Re/dt)*inner(us,v)*dx+0.5*(inner(p1,div(v))*dx-inner(p0,div(v))*dx) 

    # SUPG: inner(tau,R+grad())         +c2*h_skew*dot(u1,grad(R))#
    #+We*c2*inner(h_skew*grad(tau),grad(R))*dx+We*c3*inner(h_skew*div(tau),div(R))*dx

    F1=dot(grad(u1),tau) + dot(tau,transpose(grad(u1)))

    # Stress Full Step
    a4 = (We/dt+1.0)*inner(tau,Rt)*dx+We*(inner(dot(u1,grad(tau)),Rt)*dx-inner(F1,Rt)*dx)
    L4 = (We/dt)*inner(tau0,Rt)*dx + 2.0*(1.0-betav)*inner(D1,Rt)*dx

    # Temperature Update (Half Step)
    a8 = (2.0/dt)*inner(rho1*thetal,r)*dx + Di*inner(grad(thetal),grad(r))*dx + inner(rho1*dot(u12,grad(thetal)),r)*dx + thetat*inner(grad(thetal),grad(r))*dx
    L8 = (2.0/dt)*inner(rho1*thetar,r)*dx + Di*inner(grad(thetar),grad(r))*dx + inner(rho1*dot(u12,grad(thetar)),r)*dx \
          + (2.0/dt)*inner(rho1*theta0,r)*dx + Vh*(inner(gamdots,r)*dx + inner(gamdotp,r)*dx - inner(p0*div(u0),r)*dx) - Di*Bi*inner(theta0,r)*ds(1) \
          + thetat*(inner(grad(thetar),grad(r))*dx+inner(Dt,grad(r))*dx)
          #+ inner(,r)*dx  #Neumann Condition on the outer bearing is encoded in the weak formulation

    # Temperature Update (FIRST ORDER)
    a9 = (1.0/dt)*inner(rho1*thetal,r)*dx + Di*inner(grad(thetal),grad(r))*dx + inner(rho1*dot(u1,grad(thetal)),r)*dx + thetat*inner(grad(thetal),grad(r))*dx
    L9 = (1.0/dt)*inner(rho1*thetar,r)*dx + Di*inner(grad(thetar),grad(r))*dx + inner(rho1*dot(u1,grad(thetar)),r)*dx \
          + (1.0/dt)*inner(rho1*theta0,r)*dx + Vh*(inner(gamdots12,r)*dx + inner(gamdotp12,r)*dx-inner(p1*div(u1),r)*dx) - Di*Bi*inner(theta0,r)*ds(1) \
          + thetat*(inner(grad(thetar),grad(r))*dx+inner(Dt,grad(r))*dx)
          #+ inner(,r)*dx  #Neumann Condition on the outer bearing is encoded in the weak formulation

   


    # Assemble matrices
    #A1 = assemble(a1)
    #A2 = assemble(a2)
    #A3 = assemble(a3)
    #A4 = assemble(a4)
    #A5 = assemble(a5)
    #A6 = assemble(a6)
    #A7 = assemble(a7)
    #A8 = assemble(a8)
    #A9 = assemble(a9)


    sigma0 = (-p1*I + 2*betav*sr1+tau1)*t0
    sigma1 = (-p1*I + 2*betav*sr1+tau1)*t1

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

    #Folder To Save Plots for Paraview
    fv=File("Velocity Results Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"theta"+str(theta)+"c0="+str(c0)+"/velocity "+str(t)+".pvd")
 

    # Time/Torque Plot
    x=list()
    y=list()
    z=list()
    zx=list()
    xx=list()
    yy=list()
    zz=list()
    xxx=list()
    yyy=list()


    # Time-stepping
    t = dt
    iter = 0            # iteration counter
    maxiter = 10000000
    while t < Tf + DOLFIN_EPS and iter < maxiter:
        iter += 1
        print"t = %s,  Iteration = %d, Convergence Failures = %s, Loop = %s" %(t, iter, jj, j)

        (u0, D0)=w0.split()
                
        # Compute tentative velocity step
        #begin("Computing tentative velocity")
        A1 = assemble(a1)
        b1= assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]
        solve(A1, w12.vector(), b1, "bicgstab", "default")
        end()

        (u12, D12)=w12.split()

        # Stress Half STEP
        A3 = assemble(a3)
        b3=assemble(L3)
        [bc.apply(A3, b3) for bc in bctau]
        solve(A3, tau12.vector(), b3, "bicgstab", "default")
        end()

    
        
        #Compute Predicted U* Equation
        A2 = assemble(a2)
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcu]
        solve(A2, ws.vector(), b2, "bicgstab", "default")
        end()

        (us, Ds) = ws.split()

        #Pressure Correction Step
        A5 = assemble(a5)
        b5 = assemble(L5)
        [bc.apply(A5, b5) for bc in bcp]
        #[bc.apply(p1.vector()) for bc in bcp]
        solve(A5, p1.vector(), b5, "bicgstab", "default")
        end()

        

        #Velocity Update
        A7 = assemble(a7)
        b7 = assemble(L7)
        [bc.apply(A7, b7) for bc in bcu]
        solve(A7, w1.vector(), b7, "bicgstab", "default")
        end()

        (u1, D1) = w1.split()

        F1=dot(grad(u1),tau) + dot(tau,transpose(grad(u1)))

        a4 = (We/dt+1.0)*inner(tau,Rt+h_ka*dot(u1,grad(Rt)))*dx+We*(inner(dot(u1,grad(tau)),Rt+h_ka*dot(u1,grad(Rt)))*dx-inner(F1,Rt+h_ka*dot(u1,grad(Rt)))*dx)+inner(h_ka*grad(tau),grad(Rt))*dx
        L4 = (We/dt)*inner(tau0,Rt+h_ka*dot(u1,grad(Rt)))*dx + 2.0*(1.0-betav)*inner(D1,Rt+h_ka*dot(u1,grad(Rt)))*dx

        # Stress Full Step
        A4=assemble(a4)
        b4=assemble(L4)
        [bc.apply(A4, b4) for bc in bctau]
        solve(A4, tau1.vector(), b4, "bicgstab", "default")
        end()

        #print 'Stress Size:', norm(assemble(inner(tau1,R)*dx))
        #print 'Pressure Size:',norm(assemble(inner(p1,q)*dx))
        #print 'Velocity Size:', norm(assemble(inner(u1,v)*dx))
     

        #Temperature Equation Stabilisation DEVSS
        #Dt=grad(T0)
        #Dt=project(Dt,V) 

        #Temperature Half Step
        #A8 = assemble(a8)
        #b8 = assemble(L8)
        #[bc.apply(A8, b8) for bc in bcT]
        #solve(A8, T12.vector(), b8, "bicgstab", "default")
        #end()

        #Temperature Equation Stabilisation DEVSS
        #Dt=grad(T12)
        #Dt=project(Dt,V) 

        #Temperature Update
        #A9 = assemble(a9)
        #b9 = assemble(L9)
        #[bc.apply(A9, b9) for bc in bcT]
        #solve(A9, T1.vector(), b9, "bicgstab", "default")
        #end()

        # Energy Calculations
        E_k=assemble(0.5*dot(u1,u1)*dx)
        E_e=assemble((tau1[0,0]+tau1[1,1])*dx)


        # Break Loop if code is diverging

        if max(norm(T1.vector(), 'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) > 10E5 or np.isnan(sum(T1.vector().array())) or np.isnan(sum(u1.vector().array())):
            print 'FE Solution Diverging'   #Print message 
            #with open("Compressible Stability.txt", "a") as text_file:
            #     text_file.write("Iteration:"+str(j)+"--- Re="+str(Rey)+", We="+str(We)+", c0="+str(c0)+", t="+str(t)+", dt="+str(dt)+'\n')
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
            dt=dt/2                         #Use Smaller timestep 
            j-=1                            #Extend loop
            jj+= 1                          # Convergence Failures
            break


        # Record Error Data 
        o= tau1.vector()-tau0.vector()                         # Stress Difference per timestep
        h= p1.vector()-p0.vector()
        m=u1.vector()-u0.vector()                              # Velocity Difference per timestep
        l=T1.vector()-T0.vector()


        if iter > 1:
           xx.append(t)
           yy.append(norm(h,'linf')/(norm(p1.vector())+10E-10))
           zz.append(norm(o,'linf')/(norm(tau1.vector())+10E-10))
           xxx.append(norm(m,'linf')/(norm(u1.vector())+10E-10))
           yyy.append(norm(l,'linf')/(norm(T1.vector())+10E-10))


        # Save Plot to Paraview Folder 
        #for i in range(5000):
        #    if iter== (0.02/dt)*i:
        #       fv << u1
        #ft << T1



        # Record Elastic & Kinetic Energy Values (Method 1)
        if j==1:
           x1.append(t)
           ek1.append(E_k)
           ee1.append(E_e)
        if j==2:
           x2.append(t)
           ek2.append(E_k)
           ee2.append(E_e)
        if j==3:
           x3.append(t)
           ek3.append(E_k)
           ee3.append(E_e)
        if j==4:
           x4.append(t)
           ek4.append(E_k)
           ee4.append(E_e)
        if j==5:
           x5.append(t)
           ek5.append(E_k)
           ee5.append(E_e)


        # Plot solution
        if t>0.0:
            #plot(tauxx, title="Normal Stress", rescale=True)         # Normal Stress
            #plot(div(u1), title="Compression", rescale=True )
            plot(p1, title="Pressure", rescale=True)                # Pressure
            #plot(rho1, title="Density", rescale=True)               # Density
            #plot(u1, title="Velocity", rescale=True, mode = "auto") # Velocity
            #plot(T1, title="Temperature", rescale=True)             # Temperature
        
        # Record Torque Data 
        x.append(t)
        y.append(assemble(innertorque))
        z.append(assemble(innerforcey))
           

        # Move to next time step
        w0.assign(w1)
        u0.assign(u1)
        T0.assign(T1)
        rho0.assign(rho1)
        p0.assign(p1)
        tau0.assign(tau1)
        w.t=t
        Ret.t=t
        t += dt

        #Plot Kinetic and elasic Energies for different REYNOLDS numbers at constant Weissenberg Number    
    """if max(norm(T1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) < 10E6 and j==5:
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
        plt.savefig("Incompressible Viscoelastic Flow Results/Stability-Convergence/We0p1KineticEnergyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
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
        plt.savefig("Incompressible Viscoelastic Flow Results/Stability-Convergence/We0p1ElasticEnergyRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
        plt.clf()"""



        #Plot Kinetic and elasic Energies for different Weissenberg numbers at Re=0 (METHOD 2)
    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E6 and j==5 or j==1:
        # Kinetic Energy
        plt.figure(0)
        plt.plot(x1, ek1, 'r-', label=r'$We=0.1$')
        plt.plot(x2, ek2, 'b-', label=r'$We=0.2$')
        plt.plot(x3, ek3, 'c-', label=r'$We=0.3$')
        plt.plot(x4, ek4, 'm-', label=r'$We=0.4$')
        plt.plot(x5, ek5, 'g-', label=r'$We=0.5$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_k')
        plt.savefig("Incompressible Viscoelastic Flow Results/Stability-Convergence/KineticEnergyTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
        plt.clf()
        # Elastic Energy
        plt.figure(1)
        plt.plot(x1, ee1, 'r-', label=r'$We=0.1$')
        plt.plot(x2, ee2, 'b-', label=r'$We=0.2$')
        plt.plot(x3, ee3, 'c-', label=r'$We=0.3$')
        plt.plot(x4, ee4, 'm-', label=r'$We=0.4$')
        plt.plot(x5, ee5, 'g-', label=r'$We=0.5$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_e')
        plt.savefig("Incompressible Viscoelastic Flow Results/Stability-Convergence/ElasticEnergyTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
        plt.clf()

    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E6 and j==5 or j==1:

        # Plot Stress/Normal Stress Difference
        tau_xx=project(tau1[0,0],Q)
        mplot(tau_xx)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/tau_xxRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
        plt.clf() 
        tau_xy=project(tau1[1,0],Q)
        mplot(tau_xy)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/tau_xyRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
        plt.clf() 
        tau_yy=project(tau1[1,1],Q)
        mplot(tau_yy)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/tau_yyRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
        plt.clf() 
        #N1=project(tau1[0,0]-tau1[1,1],Q)
        #mplot(N1)
        #plt.colorbar()
        #plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/FirstNormalStressDifferenceRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
        #plt.clf()


    if max(norm(T1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(w1.vector(), 'linf')) < 10E6 and dt < 0.02:

        # Plot Torque Data
        plt.plot(x, y, 'r-', label='Torque')
        plt.xlabel('time(s)')
        plt.ylabel('Torque')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/Torque We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"t="+str(t)+".png")
        plt.clf()
        plt.plot(x, zx, 'b-', label='Horizontal Load Force')
        plt.xlabel('time(s)')
        plt.ylabel('Horzontal Load Force')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/Horizontal Load We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"t="+str(t)+".png")
        plt.clf()
        plt.plot(x, z, 'b-', label='Vertical Load Force')
        plt.xlabel('time(s)')
        plt.ylabel('Vertical Load Force')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/Vertical Load We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"t="+str(t)+".png")
        plt.clf()
        plt.plot(zx, z, 'b-', label='Evolution of Force')
        plt.xlabel('Fx')
        plt.ylabel('Fy')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/Force We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"t="+str(t)+".png")
        plt.clf()

    if max(norm(T1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) < 10E5 and dt < 0.02:

        p1=project(p1,Q)
        mplot(rho1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/DensityRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"t="+str(t)+".png")
        plt.clf()
        mplot(p1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/PressureRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"t="+str(t)+".png")
        plt.clf()
        mplot(T1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/TemperatureRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"t="+str(t)+".png")
        plt.clf()




        #Plot PRESSURE Contours USING MATPLOTLIB
        # Scalar Function code

        #Set Values for inner domain as -infty
        Q1=FunctionSpace(meshc, "CG", d)

        x = Expression('x[0]', d=d, degree=d)  #GET X-COORDINATES LIST
        y = Expression('x[1]', d=d, degree=d)  #GET Y-COORDINATES LIST
        Q1=FunctionSpace(meshc, "CG", d)
        pj=Expression('0', d=d, degree=d) #Expression for the 'pressure' in the domian
        pjq=interpolate(pj, Q1)
        pjvals=pjq.vector().array()

        xyvals=meshc.coordinates()
        xqalsv = interpolate(x, Q1)
        yqalsv= interpolate(y, Q1)

        xqals= xqalsv.vector().array()
        yqals= yqalsv.vector().array()

        pvals = p1.vector().array() # GET SOLUTION u= u(x,y) list
        xyvals = mesh.coordinates() # CLEAN THIS UP!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
        pp = mlab.griddata(xvals, yvals, pvals, xx, yy, interp='nn') # u(x,y) data so that it can be used by 

        plt.contour(XX, YY, pp, 30)
        plt.colorbar()
        plt.title('Pressure Contours')   # PRESSURE CONTOUR PLOT
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/Pressure Contours Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"t="+str(t)+".png")
        plt.clf()


        #Plot TEMPERATURE Contours USING MATPLOTLIB
        # Scalar Function code

        #Set Values for inner domain as ZERO


        Tj=Expression('0', d=d, degree=d) #Expression for the 'pressure' in the domian
        Tjq=interpolate(Tj, Q1)
        Tjvals=Tjq.vector().array()

        Tvals = T1.vector().array() # GET SOLUTION T= T(x,y) list
        Tvals = np.concatenate([Tvals, Tjvals])  #Merge two arrays for Temperature values

        TT = mlab.griddata(xvals, yvals, Tvals, xx, yy, interp='nn') # u(x,y) data so that it can be used by 

        plt.contour(XX, YY, TT, 30)
        plt.colorbar()
        plt.title('Temperature Contours')   # TEMPERATURE CONTOUR PLOT
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/Temperature Contours Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"t="+str(t)+".png")
        plt.clf()



        # Plot Velocity Field Contours (MATPLOTLIB)

        #Set Velocity in the Journal (inner circle) to -1000000 and get in 
        V1=VectorFunctionSpace(meshc, "CG", d)
        vj=Expression(('0','0'), d=d, degree=d) #Expression for the 'pressure' in the domian
        vjq=interpolate(vj, V1)
        uvjvals=vjq.vector().array()
        n1= meshc.num_vertices()
        xy = Expression(('x[0]','x[1]'), d=d, degree=d)  #GET MESH COORDINATES LIST
        gg=list()
        hh=list()
        for i in range(len(vjq.vector().array())/2):  
            gg.append(uvjvals[2*i+1])
            hh.append(uvjvals[2*i])
        ujvals=np.asarray(hh)
        vjvals=np.asarray(gg)

        xyvalsv = interpolate(xy, V1)

        qq=list()
        rr=list()

        for i in range(len(xyvalsv.vector().array())/2):  
            qq.append(xyvalsv.vector().array()[2*i+1])
            rr.append(xyvalsv.vector().array()[2*i])

        xvalsj = np.asarray(rr)
        yvalsj = np.asarray(qq)

        g=list()
        h=list()
        n= mesh.num_vertices()

        for i in range(len(u1.vector().array())/2):                     # Length of vector Depends on Degree of the Elements 
            g.append(u1.vector().array()[2*i+1])
            h.append(u1.vector().array()[2*i])

        uvals = np.asarray(h)                    # GET SOLUTION (u,v) -> u= u(x,y) list
        vvals = np.asarray(g)                    # GET SOLUTION (u,v) -> v= v(x,y) list


        xy = Expression(('x[0]','x[1]'), d=d, degree=d)  #GET MESH COORDINATES LIST
        xyvalsv = interpolate(xy, V)

        q=list()
        r=list()

        for i in range(len(u1.vector().array())/2):
           q.append(xyvalsv.vector().array()[2*i+1])
           r.append(xyvalsv.vector().array()[2*i])

        xvals = np.asarray(r)
        yvals = np.asarray(q)
     

        #Merge arrays
        uvals = np.concatenate([uvals, ujvals])  #Merge two arrays for velocity values
        vvals = np.concatenate([vvals, vjvals])  #Merge two arrays for velocity values
        xvals = np.concatenate([xvals, xvalsj])   #Merge two arrays for x-coordinate values
        yvals = np.concatenate([yvals, yvalsj])   #Merge two arrays for y-coordinate values


        uu = mlab.griddata(xvals, yvals, uvals, xx, yy, interp='nn') 
        vv = mlab.griddata(xvals, yvals, vvals, xx, yy, interp='nn') 

        #Determine Speed 
        speed = np.sqrt(uu*uu+ vv*vv)

        plot3 = plt.figure()
        plt.streamplot(XX, YY, uu, vv,  
                       density=5,              
                       color=speed/speed.max(),  
                       cmap=cm.gnuplot,                         # colour map
                       linewidth=0.5*speed/speed.max()+0.5)       # line thickness
        plt.colorbar()
        plt.title('Journal Bearing Problem')
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/Velocity Contours Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"t="+str(t)+".png")   
        plt.clf()                                                                     # display the plot

    # Update Control Parameter

    plt.close()


    if dt < tol:
       j=loopend+1
       break



    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E6 or abs(E_k) > 1:
        Tf=5 

    # Update Control Variables 
    if max(norm(T1.vector(), 'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) < 10E5 and np.isfinite(sum(u1.vector().array())):
        with open("Compressible Stability.txt", "a") as text_file:
             text_file.write("Solution Converges Re"+str(Rey)+", We="+str(We)+", dt="+str(dt)+'\n')
        dt = 0.064 #Time Stepping                        #Go back to Original Timestep
        #gammah=10E0
        jj=0
        if w_j==2000.0:
            w_j=2.5*w_j
            lambda1=10.0*c/(r_a*w_j)
        elif w_j==1000.0:
            w_j=2.0*w_j
            lambda1=10.0*c/(r_a*w_j)
        elif w_j==500.0:
            w_j=2.0*w_j
            lambda1=10.0*c/(r_a*w_j)
        elif w_j==250.0:
            w_j=2.0*w_j
            lambda1=10.0*c/(r_a*w_j)
        elif w_j==100.0:
            w_j=2.5*w_j
            lambda1=10.0*c/(r_a*w_j)
        elif w_j==50.0:
            w_j=2.0*w_j
            lambda1=10.0*c/(r_a*w_j)
        elif w_j==25.0:
            w_j=2.0*w_j
            lambda1=10.0*c/(r_a*w_j)
        elif w_j==10.0:
            w_j=2.5*w_j
            lambda1=10.0*c/(r_a*w_j)
        elif w_j==5.0:
            w_j=2.0*w_j
            lambda1=10.0*c/(r_a*w_j)
        elif w_j==2.5:
            w_j=2.0*w_j
            lambda1=10.0*c/(r_a*w_j)

         


