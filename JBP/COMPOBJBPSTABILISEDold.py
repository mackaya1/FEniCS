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

c0=Circle(Point(0.0,0.0), r_a, 500) 
c1=Circle(Point(x1,y1), r_a, 500)
c2=Circle(Point(x2,y2), r_b, 500)

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

mesh0 = generate_mesh(cyl0, 50) #Initial Meshing (FENICS Mesh generator)
mesh = generate_mesh(cyl, 50)
meshc= generate_mesh(c3, 15)

# ADAPTIVE MESH REFINEMENT (METHOD 2) "MESH"

#SKEW MESH FUNCTION

# POLAR COORDINATE FUNCTIONS
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

epsx = ex/c
epsy = ey/c

#Journal Bearing coordinate system
def pol2mesh(rho, phi):
    x = rho*np.cos(phi) - epsx*(r_b-rho)
    y = rho*np.sin(phi) - epsy*(r_b-rho)
    return(x, y)

# MESH CONSTRUCTION CODE

n= mesh0.num_vertices()
m= mesh0.num_cells()
coorX = mesh0.coordinates()[:,0]
coorY = mesh0.coordinates()[:,1]
cells0 = mesh0.cells()[:,0]
cells1 = mesh0.cells()[:,1]
cells2 = mesh0.cells()[:,2]

# CARTESIAN -> POLAR -> SKEWED CARTESIAN
r= list()
theta = list()
xx = list()
yy = list()
for i in range(n):
    r.append(cart2pol(coorX[i], coorY[i])[0])
    theta.append(cart2pol(coorX[i], coorY[i])[1])
for i in range(n):
    xx.append(pol2mesh(r[i], theta[i])[0])
    yy.append(pol2mesh(r[i], theta[i])[1])

xx = np.asarray(xx)
yy = np.asarray(yy)


# MESH GENERATION (Using Mesheditor)
mesh1 = Mesh()
editor = MeshEditor()
editor.open(mesh1,2,2)
editor.init_vertices(n)
editor.init_cells(m)
for i in range(n):
    editor.add_vertex(i, xx[i], yy[i])
for i in range(m):
    editor.add_cell(i, cells0[i], cells1[i], cells2[i])
editor.close()


# Mesh Refine Code

for i in range(1):
      g=1.0/(4*(i+1))
      #print(g)
      cell_domains = CellFunction("bool", mesh)
      cell_domains.set_all(False)
      for cell in cells(mesh):
          x = cell.midpoint()
          if  (x[0]-x1)**2+(x[1]-y1)**2 < ((1-g)*r_a**2+g*r_b**2) or (x[0]-x2)**2+(x[1]-y2)**2 > (g*r_a**2+(1-g)*r_b**2): 
              cell_domains[cell]=True
      #plot(cell_domains, interactive=True)
      mesh = refine(mesh, cell_domains, redistribute=True)

plot(mesh,interactive=True)
print 'Number of Cells:', mesh.num_cells()
print 'Number of Vertices:', mesh.num_vertices()
#quit()


#Jounral Boundary                                                                              
class Omega0(SubDomain):
      def inside(self, x, on_boundary):
          return True if (x[0]-x1)**2+(x[1]-y1)**2 < (0.9*r_a**2+0.1*r_b**2) and on_boundary  else False  # and 
omega0= Omega0()

# Bearing Boundary
class Omega1(SubDomain):
      def inside(self, x, on_boundary):
          return True if (x[0]-x2)**2 + (x[1]-y2)**2 > (0.05*r_a**2+0.95*r_b**2) and on_boundary else False  #
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

plot(sub_domains, interactive=False, scalarbar = False)
#quit()


# Define function spaces (P2-P1)
d=2

V = VectorFunctionSpace(mesh, "CG", d)
Q = FunctionSpace(mesh, "CG", d)
W = FunctionSpace(mesh, "CG", d)
Z = TensorFunctionSpace(mesh, "DG", d)
Zc = TensorFunctionSpace(mesh, "CG", d)


w_j =  300                     # Characteristic Rotation Speed (rotations/s)
lambda1=0.01#15.0*(r_b-r_a)/(r_a*w_j)         # Relaxation Time

betav=1
c0=500.0                           # Speed of Sound (Static)
Bi=0.5                                # Biot Number


#gammah=2*10E-10                  # SUPG Stabilsation Terms
#gam =0.01

theta=1.0*10E-20
thetat = 1.0*10E-6

dt = 0.002 #Time Stepping  


# Torque Plot 
y=list()
z=list()
zx=list()
xx=list()
yy=list()
zz=list()
xxx=list()
yyy=list()

loopend=5
j=0
kk=0
jj=0
tol=10E-6
while j < loopend:
    j+=1

    # Define trial and test functions
    u = TrialFunction(V)
    rho=TrialFunction(Q)
    p = TrialFunction(Q)
    T = TrialFunction(W)
    v = TestFunction(V)
    q = TestFunction(Q)
    r = TestFunction(W)
    tau = TrialFunction(Zc)
    R = TestFunction(Zc)


    #Define Discretised Functions
    u00=Function(V) 
    u0=Function(V)       # Velocity Field t=t^n
    us=Function(V)       # Predictor Velocity Field 
    u12=Function(V)      # Velocity Field t=t^n+1/2
    u1=Function(V)       # Velocity Field t=t^n+1
    irho0=Function(Q)
    rho0=Function(Q)     # Density Field t=t^n
    rho1=Function(Q)     # Density Field t=t^n+1
    irho0=Function(Q)
    irho1=Function(Q)
    p00=Function(Q)      # Pressure Field t=t^n-1
    p0=Function(Q)       # Pressure Field t=t^n
    p1=Function(Q)       # Pressure Field t=t^n+1
    diffp=Function(Q)
    mu=Function(W)       # Viscosity Field t=t^n
    T12=Function(W) 
    T0=Function(W)       # Temperature Field t=t^n
    T1=Function(W)       # Temperature Field t=t^n+1
    iT=Function(W)
    tau00=Function(Zc)
    tau0=Function(Zc)     # Stress Field t=t^n
    tau12=Function(Zc)    # Stress Field t=t^n+1/2
    tau1=Function(Zc)     # Stress Field t=t^n+1

    c0c0=Function(Q)
    ic0c0=Function(Q)
    tauxx=Function(Q)    # Normal Stress
    divu=Function(Q)

    D=TrialFunction(Z)   # DEVSS Articficial Diffusion 
    D1=Function(Z)       # Terms



    #print len(tau1.vector().array())/4
    #print len(tauxx.vector().array())

    tauxx_vec=tauxx.vector().array()
    for i in range(len(tauxx.vector().array())):
        tauxx_vec[i]=tau1.vector().array()[4*i]
    tauxx.vector()[:]=tauxx_vec

    #quit()


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
    Tf = 80.0                           # Final Time
    Cv = 800.0                         # Heat Capcity 
    mu_1 = 1.0*(5.0)*10E-3            # Solvent Viscosity 
    mu_2 = 9.0*(5.0)*10E-3            # Polymeric Viscosity
    mu_0 = mu_1+mu_2                    # Total Viscosity
    Rc = 3.33*10E1       
    T_0 = 300.0                         # Cold Reference Temperature
    T_h = 350.0                         # Hot Reference temperature
    C=250.0                             # Sutherland's Constant
    rho_0=820.0                        # Density
    #lambda1=2.0*10E-2                  # Relaxation Time
    kappa = 2.0                         # Thermal Conductivity
    heatt= 0.01                         # Heat Transfer Coeficient 
    beta = 69*10E-2                     # Thermal Expansion Coefficient
    #betav = mu_1/mu_0                   # Viscosity Ratio 
    alpha=kappa/(rho_0*Cv)              # Thermal Diffusivity
    #Bi=1                                # Biot Number
    ms=1.0                              # Equation of State Parameter
    Bs=20000.0                          # Equation of State Parameter
    #c0c0=ms*(p0+Bs)*irho0              # Speed of Sound Squared (Dynamic)
    #c0=1500.0                           # Speed of Sound (Static)
    k = Constant(dt)



    # Nondimensional Parameters
    Re = rho_0*U*(L/mu_0)                              # Reynolds Number
    We = lambda1*U/L                                 # Weisenberg NUmber
    al=0.01                                           # Nonisothermal Parameter
    Di=kappa/(rho_0*Cv*U*L)                          # Diffusion Number
    Vh= U*mu_0/(rho_0*Cv*L*(T_h-T_0))               # Viscous Heating Number


	# Scaled Parameters
    
    Re=Re/10
    Wes=We/100

    # Steady State Method (Re-->10Re)
    Ret = Expression('Re*(1.0+0.5*(1.0+tanh(0.3*t-4.0))*9.0)', t=0.0, Re=Re, d=d, degree=d)
    Wet = Expression('Wes*(0.5*(1.0+tanh(0.3*t-4.0))*100.0)', t=0.0, Wes=Wes, d=d, degree=d)

    Rey=10*Re


    # Define boundary conditions
    td= Constant('5')
    e = Constant('6')
    w = Expression(('(0.5*(1.0+tanh(e*t-2.5)))*(x[1]-y1)/r_a' , '-(0.5*(1.0+tanh(e*t-2.5)))*(x[0]-x1)/r_a' ), d=d, degree=d, w_j=w_j , r_a=r_a, x1=x1, y1=y1, e=e , t=0.0)
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

    spin =  DirichletBC(V, w, omega0)  #The inner cylinder will be rotated with constant angular velocity w_a
    noslip  = DirichletBC(V, (0.0, 0.0), omega1) #The outer cylinder remains fixed with zero velocity 
    temp0 =  DirichletBC(W, T_h, omega0)    #Temperature on Omega0 
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
    print 'Relaxation Time (s):', lambda1
    print 'Journal Temperature:', T_h
    print 'Biot Number:', Bi

    print '############# TIME SCALE ############'
    print 'Timestep size (s):', dt
    print 'Finish Time (s):', Tf

    print '############# Scalings & Nondimensional Parameters ############'
    print 'Characteristic Length (m):', L
    print 'Characteristic Velocity (m/s):', U
    print 'Speed of sound (m/s):', c0
    print 'Cylinder Speed (t=0) (m/s):', w_j*r_a*(1.0+tanh(e*t-3.0))
    print 'Pressure/Stress Scaling:', mu_0*(U/(L))
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
    quit()

    
    # Initial Density Field
    rho_array = rho0.vector().array()
    for i in range(len(rho_array)):  
        rho_array[i] = 1.0
    rho0.vector()[:] = rho_array 

    # Initial Reciprocal of Density Field
    #irho_array = irho0.vector().array()
    #for i in range(len(irho_array)):  
        #irho_array[i] = 1.0/rho_array[i]
    #irho0.vector()[:] = irho_array 

    # Initial Temperature Field
    T_array = T0.vector().array()
    for i in range(len(T_array)):  
        T_array[i] = T_0
    T0.vector()[:] = T_array


    # Update Nondimensional Variables
    c0nd=c0/U
    # Equations of State & Viscosity Relations
    c0c0=c0nd*c0nd*(1.0+al*T0/T_0)          # Nonisothermal Speed of Sound
    c0c0=project(c0c0,Q)

      
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




    # TAYLOR GALERKIN METHOD (COMPRESSIBLE VISCOELASTIC)

    """SUPG Diffusion coefficient used in 

       Finite Element Solution of the Navier-Stokes Equations using a SUPG Formulation
       -Vellando, Puertas Agudo, Marques  (Page 12)"""


    """def al1(x):               # Define Diffusion coeficient function
        return np.tanh(2.0*x)
    f1=Expression(('1','0'), degree=d, d=d)    
    f2=Expression(('0','1'), degree=d, d=d)

  
    speed = dot(u0,u0)        # Determine Speed 
    speed = project(speed,Q)  # Project Speed onto FE Space 



    uvals = dot(f1,u0)
    vvals = dot(f2,u0)
    uval = project(uvals,Q)
    vval = project(vvals,Q)

    hxi = (1.0/mm)                 # Uniform Mesh Length of x-axis side   
    heta = (1.0/mm)                # Uniform Mesh Length of y-axis side

    eta=Function(Q)            # Define Approximation Rule Functions
    xi=Function(Q)

    u_array = uval.vector().array()
    v_array = vval.vector().array()
    eta_array = eta.vector().array()
    xi_array = xi.vector().array()
    for i in range(len(u_array)):  
        eta_array[i] = al1(0.5*u_array[i]/mu_0)
        xi_array[i] = al1(0.5*v_array[i]/mu_0)
    eta.vector()[:] = eta_array  
    xi.vector()[:] = xi_array

    eta=project(eta,Q)
    xi=project(xi,Q)

    gam = gammah*(hxi*xi*uval+heta*eta*vval)/(2.0*speed+10E-8)
    gam = project(gam,Q)"""
    #gam = 0.0

    # SUPG Term
    #vm=v+gam*dot(u0,grad(v))

    # DEVSS STABILISATION

    #Momentum Equation Stabilisation DEVSS

    D=2.0*(sr0)-2.0/3*div(u0)*I
    D=project(D,Zc)

    #Temperature Equation Stabilisation DEVSS
    Dt=grad(T0)
    Dt=project(Dt,V) 

    #Mixed Finite Element Space
    #VZc=FunctionSpace(mesh, V*Zc)
    #(vm, Rm)=TestFunction(VZc)
    #(u,D)=Function(VZc)

    #gam=0.0


    #Half Step
    a1=(2.0/dt)*inner(Rey*rho0*u,v)*dx+betav*(inner(grad(u),grad(v))*dx+1.0/3*inner(div(u),div(v))*dx)#\
         #+ theta*(inner(grad(u),grad(v))*dx+1.0/3*inner(div(u),div(v))*dx)
    L1=(2.0/dt)*inner(Rey*rho0*u0,v)*dx-inner(Ret*rho0*grad(u0)*u0,v)*dx+inner(p0,div(v))*dx -inner(tau0,grad(v))*dx #+ theta*inner(D,grad(v))*dx

    # Stress Half Step
    #a3 = (2.0*We/dt)*inner(tau,R)*dx + (inner(Wet*dot(u12,grad(tau)),R)*dx - inner(Wet*F12, R)*dx+inner(Wet*div(u12)*tau,R)*dx)+0.00001*epstau*inner(grad(tau),grad(R))*dx
    #L3 = (2.0*We/dt)*inner(tau0,R)*dx + 2.0*(1.0-betav)*inner(sr0,R)*dx+0.00001*epstau*inner(grad(tau0),grad(R))*dx

    #Predicted U* Equation
    a2=(1.0/dt)*inner(Rey*rho0*u,v)*dx #+ theta*(inner(grad(u),grad(v))*dx+1.0/3*inner(div(u),div(v))*dx)
    L2=(1.0/dt)*inner(Rey*rho0*u0,v)*dx-0.5*betav*(inner(grad(u0),grad(v))*dx+1.0/3*inner(div(u0),div(v))*dx) \
        -inner(Rey*rho0*grad(u12)*u12,v)*dx+inner(p0,div(v))*dx-inner(tau12,grad(v))*dx #\
        #+ theta*inner(D,grad(v))*dx

  
    #Continuity Equation 1
    a5=(1.0/dt)*inner(p,q)*dx+0.5*dt*inner(c0nd*c0nd*grad(p),grad(q))*dx   #Using Dynamic Speed of Sound (c=c(T,x,t))
    L5=(1.0/dt)*inner(p0,q)*dx+0.5*dt*inner(c0nd*c0nd*grad(p0),grad(q))*dx-(inner(c0nd*c0nd*rho0*div(us),q)*dx+inner(c0nd*c0nd*dot(grad(rho0),us),q)*dx)

    #Continuity Equation 2 
    #a6=inner(c0c0*rho,q)*dx 
    #L6=inner(c0c0*rho0,q)*dx + inner(diffp,q)*dx 

    #Velocity Update
    a7=(1.0/dt)*inner(Rey*rho0*u,v)*dx+0.5*betav*(inner(grad(u),grad(v))*dx+1.0/3*inner(div(u),div(v))*dx)#\
         #+ theta*(inner(grad(u),grad(v))*dx+1.0/3*inner(div(u),div(v))*dx)
    L7=(1.0/dt)*inner(Rey*rho0*us,v)*dx+0.5*(inner(p1,div(v))*dx-inner(p0,div(v))*dx) #+ theta*inner(D,grad(v))*dx


    # Stress Full Step
    #a4 = (We/dt)*inner(tau,R)*dx+inner((T0/T_0)*tau,R)*dx  
    #L4 = (We/dt)*inner(tau0,R)*dx + 2.0*(1.0-betav)*inner(sr0,R)*dx- (inner(Wet*dot(u12,grad(tau12)),R)*dx + inner(Wet*Fr12, R)*dx + inner(Wet*div(u12)*tau12,R)*dx ) 


    # Temperature Update (Half Step)
    a8 = (2.0/dt)*inner(rho1*thetal,r)*dx + Di*inner(grad(thetal),grad(r))*dx + inner(rho1*dot(u12,grad(thetal)),r)*dx + thetat*inner(grad(thetal),grad(r))*dx
    L8 = (2.0/dt)*inner(rho1*thetar,r)*dx + Di*inner(grad(thetar),grad(r))*dx + inner(rho1*dot(u12,grad(thetar)),r)*dx \
          + (2.0/dt)*inner(rho1*theta0,r)*dx + Vh*(inner(gamdots,r)*dx + inner(gamdotp,r)*dx - inner(p0*div(u0),r)*dx) - 10*Di*Bi*inner(theta0,r)*ds(1) \
          + thetat*(inner(grad(thetar),grad(r))*dx+inner(Dt,grad(r))*dx)
          #+ inner(,r)*dx  #Neumann Condition on the outer bearing is encoded in the weak formulation

    # Temperature Update (FIRST ORDER)
    a9 = (1.0/dt)*inner(rho1*thetal,r)*dx + Di*inner(grad(thetal),grad(r))*dx + inner(rho1*dot(u1,grad(thetal)),r)*dx + thetat*inner(grad(thetal),grad(r))*dx
    L9 = (1.0/dt)*inner(rho1*thetar,r)*dx + Di*inner(grad(thetar),grad(r))*dx + inner(rho1*dot(u1,grad(thetar)),r)*dx \
          + (1.0/dt)*inner(rho1*theta0,r)*dx + Vh*(inner(gamdots12,r)*dx + inner(gamdotp12,r)*dx-inner(p1*div(u1),r)*dx) - 10*Di*Bi*inner(theta0,r)*ds(1) \
          + thetat*(inner(grad(thetar),grad(r))*dx+inner(Dt,grad(r))*dx)
          #+ inner(,r)*dx  #Neumann Condition on the outer bearing is encoded in the weak formulation

    
    # Temperature Update
    #a8 = (rho_0*Cv*U/c)*((1.0/dt)*inner(rho1*T,r)*dx + inner(inner(u1,grad(T)),r)*dx) + (kappa/(c*c))*inner(grad(T),grad(r))*dx #+ epsT*inner(grad(T),grad(r))*dx
    #L8 = (rho_0*Cv*U/c)*(1.0/dt)*inner(rho1*T0,r)*dx + (kappa/(c*c))*(heatt*(inner(grad(T0),n1*r)*ds(1)) +(U*U/(c*c))*(mu_1*inner(gamdots,r)*dx + inner(gamdotp,r)*dx)-mu_1*inner(p1*div(u1),r)*dx) 


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


    # Torque and Bearing Load

    sigma0 = (-p1*I + 2*betav*sr1+tau1)*t0
    sigma1 = (-p1*I + 2*betav*sr1+tau1)*t1

    omegaf0 = p1*I*n0  #Nomral component of the stress 
    omegaf1 = p1*I*n1


    innerforcex = -inner(Constant((1.0, 0.0)), omegaf0)*ds(0)
    innerforcey = -inner(Constant((0.0, 1.0)), omegaf0)*ds(0)
    outerforcex = -inner(Constant((1.0, 0.0)), omegaf1)*ds(1)
    outerforcey = -inner(Constant((0.0, 1.0)), omegaf1)*ds(1)
    innertorque = -r_a*inner(n0, sigma0)*ds(0)
    outertorque = -r_b*inner(n1, sigma1)*ds(1)



    # Use amg preconditioner if available
    prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

    # Use nonzero guesses - essential for CG with non-symmetric BC
    parameters['krylov_solver']['nonzero_initial_guess'] = True

    #Folder To Save Plots for Paraview
    #fv=File("JBP Velocity Results Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"theta"+str(theta)+"c0="+str(c0)+"/velocity "+str(t)+".pvd")
 

    # Time/Torque Plot

    if kk==0:
       x=list() # Time values
       y0999=list()
       y09=list()
       y05=list()
       y01=list()
       zy0999=list()
       zy09=list()
       zy05=list()
       zy01=list()
       zx0999=list()
       zx09=list()
       zx05=list()
       zx01=list()
    xx=list()
    yy=list()
    zz=list()
    xxx=list()
    yyy=list()


    # Normal Stress Vector
    tauxx_vec=tauxx.vector().array()

    # Time-stepping
    t = dt
    iter = 0            # iteration counter
    maxiter = 10000000
    while t < Tf + DOLFIN_EPS and iter < maxiter:
        iter += 1
        print"t = %s,  Iteration = %d, Convergence Failures = %s, Loop = %s" %(t, iter, jj, j)


        """if iter >1:

            speed = dot(u0,u0)        # Determine Speed 
            speed = project(speed,Q)  # Project Speed onto FE Space 
            uvals = dot(f1,u0)
            vvals = dot(f2,u0)
            uval = project(uvals,Q)
            vval = project(vvals,Q)

            u_array = uval.vector().array()
            v_array = vval.vector().array()
            eta_array = eta.vector().array()
            xi_array = xi.vector().array()
            for i in range(len(u_array)):  
                eta_array[i] = al1(u_array[i])
                xi_array[i] = al1(v_array[i])
            eta.vector()[:] = eta_array  
            xi.vector()[:] = xi_array

            eta=project(eta,Q)
            xi=project(xi,Q)

            gam = gammah*(hxi*xi*uval+heta*eta*vval)/(2.0*speed+10E-20)
            gam = project(gam,Q)"""

            #print gam.vector().array()
            
            #print norm(gam.vector(), 'linf')
        D=2.0*(sr0)-2.0/3*div(u0)*I
        D=project(D,Zc)
        

        # Compute tentative velocity step
        #begin("Computing tentative velocity")
        A1 = assemble(a1)
        b1= assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]
        solve(A1, u12.vector(), b1, "bicgstab", "default")
        end()
        
        #D=2.0*(sr12)-2.0/3*div(u12)*I
        #D=project(D,Zc)

        # Stress Half STEP
        #A3 = assemble(a3)
        #b3=assemble(L3)
        #[bc.apply(A3, b3) for bc in bctau]
        #solve(A3, tau12.vector(), b3, "bicgstab", "default")
        #end()

    
        
        #Compute Predicted U* Equation
        A2 = assemble(a2)
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcu]
        solve(A2, us.vector(), b2, "bicgstab", "default")
        end()
        #print(norm(us.vector(),'linf'))


        c0c0=c0nd*c0nd*(1.0+al*T0/T_0)          # Nonisothermal Speed of Sound
        c0c0=project(c0c0,Q)
        

        #Continuity Equation 1
        A5 = assemble(a5)
        b5 = assemble(L5)
        [bc.apply(A5, b5) for bc in bcp]
        #[bc.apply(p1.vector()) for bc in bcp]
        solve(A5, p1.vector(), b5, "bicgstab", "default")
        end()


        #Density Update
        ic0c0_array = ic0c0.vector().array()
        c0c0_array = c0c0.vector().array()
        for i in range(len(ic0c0_array)):  
            ic0c0_array[i] = 1.0/c0c0_array[i]
        ic0c0.vector()[:] = ic0c0_array 
        
        rho1=rho0+(p1-p0)*ic0c0
        rho1=project(rho1,Q)


        #Velocity Update
        A7 = assemble(a7)
        b7 = assemble(L7)
        [bc.apply(A7, b7) for bc in bcu]
        solve(A7, u1.vector(), b7, "bicgstab", "default")
        end()



        # Stress Full Step
        #A4 = assemble(a4)
        #b4=assemble(L4)
        #[bc.apply(A4, b4) for bc in bctau]
        #solve(A4, tau1.vector(), b4, "bicgstab", "default")
        #end()

        #print 'Stress Size:', norm(assemble(inner(tau1,R)*dx))
        #print 'Pressure Size:',norm(assemble(inner(p1,q)*dx))
        #print 'Velocity Size:', norm(assemble(inner(u1,v)*dx))

        #Plot Stress Normals

        #tauxx_vec=tauxx.vector().array()
        #for i in range(len(tauxx.vector().array())):
            #tauxx_vec[i]=tau1.vector().array()[4*i+1]
        #tauxx.vector()[:]=tauxx_vec        

        #Temperature Equation Stabilisation DEVSS
        Dt=grad(T0)
        Dt=project(Dt,V) 

        #Temperature Half Step
        A8 = assemble(a8)
        b8 = assemble(L8)
        [bc.apply(A8, b8) for bc in bcT]
        solve(A8, T12.vector(), b8, "bicgstab", "default")
        end()

        #Temperature Equation Stabilisation DEVSS
        Dt=grad(T12)
        Dt=project(Dt,V) 

        #Temperature Update
        A9 = assemble(a9)
        b9 = assemble(L9)
        [bc.apply(A9, b9) for bc in bcT]
        solve(A9, T1.vector(), b9, "bicgstab", "default")
        end()

        # Calculate Size of Artificial Term
        o= tau1.vector()-tau0.vector()                         # Stress Difference per timestep
        h= p1.vector()-p0.vector()
        m=u1.vector()-u0.vector()                              # Velocity Difference per timestep
        l=T1.vector()-T0.vector()



        # Record Error Data 
        
        #if iter > 1:
           #xx.append(t)
           #yy.append(norm(h,'linf')/(norm(p1.vector())+10E-10))
           #zz.append(norm(o,'linf')/(norm(tau1.vector())+10E-10))
           #xxx.append(norm(m,'linf')/(norm(u1.vector())+10E-10))
           #yyy.append(norm(l,'linf')/(norm(T1.vector())+10E-10))


        # Save Plot to Paraview Folder 
        #for i in range(5000):
            #if iter== (0.01/dt)*i:
               #fv << u1
        #ft << T1

        # Break Loop if code is diverging

        if max(norm(T1.vector(), 'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) > 10E6 or np.isnan(sum(T1.vector().array())) or np.isnan(sum(u1.vector().array())):
            print 'FE Solution Diverging'   #Print message 
            with open("Compressible Stability.txt", "a") as text_file:
                 text_file.write("Iteration:"+str(j)+"--- Re="+str(Rey)+", We="+str(We)+", beta="+str(betav)+", c0="+str(c0)+", t="+str(t)+", dt="+str(dt)+'\n')
            #dt=dt/2                         #Use Smaller timestep 
            thetat=thetat*10
            j-=1                            #Extend loop
            jj+= 1                          # Convergence Failures
            break

        # Plot solution
        #if t>0.2:
            #plot(tauxx, title="Normal Stress", rescale=True)         # Normal Stress
            #plot(div(u1), title="Compression", rescale=True )
            #plot(p1, title="Pressure", rescale=True)                # Pressure
            #plot(rho1, title="Density", rescale=True)               # Density
            #plot(u1, title="Velocity", rescale=True, mode = "auto") # Velocity
            #plot(T1, title="Temperature", rescale=True)             # Temperature
        
        # Record Torque Data 
        """if kk==3:
           x.append(t)
           y0999.append(assemble(innertorque))
           zx0999.append(assemble(innerforcex))
           zy0999.append(assemble(innerforcey))

        if kk==2:
           y09.append(assemble(innertorque))
           zx09.append(assemble(innerforcex))
           zy09.append(assemble(innerforcey))

        if kk==1:
           y05.append(assemble(innertorque))
           zx05.append(assemble(innerforcex))
           zy05.append(assemble(innerforcey))

        if kk==0:
           y01.append(assemble(innertorque))
           zx01.append(assemble(innerforcex))
           zy01.append(assemble(innerforcey))"""
           
        #print 'torque:', assemble(innertorque)
        #print 'Fx:', assemble(innerforcex)
        #print 'Fy:', assemble(innerforcey)


        # Move to next time step
        u0.assign(u1)
        T0.assign(T1)
        rho0.assign(rho1)
        p0.assign(p1)
        tau0.assign(tau1)
        w.t=t
        Ret.t=t
        Wet.t=t
        t += dt


        # Plot Pressure Convergence Data
    if  max(norm(T1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) > 10E6 and dt < 0.02:
        fig1=plt.figure()
        plt.plot(xx, yy, 'r-', label='Pressure Timestep Error')
        plt.xlabel('time(s)')
        plt.ylabel('||p1-p0||/||p1||')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/PressureTimestepErrorRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"dt="+str(dt)+".png")
        plt.clf()
        plt.plot(xx, zz, 'r-', label='Stress Timestep Error')
        plt.xlabel('time(s)')
        plt.ylabel('||S1-S0||/||S1||')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/StressCovergenceRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"dt="+str(dt)+".png")
        plt.clf()
        plt.plot(xx, xxx, 'g-', label='Velocity Field Timestep Error')
        plt.xlabel('time(s)')
        plt.ylabel('||u1-u0||/||u1||')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/VelocityCovergenceRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"dt="+str(dt)+".png")
        plt.clf()
        plt.plot(xx, yyy, 'g-', label='Velocity Field Timestep Error')
        plt.xlabel('time(s)')
        plt.ylabel('||T1-T0||/||T1||')
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/TemperatureCovergenceRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"dt="+str(dt)+".png")
        plt.clf()    




    if max(norm(T1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) < 10E6 and dt < 0.02 and kk==3:

        # Plot Torque Data
        plt.plot(x, y0999, 'r-', label=r'$\beta_s=1$')
        plt.plot(x, y09, 'b-', label=r'$\beta_s=0.9$')
        plt.plot(x, y05, 'g-', label=r'$\beta_s=0.5$')
        plt.plot(x, y01, 'm-', label=r'$\beta_s=0.1$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('Torque')
        #plt.show()
        plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Torque We="+str(We)+"Re="+str(Rey)+"c0="+str(c0)+"al="+str(al)+"t="+str(t)+".png")
        plt.clf()
        plt.plot(x, zx0999, 'r-', label=r'$\beta_s=1$')
        plt.plot(x, zx09, 'b-', label=r'$\beta_s=0.9$')
        plt.plot(x, zx05, 'g-', label=r'$\beta_s=0.5$')
        plt.plot(x, zx01, 'm-', label=r'$\beta_s=0.1$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('Horzontal Load Force')
        plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Horizontal Load We="+str(We)+"Re="+str(Rey)+"c0="+str(c0)+"al="+str(al)+"t="+str(t)+".png")
        plt.clf()
        plt.plot(x, zy0999, 'r-', label=r'$\beta_s=1$')
        plt.plot(x, zy09, 'b-', label=r'$\beta_s=0.9$')
        plt.plot(x, zy05, 'g-', label=r'$\beta_s=0.5$')
        plt.plot(x, zy01, 'm-', label=r'$\beta_s=0.1$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('Vertical Load Force')
        plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Vertical Load We="+str(We)+"Re="+str(Rey)+"c0="+str(c0)+"al="+str(al)+"t="+str(t)+".png")
        plt.clf()
        plt.plot(zx0999, zy0999, 'r-', label=r'$\beta_s=1$')
        plt.plot(zx09, zy09, 'b-', label=r'$\beta_s=0.9$')
        plt.plot(zx05, zy05, 'g-', label=r'$\beta_s=0.5$')
        plt.plot(zx01, zy01, 'm-', label=r'$\beta_s=0.1$')
        plt.legend(loc='best')
        plt.xlabel('Fx')
        plt.ylabel('Fy')
        plt.savefig("Compressible Viscoelastic Flow Results/Load-Torque/Force We="+str(We)+"Re="+str(Rey)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"t="+str(t)+".png")
        plt.clf()
        #quit()

    torque = assemble(innertorque)
    Fx = assemble(innerforcex)
    Fy = assemble(innerforcey)


        # Write LOad & Torque Values to txt
    if max(norm(T1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) < 10E6 and dt < 0.0001:
        with open("Compressible Flow Drag.txt", "a") as text_file:
             text_file.write("Rj="+str(r_a)+" Rb="+str(r_b)+" Eccentricity="+str(ecc)+" Angular Vel="+str(w_j)+", We="+str(We)+", beta="+str(betav)+", c0="+str(c0)+", t="+str(t)+", dt="+str(dt)+'\n')
             text_file.write("Torque:"+str(torque)+'\n')
             text_file.write("Fx (Horizontal Load):"+str(Fx)+'\n')
             text_file.write("Fy (Verical Load):"+str(Fy)+'\n')

    if max(norm(T1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) < 10E6 and dt < 0.02:
        # Matlab Plot of the Solution at t=Tf
        rho1=rho_0*rho1
        rho1=project(rho1,Q)
        p1=mu_0*(U/(L))*p1  #Dimensionalised Pressure
        p1=project(p1,Q)
        mplot(rho1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/DensityBi="+str(Bi)+"Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"t="+str(t)+".png")
        plt.clf()
        mplot(p1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/PressureBi="+str(Bi)+"Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"t="+str(t)+".png")
        plt.clf()
        mplot(T1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/TemperatureBi"+str(Bi)+"Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"t="+str(t)+".png")
        plt.clf()
        divu=project(div(u1),Q)
        mplot(divu)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/CompressionBi="+str(Bi)+"Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"t="+str(t)+".png")
        plt.clf()



        #Plot PRESSURE Contours USING MATPLOTLIB
        # Scalar Function code

        #Set Values for inner domain as -infty
        u1=U*u1
        u1=project(u1,V)
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
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/Pressure Contours Bi="+str(Bi)+" Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"t="+str(t)+".png")
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
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/Temperature Contours Bi="+str(Bi)+"Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"t="+str(t)+".png")
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
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/Velocity ContoursBi="+str(Bi)+"Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"al="+str(al)+"t="+str(t)+".png")   
        plt.clf()                                                                     # display the plot

    # Update Control Parameter

    plt.close()


    if dt < tol:
       j=loopend+1
       break

    if jj==2:
       We=We/10


    # Viscosity Fraction Update
    """if max(norm(T1.vector(), 'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) < 10E6 or dt<tol:
        #with open("Continuation Compressible Stability.txt", "a") as text_file:
             #text_file.write("Solution Converges Re"+str(Rey)+", We="+str(We)+", beta="+str(betav)+", dt="+str(dt)+'\n')
        dt = 0.001 #Time Stepping                        #Go back to Original Timestep
        jj = 0
        if betav==0.9999:
           betav=0.1
           #kk=4
        if betav==0.9:
           kk=3
           betav=0.9999
        if betav==0.5:
           kk=2
           betav=0.9
        if betav==0.1:
           #kk=1
           betav=0.9999"""


    # Speed of Sound Update
    if kk==5 or dt<tol:
        #with open("Continuation Compressible Stability.txt", "a") as text_file:
             #text_file.write("Solution Converges Re"+str(Rey)+", We="+str(We)+", beta="+str(betav)+", dt="+str(dt)+'\n')
        dt = 0.002 #Time Stepping                        #Go back to Original Timestep
        #c0=c0/2
        betav=1
        kk=0
        j=0
        if c0==250:
           j=loopend+1
           break
        if c0==750:
           c0=500
        if c0==1000:
           c0=750
        if c0==1500:
           c0=1000

    if kk==0 or dt<tol:
        #with open("Continuation Compressible Stability.txt", "a") as text_file:
             #text_file.write("Solution Converges Re"+str(Rey)+", We="+str(We)+", beta="+str(betav)+", dt="+str(dt)+'\n')
        dt = 0.002 #Time Stepping                        #Go back to Original Timestep
        #c0=c0/2
        betav=1
        kk=0
        if Bi==5:
           Bi=7.5
           j=loopend+1
           break
        if Bi==2.5:
           Bi=5
        if Bi==1:
           Bi=2.5
        if Bi==0.5:
           Bi=1






    # Update Control Variables 
    """if max(norm(T1.vector(), 'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) < 10E6 and np.isfinite(sum(u1.vector().array())):
        with open("Compressible Stability.txt", "a") as text_file:
             text_file.write("Solution Converges Re"+str(Rey)+", We="+str(We)+", beta="+str(betav)+", dt="+str(dt)+'\n')
        dt = 0.064 #Time Stepping                        #Go back to Original Timestep
        #gammah=10E0
        jj=0
        if w_j==2000.0:
            w_j=2.5*w_j
            lambda1=5.0*L/(r_a*w_j) 
        elif w_j==1000.0:
            w_j=2.0*w_j
            lambda1=5.0*L/(r_a*w_j) 
        elif w_j==500.0:
            w_j=2.0*w_j
            lambda1=5.0*L/(r_a*w_j) 
        elif w_j==250.0:
            w_j=2.0*w_j
            lambda1=5.0*L/(r_a*w_j) 
        elif w_j==100.0:
            w_j=2.5*w_j
            lambda1=5.0*L/(r_a*w_j) 
        elif w_j==50.0:
            w_j=2.0*w_j
            lambda1=5.0*L/(r_a*w_j) 
        elif w_j==25.0:
            w_j=2.0*w_j
            lambda1=5.0*L/(r_a*w_j) 
        elif w_j==10.0:
            w_j=2.5*w_j
            lambda1=5.0*L/(r_a*w_j) 
        elif w_j==5.0:
            w_j=2.0*w_j
            lambda1=5.0*L/(r_a*w_j) 
        elif w_j==2.5:
            w_j=2.0*w_j
            lambda1=5.0*L/(r_a*w_j) """

         

    """if max(norm(T1.vector(), 'linf'),norm(p1.vector(), 'linf')) < 10E6:
       with open("Compressible Weissenberg  Stability.txt", "a") as text_file:
            text_file.write("Solution Converges Re"+str(Rey)+", We="+str(We)+", dt="+str(dt)+'\n')
       dt = 0.064
       jj=0
       if lambda1=50.0*(r_b-r_a)/(r_a*w_j):
            lambda1=2.0*lambda_1
       elif lambda1=20.0*(r_b-r_a)/(r_a*w_j):
            lambda1=2.5*lambda_1
       elif lambda1=10.0*(r_b-r_a)/(r_a*w_j):
            lambda1=2*lambda_1
       elif lambda1=5.0*(r_b-r_a)/(r_a*w_j):
            lambda1=2*lambda_1
       elif lambda1=1.0*(r_b-r_a)/(r_a*w_j):
            lambda1=5*lambda_1
       elif lambda1=0.1*(r_b-r_a)/(r_a*w_j):
            lambda1=10*lambda_1
       elif lambda1=0.01*(r_b-r_a)/(r_a*w_j):
            lambda1=10*lambda_1"""

