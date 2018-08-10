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
r_a=0.03125 #Journal Radius
r_b=0.04125 #Bearing Radius
x1=-0.005
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
mesh = generate_mesh(cyl, 50)
meshc= generate_mesh(c3, 25)

# ADAPTIVE MESH REFINEMENT (METHOD 2) "MESH"

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


print'Number of Cells:', mesh.num_cells()
print'Number of Vertices:', mesh.num_vertices()

                                                                              
class Omega0(SubDomain):
      def inside(self, x, on_boundary):
          return True if (x[0]-x1)**2+(x[1]-y1)**2 < (0.92*r_a**2+0.08*r_b**2) and on_boundary  else False  # and 
omega0= Omega0()

class Omega1(SubDomain):
      def inside(self, x, on_boundary):
          return True if (x[0]-x2)**2 + (x[1]-y2)**2 > (0.07*r_a**2+0.93*r_b**2) and on_boundary else False  #
omega1= Omega1()

# Create mesh functions over the cell facets (Verify Boundary Classes)
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
sub_domains.set_all(3)
omega0.mark(sub_domains, 0)
omega1.mark(sub_domains, 1)
plot(sub_domains, interactive=False)
#quit()


# Define function spaces (P2-P1)
d=2

V = VectorFunctionSpace(mesh, "CG", d)
Q = FunctionSpace(mesh, "CG", d)
W = FunctionSpace(mesh, "CG", d)
Z = TensorFunctionSpace(mesh, "CG", d)



#Control Variables (to be changed each time code runs)
c0 = 1500

n=5
for j in range(n):

    # Define trial and test functions
    u = TrialFunction(V)
    rho=TrialFunction(Q)
    p = TrialFunction(Q)
    T = TrialFunction(W)
    v = TestFunction(V)
    q = TestFunction(Q)
    r = TestFunction(W)
    tau = TrialFunction(Z)
    D = TrialFunction(Z)
    R = TestFunction(Z)



    #Define Discretised Functions

    u0=Function(V)
    us=Function(V)
    u12=Function(V)
    u1=Function(V)
    rho0=Function(Q)
    rho1=Function(Q)
    p00=Function(Q)
    p0=Function(Q)
    p1=Function(Q)
    mu=Function(W)
    T0=Function(W)
    T1=Function(W)
    tau0=Function(Z)
    tau12=Function(Z)
    tau1=Function(Z)

    D1=Function(Z)

    # Set parameter values
    hm = mesh.hmin()
    hM = mesh.hmax()
    dt = 0.005                 #Time Stepping [s]
    Tf = 3.0                   #Stopping Time [s] 
    rho_0 = 820.0              #Density [kg/m^3]
    Cv = 1000.0                #Heat Capacity [J/(g.K)]
    mu1 = 25.0*10E-2           #Solvent KINEMATIC Viscosity [Pa.s]
    mu2 = 25.0*10E-2           #Polymeric Viscosity [Pa.s]
    lambda1 = 2.0*10E-5        #Relaxation Time [s]
    w_j = 75.0                 #Angular VELOCITY (inner bearing) [rad/s]
    T00 = 300                  #Reference Temperature
    kappa = 2.0                #Thermal condusctivity  
    heatt= 5.0                 #Heat Transfer Coefficient
    mu = mu1+mu2               # Total Viscosity 
    betav = mu1/mu             # Viscosity Ratio (Viscoelastic Fluid)
    Rc = 3.33*10E1
    T_0 = 300.0 
    T_h = 350.0                #Reference temperature
    C=250.0                    #Sutherland's Constant
    kappa = 2.0
    Pr=20.0                    #Prandtl Number
    Ra=60.0                    #Rayleigh Number
    V_h=0.01                   #Viscous Heating Number
    kappa = 0.2
    beta = 69*10E-2            # Thermal Expansion Coefficient
    alpha=1.0/(rho_0*Cv)
    #c0=100.0                  # Speed of Sound
    k = Constant(dt)           # Constant Time Stepping 
    U = w_j*r_a

    P_0 = 1.0                  # Initial Pressure Field

    eta = lambda1/dt           #Ratio of Weissenberg number to time step

    # Nondimensional Parameters

     # Non Thermal
    Re = rho_0*w_j*c/mu                                       #Reynolds Number
    We = lambda1*w_j/c                                        #Weisenberg NUmber
    De = lambda1*w_j                                          #Deborah Number 
    tstar = r_a*w_j/c                                         #Nondimensional Time
    theta = 1.0*10E-5*mu1                                               #Momentum Equation Stabilisation term size
    thetat = 1.0*10E-10*kappa                                   #Energy Equation Stabilisation term size


    #Print Parameter Values

    print'Minimum Cell Diameter:', hm
    print'Eccentricity:' ,ec
    print'Radius DIfference:',c
    print'Eccentricity Ratio:',ecc
    print'Reynolds Number:',Re
    print'Weissenberg Number:',We
    print'Deborah Number:',De
    print'Viscosity Ratio:', betav
    print'Speed of sound ^2 x Time Step:', c0*c0*dt
    #quit()
     


    #Inner cylinder velocity (TIME DEPENDENT)
    td= Constant('5')
    e = Constant('6')

    w = Expression(('(0.5*(1.0+tanh(e*t-2.5)))*w_j*(x[1]-y1)' , '-(0.5*(1.0+tanh(e*t-2.5)))*w_j*(x[0]-x1)' ), d=d, degree=d, w_j=w_j , r_a=r_a, x1=x1, y1=y1, e=e , t=0.0)

    # Define unit Normal/tangent Vector at inner and outer Boundary (Method 2)
    n0 =  Expression(('(x[0]-x1)/r_a' , '(x[1]-y1)/r_a' ), d=d, degree=d, r_a=r_a, x1=x1, y1=y1)
    n1 =  Expression(('(x[0]-x2)/r_b' , '(x[1]-y2)/r_b' ), d=d, degree=d, r_b=r_b, x2=x2, y2=y2)
    t0 =  Expression(('(x[1]-y1)/r_a' , '-(x[0]-x1)/r_a' ),d=d, degree=d, r_a=r_a, x1=x1, y1=y1)
    t1 =  Expression(('(x[1]-y2)/r_b' , '-(x[0]-x2)/r_b' ),d=d, degree=d, r_b=r_b, x2=x2, y2=y2)

    n0v = interpolate(n0, V)
    I = Expression((('1.0','0.0'),
                    ('0.0','1.0')), degree=2)

    # Define boundary conditions
    spin =  DirichletBC(V, w, omega0)  #The inner cylinder will be rotated with constant angular velocity w_a
    noslip  = DirichletBC(V, (0.0, 0.0), omega1) #The outer cylinder remains fixed with zero velocity 
    temp0 =  DirichletBC(Q, 0.0, omega0)    #Dirichlet Boundary Condition on the inner bearing 


    #Collect Boundary Conditions
    bcu = [spin, noslip]
    bcp = []
    bcT = [temp0]
    bctau = []


    # Initial Density Field
    rho_array = rho0.vector().array()
    for i in range(len(rho_array)):  
        rho_array[i] = rho_0
    rho0.vector()[:] = rho_array 

    # Initial Temperature Field
    T_array = T0.vector().array()
    for i in range(len(T_array)):  
        T_array[i] = T_0
    T0.vector()[:] = T_array

      
    I = Expression((('1.0','0.0'),
                    ('0.0','1.0')), d=d, degree=d)


    #Define Variable Parameters, Strain Rate and other tensors
    sr = 0.5*(grad(u) + transpose(grad(u)))
    sr0 = 0.5*(grad(u0) + transpose(grad(u0)))
    sr12 = 0.5*(grad(u12) + transpose(grad(u12)))
    sr1 = 0.5*(grad(u1) + transpose(grad(u1)))
    F0 = (grad(u0)*tau0 + tau0*transpose(grad(u0)))
    F12 = (grad(u12)*tau + tau*transpose(grad(u12)))
    F1 = (grad(u1)*tau + tau*transpose(grad(u1)))
    gamdot0 = inner(sr0,grad(u0))
    gamdots = inner(sr1,grad(u0))
    gamdotp = inner(tau1,grad(u0))
    theta0 = (T0-T_0)/(T_h-T_0)
    alpha = 1.0/(rho*Cv)



    # Artificial Diffusion Term
    o= tau1.vector()-tau0.vector()                         # Stress Difference per timestep
    h= p1.vector()-p0.vector()
    m=u1.vector()-u0.vector()                              # Velocity Difference per timestep
    l=T1.vector()-T0.vector()
    alt=norm(o)/(norm(tau1.vector())+10E-10)
    alp=norm(h)/(norm(p1.vector())+10E-10)
    alu=norm(m)/(norm(u1.vector())+10E-10)
    alT=norm(l, 'linf')/(norm(T1.vector(),'linf')+10E-10)
    epstau = alt*betav+10E-8                                    #Stabilisation Parameter (Stress)
    epsp = alp*betav+10E-8                                      #Stabilisation Parameter (Pressure)
    epsu = alu*betav+10E-8                                      #Stabilisation Parameter (Stress)
    epsT = 0.5*alT*kappa+10E-8                                  #Stabilisation Parameter (Temperature)

    # TAYLOR GALERKIN METHOD (COMPRESSIBLE VISCOELASTIC)

    # Weak Formulation (DEVSS in weak formulation)

    """DEVSS Stabilisation used in 'Extra Stress' and 'Velocity' computation stages of the Taylor Galerkin Scheme"""
    """Artificial DIffusion Stabilisation used """

    #Stabilisation terms (DEVSS)

    """Momentum Equation Stabilisation"""

    D=2*(sr0-1.0/3*div(u0)*I)
    D=project(D,Z)

    """Temperature Equation Stabilisation"""
    Dt=grad(T0)
    Dt=project(Dt,V)

    """Stbilisation Used"""
    """Stabilisation Term: +(theta)*(inner(grad(u),grad(v))*dx+1.0/3*inner(div(u),div(v))*dx), +(theta)*inner(D,grad(v))*dx"""

    #Half Step
    a1=(1/(dt/2.0))*inner(rho0*u,v)*dx+mu1*(inner(grad(u),grad(v))*dx+1.0/3*inner(div(u),div(v))*dx)+(theta)*(inner(grad(u),grad(v))*dx+1.0/3*inner(div(u),div(v))*dx)
    L1=(1/(dt/2.0))*inner(rho0*u0,v)*dx-inner(rho0*grad(u0)*u0,v)*dx+inner(p0,div(v))*dx-inner(tau0,grad(v))*dx+(theta)*inner(D,grad(v))*dx

    """No Stbilisation Used"""
    """Stabilisation Term: +(theta)*(inner(grad(u),grad(v))*dx+1.0/3*inner(div(u),div(v))*dx), +(theta)*inner(D,grad(v))*dx"""

    #Predicted U* Equation
    a2=(1/dt)*inner(rho0*u,v)*dx +(theta)*(inner(grad(u),grad(v))*dx+1.0/3*inner(div(u),div(v))*dx)
    L2=(1/dt)*inner(rho0*u0,v)*dx-0.5*mu1*(inner(grad(u0),grad(v))*dx+1.0/3*inner(div(u0),div(v))*dx)-inner(rho0*grad(u12)*u12,v)*dx+inner(p0,div(v))*dx-0.5*inner(tau0,grad(v))*dx \
       +(theta)*inner(D,grad(v))*dx

    # Stress Half Step
    a3 = (2*eta)*inner(tau,R)*dx + lambda1*(inner(dot(u12,grad(tau)),R)*dx + inner(F12, R)*dx+inner(div(u12)*tau,R)*dx)
    L3 =  2*eta*inner(tau0,R)*dx + 2.0*mu2*inner(sr0,R)*dx 

    """Stbilisation Used"""
    """Stabilisation Term: +epstau*inner(grad(tau),grad(R))*dx"""

    # Stress Full Step
    a4 = (eta+1)*inner(tau,R)*dx + lambda1*(inner(dot(u1,grad(tau)),R)*dx + inner(F1, R)*dx+inner(div(u1)*tau,R)*dx)
    L4 = eta*inner(tau0,R)*dx + 2.0*mu2*inner(sr0,R)*dx 

    #Continuity Equation 1
    a5=(1.0/(c0*c0*dt))*inner(p,q)*dx+0.5*dt*inner(grad(p),grad(q))*dx   #Using Dynamic Speed of Sound (c=c(x,t))
    L5=(1.0/(c0*c0*dt))*inner(p0,q)*dx+0.5*dt*inner(grad(p0),grad(q))*dx-(inner(rho0*div(us),q)*dx+inner(dot(grad(rho0),us),q)*dx)

    #Continuity Equation 2 
    a6=c0*c0*inner(rho,q)*dx 
    L6=c0*c0*inner(rho0,q)*dx + inner(p1-p0,q)*dx 

    """ Stbilisation Used"""
    """Stabilisation Term: +(theta)*(inner(grad(u),grad(v))*dx+1.0/3*inner(div(u),div(v))*dx), +(theta)*inner(D1,grad(v))*dx"""

    #Velocity Update
    a7=(1/dt)*inner(rho0*u,v)*dx+0.5*mu1*(inner(grad(u),grad(v))*dx+1.0/3*inner(div(u),div(v))*dx)+ (theta)*(inner(grad(u),grad(v))*dx+1.0/3*inner(div(u),div(v))*dx)
    L7=(1/dt)*inner(rho0*us,v)*dx+0.5*(inner(p1,div(v))*dx-inner(p0,div(v))*dx)-0.5*inner(tau1,grad(v))*dx +(theta)*inner(D,grad(v))*dx

    
    # Temperature Update
    a8 = ((Cv/dt)*inner(rho1*T,r)*dx + inner(rho1*inner(u1,grad(T)),r)*dx) + (kappa)*inner(grad(T),grad(r))*dx + thetat*inner(grad(T),grad(r))*dx
    L8 = (Cv/dt)*inner(rho1*T0,r)*dx + (kappa)*(heatt*(inner(grad(T0),n1*r)*ds(1)) +(mu1*inner(gamdots,r)*dx + inner(gamdotp,r)*dx)-mu1*inner(p1*div(u1),r)*dx)+ thetat*inner(Dt,grad(r))*dx
          #+ inner(,r)*dx  #Neumann Condition on the outer bearing is encoded in the weak formulation


    # Assemble matrices
    #A0 = assemble(a0)
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)
    A4 = assemble(a4)
    A5 = assemble(a5)
    A6 = assemble(a6)
    A7 = assemble(a7)
    A8 = assemble(a8)

    #Define Boundary 

    boundary_parts = FacetFunction("size_t", mesh)
    omega0.mark(boundary_parts,0)
    omega1.mark(boundary_parts,1)
    ds = Measure("ds")[boundary_parts]

    sigma0 = (p1*I + 2*betav*sr1+tau1)*t0
    sigma1 = (p1*I + 2*betav*sr1+tau1)*t1

    omegaf0 = p1*I*n0  #Nomral component of the stress 
    omegaf1 = p1*I*n1


    innerforcex = -inner(Constant((1.0, 0.0)), omegaf0)*ds(0)
    innerforcey = -inner(Constant((0.0, 1.0)), omegaf0)*ds(0)
    outerforcex = -inner(Constant((1.0, 0.0)), omegaf1)*ds(1)
    outerforcey = -inner(Constant((0.0, 1.0)), omegaf1)*ds(1)
    innertorque = -inner(n0, sigma0)*ds(0)
    outertorque = -inner(n1, sigma1)*ds(1)

    stab = assemble((theta)*(inner(grad(u1),grad(v))*dx+1.0/3*inner(div(u1),div(v))*dx-inner(D,grad(v))*dx))

    # Use amg preconditioner if available
    prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

    # Use nonzero guesses - essential for CG with non-symmetric BC
    parameters['krylov_solver']['nonzero_initial_guess'] = True

    # Time-stepping
    t = dt

    # Time/Torque Plot
    x=list()
    y=list()
    z=list()
    xx=list()
    yy=list()
    yyy=list()
    zz=list()
    zzz=list()
    zzzz=list()

    # Time-stepping
    iter = 0            # iteration counter
    maxiter = 10000
    while t < Tf + DOLFIN_EPS and iter < maxiter and norm(u1.vector(), 'linf') < 2*10E4:
        iter += 1
        print"t = %s,  Iteration = %d, Loop = %s" %(t, iter, j)
        

        # Stabilisation DEVSS

        #A0 = assemble(a0)
        #b0 = assemble(L0)
        #[bc.apply(A1, b0) for bc in bctau]
        #solve(A0, D1.vector(), b0, "bicgstab", "default")
        #end()


        # Compute tentative velocity step
        #begin("Computing tentative velocity")
        A1 = assemble(a1)
        b1 = assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]
        solve(A1, u12.vector(), b1, "bicgstab", "default")
        end()
        
        #print(norm(u12.vector(),'linf'))
        
        #Compute Predicted U* Equation
        A2 = assemble(a2)
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcu]
        solve(A2, us.vector(), b2, "bicgstab", "default")
        end()
        #print(norm(us.vector(),'linf'))

        # Stress Half STEP
        b3=assemble(L3)
        [bc.apply(A3, b3) for bc in bctau]
        solve(A3, tau12.vector(), b3, "bicgstab", "default")
        end()

        # Stress Full Step
        b4=assemble(L4)
        [bc.apply(A4, b4) for bc in bctau]
        solve(A4, tau1.vector(), b4, "bicgstab", "default")
        end()
        
        #Continuity Equation 1
        A5 = assemble(a5)
        b5 = assemble(L5)
        [bc.apply(A5, b5) for bc in bcp]
        #[bc.apply(p1.vector()) for bc in bcp]
        solve(A5, p1.vector(), b5, "bicgstab", "default")
        end()

        #Continuity Equation 2
        A6 = assemble(a6)
        b6 = assemble(L6)
        [bc.apply(A6, b6) for bc in bcp]
        solve(A6, rho1.vector(), b6, "bicgstab", "default")
        end()


        #Velocity Update
        A7 = assemble(a7)
        b7 = assemble(L7)
        [bc.apply(A7, b7) for bc in bcu]
        solve(A7, u1.vector(), b7, "bicgstab", "default")
        end()

        #Temperature Equation
        A8 = assemble(a8)
        b8 = assemble(L8)
        [bc.apply(A8, b8) for bc in bcT]
        solve(A6, T1.vector(), b6, "bicgstab", "default")
        end()

        if max(norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) > 10E6:
           print 'FEM Solution Diverging'
           theta=10*theta
           break
           

        o= tau1.vector()-tau0.vector()
        h= p1.vector()-p0.vector()
        m=u1.vector()-u0.vector()
        mm=T1.vector()-T0.vector()

        # Record Torque Data 
        
        if iter > 1:
           xx.append(t)
           yy.append(norm(h,'linf')/norm(p1.vector(),'linf'))
           zz.append(norm(o,'linf')/norm(tau1.vector(),'linf'))
           zzz.append(norm(m,'linf')/norm(u1.vector()))
           yyy.append(norm(mm,'linf')/norm(T1.vector()))

        
        # Calcultate Force on the bearing
        #print("Normal Force on inner bearing: ",  (assemble(innerforcex), assemble(innerforcey)))
        #print("Torque on inner bearing: ", assemble(innertorque))

        # Record Torque Data 
        x.append(t)
        y.append(assemble(innertorque))
        z.append(assemble(innerforcey))
        zzzz.append(stab.sum())
        
        print(stab.sum())
        # Plot solution
        #plot(rho1, title="Density Field", rescale=True)
        #plot(p1, title= "Pressure Field", rescale= True)
        #plot(u1, title="Velocity", rescale=True, mode = "auto")
        #plot(T1, title="Temperature", rescale=True)

        # Move to next time step
        u0.assign(u1)
        T0.assign(T1)
        p0.assign(p1)
        rho0.assign(rho1)
        tau0.assign(tau1)
        w.t=t
        t += dt

    # Plot Pressure Convergence Data
    fig1=plt.figure()
    plt.plot(xx, yy, 'r-', label='Pressure Convergence')
    plt.xlabel('time(s)')
    plt.ylabel('||p1-p0||/||p1||')
    plt.savefig("Compressible Flow Results/Pressure Timestep Error We="+str(We)+"Re="+str(Re)+"c0="+str(c0)+"t="+str(t)+".png")
    plt.clf()
    plt.plot(xx, zz, 'r-', label='Extra Stress Timestep Error')
    plt.xlabel('time(s)')
    plt.ylabel('||tau1-tau0||/||tau1||')
    plt.savefig("Compressible Flow Results/Extra Stress Timestep Error We="+str(We)+"Re="+str(Re)+"c0="+str(c0)+"t="+str(t)+".png")
    plt.clf()
    plt.plot(xx, zzz, 'g-', label='Velocity Field Timestep Error')
    plt.xlabel('time(s)')
    plt.ylabel('||u1-u0||/||u1||')
    plt.savefig("Compressible Flow Results/Velocity Covergence Timestep Error We="+str(We)+"Re="+str(Re)+"c0="+str(c0)+"t="+str(t)+".png")
    plt.clf()
    plt.plot(x, zzzz, 'g-', label='Stabilisation term size')
    plt.xlabel('time(s)')
    plt.ylabel('theta')
    plt.savefig("Compressible Flow Results/Stabilisation term size We="+str(We)+"Re="+str(Re)+"c0="+str(c0)+"t="+str(t)+".png")
    plt.clf()
    plt.plot(xx, yyy, 'g-', label='Temperature Timestep Error')
    plt.xlabel('time(s)')
    plt.ylabel('||T1-T0||/||T1||')
    plt.savefig("Compressible Flow Results/Temperature Timestep Error We="+str(We)+"Re="+str(Re)+"c0="+str(c0)+"t="+str(t)+".png")
    plt.clf()
    #quit()

    # Impulse data (Trapezium Rule)
    yarr=np.asarray(y)
    impy=sum(yarr)
    zarr=np.asarray(z)
    impz=sum(zarr)
    print('Angular Impulse:', dt*impy)
    print('Normal Impulse:', dt*impz)
 

    # Matlab Plot of the Solution at t=Tf
    mplot(p1)
    plt.colorbar()
    plt.savefig("Compressible Flow Results/Pressure We="+str(We)+"Re="+str(Re)+"c0="+str(c0)+"t="+str(t)+".png")
    plt.clf()
    mplot(T1)
    plt.colorbar()
    plt.savefig("Compressible Flow Results/Temperature We="+str(We)+"Re="+str(Re)+"c0="+str(c0)+"t="+str(t)+".png")
    plt.clf()


    # Plot Torque Data
    plt.plot(x, y, 'r-', label='Torque')
    plt.xlabel('time(s)')
    plt.ylabel('Torque')
    plt.savefig("Compressible Flow Results/Torque We="+str(We)+"Re="+str(Re)+"c0="+str(c0)+"t="+str(t)+".png")
    plt.clf()
    plt.plot(x, z, 'b-', label='Vertical Force')
    plt.xlabel('time(s)')
    plt.ylabel('Vertical Force')
    plt.savefig("Compressible Flow Results/Vertical Load We="+str(We)+"Re="+str(Re)+"c0="+str(c0)+"t="+str(t)+".png")
    plt.clf()


    #Plot PRESSURE Contours USING MATPLOTLIB
    # Scalar Function code

    #Set Values for inner domain as -infty

    x = Expression('x[0]', d=d, degree=d)  #GET X-COORDINATES LIST
    y = Expression('x[1]', d=d, degree=d)  #GET Y-COORDINATES LIST
    Q1=FunctionSpace(meshc, "CG", 2)
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
    plt.savefig("Compressible Flow Results/Pressure Contours We="+str(We)+"Re="+str(Re)+"c0="+str(c0)+"t="+str(t)+".png")
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
    plt.savefig("Compressible Flow Results/Temperature Contours We="+str(We)+"Re="+str(Re)+"c0="+str(c0)+"t="+str(t)+".png")
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
                   density=3,              
                   color=speed,  
                   cmap=cm.gnuplot,                         # colour map
                   linewidth=1.0)       # line thickness
    plt.colorbar()
    plt.title('Journal Bearing Problem')
    plt.savefig("Compressible Flow Results/Velocity Contours We="+str(We)+"Re="+str(Re)+"c0="+str(c0)+"t="+str(t)+".png")   
    plt.clf()                                                                     # display the plot

    # Update Control Parameter

    c0=c0/2                      


