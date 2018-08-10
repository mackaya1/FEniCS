"""Test for Convergence of Numerical Scheme"""
"""Ernesto Castillo - PHD Thesis 2016"""

"""Base Code for the Incompressible flow Numerical Scheme Test"""

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

loopend = 4
j = 0
jj = 0
jjj = 0

m_size = list()
u_errl2 = list()
u_errL2 = list()
u_errlinf = list()
u_errLinf = list()

tau_errl2 = list()
tau_errL2 = list()
tau_errlinf = list()
tau_errLinf = list()

p_errl2 = list()
p_errL2 = list()
p_errlinf = list()
p_errLinf = list()

while j < loopend:
    j+=1
    t=0.0

    # Mesh refinement comparison Loop

    if j==1:
        mm = 30
    if j==2:
        mm = 40
    if j==3:
        mm = 50
    if j==4:
        mm = 60


     
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

    for i in range(0):
          g = (max(x_1,y_1)-max(x_0,y_0))*0.05/(i+1)
          cell_domains = CellFunction("bool", base_mesh)
          cell_domains.set_all(False)
          for cell in cells(base_mesh):
              x = cell.midpoint()
              if  (x[0] < x_0+g or x[1] < y_0+g) or (x[0] > x_1-g or x[1] > y_1-g): # or (x[0] < x0+g and x[1] < y0+g)  or (x[0] > x1-g and x[1] < g): 
                  cell_domains[cell]=True
          #plot(cell_domains, interactive=True)
          base_mesh = refine(base_mesh, cell_domains, redistribute=True)

    # Choose Mesh to Use

    mesh = base_mesh

    class No_slip(SubDomain):
          def inside(self, x, on_boundary):
              return True if on_boundary else False 
                                                                                

    no_slip = No_slip()


    # MARK SUBDOMAINS (Create mesh functions over the cell facets)
    sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    sub_domains.set_all(5)
    no_slip.mark(sub_domains, 0)


    plot(sub_domains, interactive=False)        # DO NOT USE WITH RAVEN
    #quit()

    #Define Boundary Parts
    boundary_parts = FacetFunction("size_t", mesh)
    no_slip.mark(boundary_parts,0)
    ds = Measure("ds")[boundary_parts]

    # Define function spaces (P2-P1)

    # Discretization  parameters
    family = "CG"; dfamily = "DG"; rich = "Bubble"
    shape = "triangle"; order = 2

    V_s = VectorElement(family, mesh.ufl_cell(), order)       # Elements
    V_d = VectorElement(dfamily, mesh.ufl_cell(), order-1) 
    Z_c = VectorElement(family, mesh.ufl_cell(),  order, 3)
    Z_s = VectorElement(dfamily, mesh.ufl_cell(),  order-1, 3)
    Z_se = VectorElement(rich, mesh.ufl_cell(),  order+1, 3)
    Z_d = VectorElement(dfamily, mesh.ufl_cell(),  order-2, 3)
    Q_s = FiniteElement(family, mesh.ufl_cell(), order-1)
    Q_p = FiniteElement(rich, mesh.ufl_cell(), order+1, 3)
    Z_e = EnrichedElement(Z_c,Z_se)                 # Enriched Elements
    Q_rich = EnrichedElement(Q_s,Q_p)


    W = FunctionSpace(mesh,V_s*Z_s)             # F.E. Spaces 
    V = FunctionSpace(mesh,V_s)
    Vd = FunctionSpace(mesh,V_d)
    Z = FunctionSpace(mesh,Z_s)
    Ze = FunctionSpace(mesh,Z_e)
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


    (v, R_vec) = TestFunctions(W)
    (u, D_vec) = TrialFunctions(W)

    tau_vec = TrialFunction(Z)
    Rt_vec = TestFunction(Z)


    tau0_vec=Function(Z)     # Stress Field (Vector) t=t^n
    tau12_vec=Function(Z)    # Stress Field (Vector) t=t^n+1/2
    tau1_vec=Function(Z)     # Stress Field (Vector) t=t^n+1

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

    def sigma(u, p, Tau):
        return 2*betav*Dcomp(u) - p*Identity(len(u)) + Tau

    def Fdef(u, Tau):
        return dot(u,grad(Tau)) - dot(grad(u),Tau) - dot(Tau,tgrad(u)) + div(u)*Tau 

    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    """Returns the trapezium rule integration of an array with coodinates (i*dt, list_i)"""
    def trapezium(array, dt):
        summation = 0
        for i in range(len(array)-1):
            summation = summation + 0.5*(array[i]+array[i+1])*dt
        return summation
        

    # The  projected  rate -of-strain
    D_proj_vec = Function(Z)
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

    # Parameters

    dt = mesh.hmin()**2 #Time Stepping  
    T_f = 2.0
    Tf = T_f
    tol = 10E-6
    defpar = 1.0

    conv = 1                                      # Non-inertial Flow Parameter (Re=0)
    We = 0.25
    Re = 0.5
    Ma = 0.05
    betav = 0.75

    alph1 = 0.0
    alph2 = 10E-20
    alph3 = 10E-20
    th = 0.1                # DEVSS

    c1 = 0.1                # SUPG / SU
    c2 = 0.05               # Artificial Diffusion
    c3 = 0.05

    Rey = Re



    # Define boundary FUNCTIONS
    ft = exp(-t)
    fdt = -exp(-t)
    u_exact = Expression(('exp(-t)*x[0]','-exp(-t)*x[1]'), degree=2, pi=pi, Re=Re, t=t, ft=ft) # Velocity
    tau_vec_exact = Expression(('exp(-t)*x[0]',
                                'exp(-t)*(x[0] + x[1])',
                                'exp(-t)*x[1]'), degree=2, pi=pi, betav=betav, ft=ft, fdt=fdt, t=t)
    p_exact = Expression('0.0', degree=2, t=t) # Pressure



    # Force Terms

    """F_u = Expression(('Re*x[0]*(fdt + ft*ft) - 2*ft',
                  '-Re*x[1]*(fdt + ft*ft) - 2*ft'), degree=2, pi=pi, Re=Re, ft=ft, fdt=fdt, t=t)
    F_tau_vec = Expression(('x[0]*(ft + We*(fdt - ft*ft)) - 2*(1-betav)*ft',
                        'ft*(x[0]+x[1]) + We*(fdt*(x[0]+x[1]) - ft*ft*(x[0]-x[1]))',
                        '-x[1]*(ft + We*(fdt + ft*ft)) + 2*(1-betav)*ft'), degree=2, pi=pi, Re=Re, ft=ft, fdt=fdt, t=t, We=We, betav=betav)"""


    F_u = Expression(('Re*x[0]*(-exp(-t) + exp(-t)*exp(-t)) - 2*exp(-t)',
                      '-Re*x[1]*(-exp(-t) + exp(-t)*exp(-t)) - 2*exp(-t)'), degree=2, pi=pi, Re=Re, ft=ft, fdt=fdt, t=t)
    F_tau_vec = Expression(('x[0]*(exp(-t) + We*(-exp(-t) - exp(-t)*exp(-t))) - 2*(1-betav)*exp(-t)',
                            'exp(-t)*(x[0]+x[1]) + We*(-exp(-t)*(x[0]+x[1]) - exp(-t)*exp(-t)*(x[0]-x[1]))',
                            '-x[1]*(exp(-t) + We*(-exp(-t) + exp(-t)*exp(-t))) + 2*(1-betav)*exp(-t)'), degree=2, pi=pi, Re=Re, ft=ft, fdt=fdt, t=t, We=We, betav=betav)
    F_tau_vecl2 = interpolate(F_tau_vec, Z)
    F_tau = as_matrix([[F_tau_vec[0], F_tau_vec[1]],
                       [F_tau_vec[1], F_tau_vec[2]]])  

    #quit()

    # Interpolate Stabilisation Functions
    h = CellSize(mesh)
    h_k = project(h/mesh.hmax(), Qt)
    n = FacetNormal(mesh)

    u_l2 = interpolate(u_exact, V)
    tau_vec_l2 = interpolate(tau_vec_exact, Z)
    p_l2 = interpolate(p_exact, Q)

    # Initial conditions
    u0 = interpolate(u0,V)
    u0.vector()[:] = u_l2.vector().array()
    tau0_vec.vector()[:] = tau_vec_l2.vector().array() 
    p0.vector()[:] = p_l2.vector().array()


    tau0 = as_matrix([[tau0_vec[0], tau0_vec[1]],
                      [tau0_vec[1], tau0_vec[2]]])        # Stress 


    # Dirichlet Boundary Conditions  (Test Problem)
    u_bound  = DirichletBC(W.sub(0), u_exact, no_slip)  # Dirichlet Boundary conditions for Velocity 
    tau_bound  =  DirichletBC(Z, tau_vec_exact, no_slip)  # Dirichlet Boundary conditions for Stress
    p_bound = DirichletBC(Q, p_exact, no_slip) # Dirichlet Boundary conditions for Pressure
    bcu = [u_bound]
    bcp = [p_exact]
    bctau = [tau_bound]





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
    x_axis=list()
    y_axis=list()
    u_xg = list()
    u_yg = list()
    tau_xxg = list()
    tau_xyg = list()
    tau_yyg = list()





    print '############# TIME SCALE ############'
    print 'Timestep size (s):', dt
    print 'Finish Time (s):', Tf

    print '############# Scalings & Nondimensional Parameters ############'
    print 'Characteristic Length (m):', L
    print 'Reynolds Number:', Rey
    print 'Weissenberg Number:', We
    print 'Viscosity Ratio:', betav


    Np= len(p0.vector().array())
    Nv= len(w0.vector().array())   
    Ntau= len(tau0_vec.vector().array())
    dof= 3*Nv+2*Ntau+Np
    print '############# Discrete Space Characteristics ############'
    print 'Degree of Elements', order
    print 'Mesh: %s x %s' %(mm, mm)
    print('Size of Pressure Space = %d ' % Np)
    print('Size of Velocity Space = %d ' % Nv)
    print('Size of Stress Space = %d ' % Ntau)
    print('Degrees of Freedom = %d ' % dof)
    print 'Number of Cells:', mesh.num_cells()
    print 'Number of Vertices:', mesh.num_vertices()
    print 'Minimum Cell Diamter:', mesh.hmin()
    print 'Maximum Cell Diamter:', mesh.hmax()
    print '############# Stabilisation Parameters ############'
    print 'DEVSS Parameter:', th




    #Define Variable Parameters, Strain Rate and other tensors
    sr = (grad(u) + transpose(grad(u)))
    srg = grad(u)
    gamdots = inner(Dincomp(u1),grad(u1))
    gamdotp = inner(tau1,grad(u1))

 
    # STABILISATION TERMS
    F1 = dot(u1,grad(tau)) - dot(grad(u1),tau) - dot(tau,tgrad(u1))                             # Convection/Deformation Terms
    F12 = dot(u12,grad(tau)) - dot(grad(u12),tau) - dot(tau,tgrad(u12))                         # Convection/Deformation Terms
 


    # DEVSS Stabilisation
    
    DEVSSl_u12 = 2*(1-betav)*inner(Dincomp(u),Dincomp(v))*dx    
    DEVSSr_u12 = 2*(1-betav)*inner(D0,Dincomp(v))*dx   
    DEVSSl_u1 = 2*(1-betav)*inner(Dincomp(u),Dincomp(v))*dx    
    DEVSSr_u1 = 2*(1-betav)*inner(D12,Dincomp(v))*dx   



    # Use amg preconditioner if available
    prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

    # Use nonzero guesses - essential for CG with non-symmetric BC
    parameters['krylov_solver']['nonzero_initial_guess'] = True

    #Folder To Save Plots for Paraview
    #fv=File("Velocity Results Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"c0="+str(c0)+"/velocity "+str(t)+".pvd")
 

    #Lists for Energy Values
    x=list()
    ee=list()
    ek=list()
    u_err_L2_list=list()
    u_err_H1_list=list()
    tau_err_L2_list=list()
    tau_err_H1_list=list()
    p_err_L2_list=list()
    p_err_H1_list=list()

    #ftau=File("Incompressible Viscoelastic Flow Results/Paraview/Stress_th"+str(th)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"/stress "+str(t)+".pvd")
    #fv=File("Incompressible Viscoelastic Flow Results/Paraview/Velocity_th"+str(th)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"/velocity "+str(t)+".pvd")

    # Time-stepping
    tau_step = project(tau1_vec - tau0_vec, Z)
    stop_crit = norm(tau_step.vector(), 'l2')

while j < loopend:
    j+=1

    t = 0.0



    # Continuation in Reynolds/Weissenberg Number Number (Re-->10Re)
    Ret=Expression('Re*(1.0+0.5*(1.0+tanh(0.7*t-4.0))*19.0)', t=0.0, Re=Re, degree=2)
    Rey=Re
    Wet=Expression('(0.1+(We-0.1)*0.5*(1.0+tanh(500*(t-2.5))))', t=0.0, We=We, degree=2)


    print '############# TIME SCALE ############'
    print 'Timestep size (s):', dt
    print 'Finish Time (s):', Tf

    print '############# Scalings & Nondimensional Parameters ############'
    print 'Characteristic Length (m):', L
    print 'Characteristic Velocity (m/s):', U
    print 'Lid velocity:', (U*0.5*(1.0+tanh(e*t-3.0)),0)
    print 'Speed of sound (m/s):', U/Ma
    print 'Mach Number', Ma
    print 'Reynolds Number:', Rey
    print 'Non-inertial parameter:', conv
    print 'Weissenberg Number:', We
    print 'Viscosity Ratio:', betav
    print 'Diffusion Number:' ,Di
    print 'Viscous Heating Number:', Vh

    Np= len(p0.vector().array())
    Nv= len(w0.vector().array())   
    Ntau= len(tau0_vec.vector().array())
    dof= 3*Nv+2*Ntau+Np
    print '############# Discrete Space Characteristics ############'
    print 'Degree of Elements', order
    print 'Mesh: %s x %s' %(mm, mm)
    print('Size of Pressure Space = %d ' % Np)
    print('Size of Velocity/DEVSS Space = %d ' % Nv)
    print('Size of Stress Space = %d ' % Ntau)
    print('Degrees of Freedom = %d ' % dof)
    print 'Number of Cells:', mesh.num_cells()
    print 'Number of Vertices:', mesh.num_vertices()
    print 'Minimum Cell Diamter:', mesh.hmin()
    print 'Maximum Cell Diamter:', mesh.hmax()
    print '############# Stabilisation Parameters ############'
    print 'DEVSS Momentum Term:', th

    #quit()

    # Initial Density Field
    rho_array = rho0.vector().array()
    for i in range(len(rho_array)):  
        rho_array[i] = 1.0
    rho0.vector()[:] = rho_array 

    # Initial Temperature Field
    T_array = T0.vector().array()
    for i in range(len(T_array)):  
        T_array[i] = T_0
    T0.vector()[:] = T_array

    # Identity Tensor   
    I = Expression((('1.0','0.0'),
                    ('0.0','1.0')), degree=2)


    #Define Variable Parameters, Strain Rate and other tensors
    gamdots = inner(Dincomp(u1),grad(u1))
    gamdots12 = inner(Dincomp(u12),grad(u12))
    gamdotp = inner(tau1,grad(u1))
    gamdotp12 = inner(tau12,grad(u12))
    thetal = (T)/(T_h-T_0)
    thetar = (T_0)/(T_h-T_0)
    thetar = project(thetar,Qt)
    theta0 = (T0-T_0)/(T_h-T_0)
    #alpha = 1.0/(rho*Cv)



    # STABILISATION TERMS
    F1 = dot(u1,grad(tau)) - dot(grad(u1),tau) - dot(tau,tgrad(u1) + div(u1)*tau)                   # Convection/Deformation Terms
    F12 = dot(u12,grad(tau)) - dot(grad(u12),tau) - dot(tau,tgrad(u12)) + div(u12)*tau               # Convection/Deformation Terms
    
    velocity = as_vector([1.0, 1.0])
    h = CellSize(mesh)
    unorm = sqrt(dot(velocity,velocity))
    c1 = alph1*(h/(2.0*unorm))
    
    # SU Stabilisation
    SUl3 = inner(c1*dot(u0 , grad(Rt)), dot(u12, grad(tau)))*dx
    SUl4 = inner(c1*dot(u12 , grad(Rt)), dot(u1, grad(tau)))*dx

    # SUPG Stabilisation

    SUPGl3 = inner(tau+We*F12,c1*dot(u12,grad(Rt)))*dx
    SUPGr3 = inner(Dincomp(u12),c1*dot(u12,grad(Rt)))*dx    
    SUPGl4 = inner(We*F1,c1*dot(u1,grad(Rt)))*dx
    SUPGr4 = inner(2*(1-betav)*Dincomp(u1),c1*dot(u1,grad(Rt)))*dx 

    # DEVSS Stabilisation
    devss_Z = inner(D-Dcomp(u),R)*dx                        # L^2 Projection of rate-of strain
    
    DEVSSl_u12 = 2*(1-betav)*inner(Dcomp(u),Dincomp(v))*dx    
    DEVSSr_u12 = 2*inner(D0,Dincomp(v))*dx   
    DEVSSl_u1 = 2*(1-betav)*inner(Dcomp(u),Dincomp(v))*dx    
    DEVSSr_u1 = 2*inner(D12,Dincomp(v))*dx 

    #DEVSSl_temp1 = (1-Di)*inner(grad(theta),grad(r))
    #DEVSSr_temp1 = (1-Di)*inner(grad(theta),grad(r))


    # Use amg preconditioner if available
    prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

    # Use nonzero guesses - essential for CG with non-symmetric BC
    parameters['krylov_solver']['nonzero_initial_guess'] = True

    #Folder To Save Plots for Paraview
    #fv=File("Velocity Results Re="+str(Rey)+"We="+str(We)+"b="+str(betav)+"theta"+str(theta)+"c0="+str(c0)+"/velocity "+str(t)+".pvd")
 
    #Lists for Energy Values
    x=list()
    ee=list()
    ek=list()
    z=list()

    conerr=list()
    deferr=list()
    tauerr=list()


    # Time-stepping
    t = dt
    iter = 0            # iteration counter
    maxiter = 100000000
    while t < Tf + DOLFIN_EPS and iter < maxiter:
        iter += 1
        print"t = %s,  Iteration = %d, Convergence Failures = %s, Loop = %s" %(t, iter, jj, j)

        (u0, D0_vec)=w0.split()   
        # Update Stabilisation (Copy and Paste Stabilisation Technique from above)
        F0R = dot(u0,grad(tau0)) - dot(grad(u0),tau0) - dot(tau0,tgrad(u0)) + div(u0)*tau0
        F0R_vec = as_vector([F0R[0,0], F0R[1,0], F0R[1,1]])
        res0 = ((1.0 + We/dt)*tau0_vec + We*F0R_vec)-2*(1-betav)*D1_vec 
        res_test = project(res0,Z)
        res_orth = project(res0-res_test,Ze)                                # Project the residual!!!!!!!!!!!!
        tau_stab = as_matrix([[res_orth[0]*tau_vec[0], res_orth[1]*tau_vec[1]],
                              [res_orth[1]*tau_vec[1], res_orth[2]*tau_vec[2]]])
        tau_stab1 = as_matrix([[res_orth[0]*tau1_vec[0], res_orth[1]*tau1_vec[1]],
                              [res_orth[1]*tau1_vec[1], res_orth[2]*tau1_vec[2]]])
        Rt_stab = as_matrix([[res_orth[0]*Rt_vec[0], res_orth[1]*Rt_vec[1]],
                              [res_orth[1]*Rt_vec[1], res_orth[2]*Rt_vec[2]]]) 

        res_orth_norm_sq = project(inner(res_orth,res_orth),Qt)
        res_orth_norm = np.power(res_orth_norm_sq,0.5)
        kapp = project(res_orth_norm, Qt)
        #if iter > 2:
        #   kapp = normalize_solution(kapp)
        LPSl_stress = alph1*(inner(kapp*h*0.01*div(tau),div(Rt))*dx + inner(kapp*h*0.75*grad(tau),grad(Rt))*dx)

       
        DEVSSr_u12 = 2*(1-betav)*inner(D0,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS


     

        # Velocity Half Step

        lhs_u12 = (Re/(dt/2.0))*rho0*u
        rhs_u12 = (Re/(dt/2.0))*rho0*u0 - Re*rho0*conv*grad(u0)*u0
        visc_12 = betav*(inner(grad(u),grad(v))*dx + (1.0/3)*inner(div(u),div(v))*dx)

        a1=inner(lhs_u12,v)*dx + visc_12 + (inner(D-Dcomp(u),R)*dx)
        L1=inner(rhs_u12,v)*dx + inner(p0,div(v))*dx - inner(tau0,grad(v))*dx 

            #DEVSS Stabilisation
        a1+= th*DEVSSl_u12                     
        L1+= th*DEVSSr_u12 


        A1 = assemble(a1)
        b1= assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]
        solve(A1, w12.vector(), b1, "bicgstab", "default")
        end()
        
        (u12, D12_vec)=w12.split()
        D12 = as_matrix([[D12_vec[0], D12_vec[1]],
                        [D12_vec[1], D12_vec[2]]])

        DEVSSr_u1 = 2*(1-betav)*inner(D12,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS

        """# STRESS Half Step
        F12 = dot(u12,grad(tau)) - dot(grad(u12),tau) - dot(tau,tgrad(u12) + div(u12)*tau) # Convection/Deformation Terms
        lhs_tau12 = (We/dt+1.0)*tau + We*F12                             # Left Hand Side
        rhs_tau12= (We/dt)*tau0 + 2.0*(1.0-betav)*Dcomp(u0)                     # Right Hand Side

        a3 = inner(lhs_tau12,Rt)*dx                                 # Weak Form
        L3 = inner(rhs_tau12,Rt)*dx

        a3 += SUPGl3             # SUPG Stabilisation LHS
        L3 += SUPGr3             # SUPG / SU Stabilisation RHS
        A3=assemble(a3)
        b3=assemble(L3)
        [bc.apply(A3, b3) for bc in bctau]
        solve(A3, tau12_vec.vector(), b3, "bicgstab", "default")
        end()"""

        #Temperature Half Step
        #A8 = assemble(a8)
        #b8 = assemble(L8)
        #[bc.apply(A8, b8) for bc in bcT]
        #solve(A8, T12.vector(), b8, "bicgstab", "default")
        #end()
        
        #Compute Predicted U* Equation

        lhs_us = (Re/dt)*rho0*u
        rhs_us = (Re/dt)*rho0*u0 - Re*conv*rho0*grad(u12)*u12
        rhs_stress = -0.5*betav*grad(u0) - tau0
        rhs_press = p0 - 0.5*(betav/3)*div(u0)
         

        a2=inner(lhs_us,v)*dx + (inner(D-Dcomp(u),R)*dx)
        L2=inner(rhs_us,v)*dx + inner(rhs_stress,grad(v))*dx + inner(rhs_press,div(v))*dx


        A2 = assemble(a2)
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcu]
        solve(A2, ws.vector(), b2, "bicgstab", "default")
        end()

        (us, Ds_vec) = ws.split()



        #Continuity Equation 1
        lhs_p_1 = (Ma*Ma/(dt*Re))*p
        rhs_p_1 = (Ma*Ma/(dt*Re))*p0 - rho0*div(us) + dot(grad(rho0),us)

        lhs_p_2 = 0.5*dt*grad(p)
        rhs_p_2 = 0.5*dt*grad(p0)
        
        a5=inner(lhs_p_1,q)*dx + inner(lhs_p_2,grad(q))*dx   
        L5=inner(rhs_p_1,q)*dx + inner(rhs_p_2,grad(q))*dx

        A5 = assemble(a5)
        b5 = assemble(L5)
        [bc.apply(A5, b5) for bc in bcp]
        #[bc.apply(p1.vector()) for bc in bcp]
        solve(A5, p1.vector(), b5, "bicgstab", "default")
        end()


        #Continuity Equation 2
        rho1 = rho0 + (Ma*Ma/Re)*(p1-p0)
        rho1 = project(rho1,Q)


        #Velocity Update
        visc_1 = 0.5*betav*(inner(grad(u),grad(v))*dx + 1.0/3*inner(div(u),div(v))*dx)

        lhs_u1 = (Re/dt)*rho0*u                                          # Left Hand Side
        rhs_u1 = (Re/dt)*rho0*us                                         # Right Hand Side

        a7=inner(lhs_u1,v)*dx + visc_1 + inner(D-Dcomp(u),R)*dx  # Weak Form
        L7=inner(rhs_u1,v)*dx + 0.5*inner(p1-p0,div(v))*dx 

        a1+= th*DEVSSl_u1                                                #DEVSS Stabilisation
        L1+= th*DEVSSr_u1 

        A7 = assemble(a7)
        b7 = assemble(L7)
        [bc.apply(A7, b7) for bc in bcu]
        solve(A7, w1.vector(), b7, "bicgstab", "default")
        end()

        (u1, D1_vec) = w1.split()
        D1 = as_matrix([[D1_vec[0], D1_vec[1]],
                        [D1_vec[1], D1_vec[2]]])



        # Stress Full Step
        F1 = dot(u1,grad(tau)) - dot(grad(u1),tau) - dot(tau,tgrad(u1)) + div(u1)*tau # Convection/Deformation Terms
        lhs_tau1 = (We/dt+1.0)*tau + We*F1                             # Left Hand Side
        rhs_tau1= (We/dt)*tau0 + 2.0*(1.0-betav)*D12          # Right Hand Side

        a4 = inner(lhs_tau1,Rt)*dx
        L4 = inner(rhs_tau1,Rt)*dx

        a4 += LPSl_stress   # [SUPGl4, SUl4, LPSl_stab, LPSl_stress, diff_stab, 0]
        L4 += 0   # [SUPGr4, SUr4, LPSr_stab, LPS_res_stab, 0]   
  
        A4=assemble(a4)                                     # Assemble System
        b4=assemble(L4)
        [bc.apply(A4, b4) for bc in bctau]
        solve(A4, tau1_vec.vector(), b4, "bicgstab", "default")
        end()

        #Temperature Full Step
        #lhs_temp1 = (1.0/dt)*rho1*thetal + rho1*dot(u1,grad(thetal))
        #difflhs_temp1 = Di*grad(thetal)
        #rhs_temp1 = (1.0/dt)*rho1*thetar + rho1*dot(u1,grad(thetar)) + (1.0/dt)*rho1*theta0 + Vh*(gamdots12 + gamdotp12 - p1*div(u1))
        #diffrhs_temp1 = Di*grad(thetar)
        #a9 = inner(lhs_temp1,r)*dx + inner(difflhs_temp1,grad(r))*dx 
        #L9 = inner(rhs_temp1,r)*dx + inner(diffrhs_temp1,grad(r))*dx - Di*Bi*inner(theta0,r)*ds(1) \

        #a9+= th*DEVSSl_T1                                                #DEVSS Stabilisation
        #L9+= th*DEVSSr_T1 

        #A9 = assemble(a9)
        #b9 = assemble(L9)
        #[bc.apply(A9, b9) for bc in bcT]
        #solve(A9, T1.vector(), b9, "bicgstab", "default")
        #end()


        # Energy Calculations
        E_k=assemble(0.5*rho1*dot(u1,u1)*dx)
        E_e=assemble((tau1_vec[0]+tau1_vec[2])*dx)



        # Calculate Size of Artificial Term
        #o= tau1.vector()-tau0.vector()                         # Stress Difference per timestep
        #h= p1.vector()-p0.vector()
        #m=u1.vector()-u0.vector()                              # Velocity Difference per timestep
        #l=T1.vector()-T0.vector()



        # Record Error Data 
        
        #x.append(t)
        #y.append(norm(h,'linf')/norm(p1.vector()))
        #z.append(norm(o,'linf')/(norm(tau1.vector())+0.00000000001))
        #zz.append(norm(m,'linf')/norm(u1.vector()))
        #zzz.append(norm(l,'linf')/(norm(u1.vector())+0.0001))

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

        # Record Error Data
        err = project(h*kapp,Qt)
        x.append(t)
        ee.append(norm(err.vector(),'linf'))
        ek.append(norm(tau1_vec.vector(),'linf'))
        

        # Save Plot to Paraview Folder 
        #for i in range(5000):
            #if iter== (0.02/dt)*i:
               #fv << u1
        #ft << T1

        # Break Loop if code is diverging

        if max(norm(tau1_vec.vector(), 'linf'),norm(w1.vector(), 'linf')) > 10E6 or np.isnan(sum(tau1_vec.vector().array())):
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
            dt=dt/2                        # Use Smaller timestep 
            j-=1                            # Extend loop
            jj+= 1                          # Convergence Failures
            Tf= (iter-40)*dt
            # Reset Functions
            rho0 = Function(Q)
            rho1 = Function(Q)
            p0=Function(Q)       # Pressure Field t=t^n
            p1=Function(Q)       # Pressure Field t=t^n+1
            T0=Function(Qt)       # Temperature Field t=t^n
            T1=Function(Qt)       # Temperature Field t=t^n+1
            tau0_vec=Function(Z)     # Stress Field (Vector) t=t^n
            tau12_vec=Function(Z)    # Stress Field (Vector) t=t^n+1/2
            tau1_vec=Function(Z)     # Stress Field (Vector) t=t^n+1
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
        #if t>0.5:
            #plot(kapp, title="tau_xy Stress", rescale=True, interactive=False)
            #plot(tau1[0,0], title="tau_xx Stress", rescale=True, interactive=False)
            #plot(p1, title="Pressure", rescale=True)
            #plot(rho1, title="Density", rescale=True)
            #plot(u1, title="Velocity", rescale=True, mode = "auto")
            #plot(T1, title="Temperature", rescale=True)
           

        # Move to next time step (Continuation in Reynolds Number)
        w0.assign(w1)
        T0.assign(T1)
        rho0.assign(rho1)
        p0.assign(p1)
        tau0_vec.assign(tau1_vec)
        t += dt



    # PLOTS
    # Plot Error Control Data
    plt.figure(0)
    plt.plot(x, ee, 'r-', label=r'$\kappa$')
    plt.plot(x, ek, 'b-', label=r'$||\tau||$')
    plt.legend(loc='best')
    plt.xlabel('time(s)')
    plt.ylabel('$||\cdot||_{\infty}$')
    plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/Error_controlRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
    plt.clf()
    plt.close()




    
    if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==5 or jjj==4:

        # Plot First Normal Stress Difference
        tau_xx=project(tau1[0,0],Q)
        mplot(tau_xx)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/tau_xxRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf() 
        tau_xy=project(tau1[1,0],Q)
        mplot(tau_xy)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/tau_xyRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf() 
        tau_yy=project(tau1[1,1],Q)
        mplot(tau_yy)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/tau_yyRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf() 
        divu = project(div(u1),Q)
        mplot(divu)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/div_uRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()

    if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==5 or jjj==4:
 
       # Plot Velocity Components
        ux=project(u1[0],Q)
        mplot(ux)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/u_xRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()
        uy=project(u1[1],Q)
        mplot(uy)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/u_yRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()

    if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==5:


        # Matlab Plot of the Solution at t=Tf
        rho1=rho_0*rho1
        rho1=project(rho1,Q)
        #p1=mu_0*(L/U)*p1  #Dimensionalised Pressure
        #p1=project(p1,Q)
        mplot(rho1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/DensityRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf() 
        mplot(p1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/PressureRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()
        mplot(T1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/TemperatureRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()



    if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==5:
        #Plot Contours USING MATPLOTLIB
        # Scalar Function code


        x = Expression('x[0]', degree=2)     #GET X-COORDINATES LIST
        y = Expression('x[1]', degree=2)     #GET Y-COORDINATES LIST
        pvals = p1.vector().array()          # GET SOLUTION p= p(x,y) list
        Tvals = T1.vector().array()          # GET SOLUTION T= T(x,y) list
        rhovals = rho1.vector().array()      # GET SOLUTION p= p(x,y) list
        tauxx = project(tau1_vec[0], Q)
        tauxxvals = tauxx.vector().array()
        xyvals = mesh.coordinates()     # CLEAN THIS UP!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        xvalsq = interpolate(x, Q)#xyvals[:,0]
        yvalsq= interpolate(y, Q)#xyvals[:,1]
        xvalsw = interpolate(x, Qt)#xyvals[:,0]
        yvalsw= interpolate(y, Qt)#xyvals[:,1]

        xvals = xvalsq.vector().array()
        yvals = yvalsq.vector().array()


        xx = np.linspace(x_0,x_1)
        yy = np.linspace(y_0,y_1)
        XX, YY = np.meshgrid(xx,yy)   # (x,y) coordinate data formatted so that it can be used by plt.contour()
        pp = mlab.griddata(xvals, yvals, pvals, xx, yy, interp='nn') # u(x,y) data so that it can be used by 
        dd = mlab.griddata(xvals, yvals, rhovals, xx, yy, interp='nn') # u(x,y) data so that it can be used by 

        plt.contour(XX, YY, dd, 25)
        plt.title('Density Contours')   # DENSITY CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/DensityContoursRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()

        plt.contour(XX, YY, pp, 25)
        plt.title('Pressure Contours')   # PRESSURE CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/PressureContoursRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()

        """TT = mlab.griddata(xvals, yvals, Tvals, xx, yy, interp='nn') 
        plt.contour(XX, YY, TT, 20) 
        plt.title('Temperature Contours')   # TEMPERATURE CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/TemperatureContoursRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()"""


        normstress = mlab.griddata(xvals, yvals, tauxxvals, xx, yy, interp='nn')

        """plt.contour(XX, YY, normstress, 20) 
        plt.title('Stress Contours')   # NORMAL STRESS CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/StressContoursRe="+str(Re)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()"""


        #Plot Velocity Streamlines USING MATPLOTLIB
        u1_q = project(u1[0],Q)
        uvals = u1_q.vector().array()
        v1_q = project(u1[1],Q)
        vvals = v1_q.vector().array()

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
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/VelocityContoursRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")   
        plt.clf()                                             # display the plot


    plt.close()


    if dt < tol:
       j=loopend+1
       break

    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E5 and abs(E_k) < 10:
        Tf=T_f   

    if j==5:
        jjj+=1
        if jjj==1:
            Ma = 0.01
        if jjj==2:
            Ma = 0.1        
        #Re = Re/2
        j=0
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


    if jjj==4:
        quit()


    # Reset Functions
    rho0 = Function(Q)
    rho1 = Function(Q)
    p0 = Function(Q)       # Pressure Field t=t^n
    p1 = Function(Q)       # Pressure Field t=t^n+1
    T0 = Function(Qt)       # Temperature Field t=t^n
    T1 = Function(Qt)       # Temperature Field t=t^n+1
    tau0_vec = Function(Ze)     # Stress Field (Vector) t=t^n
    tau12_vec = Function(Ze)    # Stress Field (Vector) t=t^n+1/2
    tau1_vec = Function(Ze)     # Stress Field (Vector) t=t^n+1
    w0 = Function(W)
    w12 = Function(W)
    ws = Function(W)
    w1 = Function(W)
    (u0, D0_vec) = w0.split()
    (u12, D12_vec) = w0.split()
    (us, Ds_vec) = w0.split()
    (u1, D1_vec) = w0.split()
    D_proj_vec = Function(Ze)
    D_proj = as_matrix([[D_proj_vec[0], D_proj_vec[1]],
                        [D_proj_vec[1], D_proj_vec[2]]])
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





