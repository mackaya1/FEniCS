"""Compressible Lid Driven Cavity Problem for an COMPRESSIBLE Oldroyd-B Fluid"""
"""Test for Convergence of Numerical Scheme"""


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
        mm = 20
    if j==2:
        mm = 40
    if j==3:
        mm = 80
    if j==4:
        mm = 100


     
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

    class Inflow(SubDomain):
          def inside(self, x, on_boundary):
              return True if x[0] < DOLFIN_EPS and on_boundary else False    


    class Walls(SubDomain):
          def inside(self, x, on_boundary):
              return True if x[1] < DOLFIN_EPS or x[1] > L - DOLFIN_EPS and on_boundary  else False 

    
    class Outflow(SubDomain):
          def inside(self, x, on_boundary):
              return True if x[0] > L - 10*DOLFIN_EPS and on_boundary else False                                                                                

    inflow = Inflow()
    walls = Walls()
    outflow = Outflow()

    # MARK SUBDOMAINS (Create mesh functions over the cell facets)
    sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    sub_domains.set_all(5)
    walls.mark(sub_domains, 0)
    outflow.mark(sub_domains, 3)
    inflow.mark(sub_domains, 2)

    plot(sub_domains, interactive=False)        # DO NOT USE WITH RAVEN
    #quit()

    #Define Boundary Parts
    boundary_parts = FacetFunction("size_t", mesh)
    walls.mark(boundary_parts,0)
    ds = Measure("ds")[boundary_parts]

    # Define function spaces (P2-P1)

    # Discretization  parameters
    family = "CG"; dfamily = "DG"; rich = "Bubble"
    shape = "triangle"; order = 2

    #mesh.ufl_cell()
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

    U12 = 0.5*(u1 + u0)


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
        return ((grad(w) + tgrad(w))-(2.0/3)*div(w)*Identity(len(u)))/2

    def sigma(u, p, Tau):
        return 2*betav*Dincomp(u) - p*Identity(len(u)) + Tau

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
    D_proj_vec = Function(Zd)
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

    dt = mesh.hmin()/25 #Time Stepping  
    T_f = 3.0
    Tf = T_f
    tol = 10E-6
    defpar = 1.0

    conv = 1.0                                      # Non-inertial Flow Parameter (Re=0)
    We = 0.25
    Re = 0.5
    c0 = 1000
    Ma = 0.001
    betav = 0.75

    alph1 = 0.0
    alph2 = 10E-20
    alph3 = 10E-20
    th = 1.0              # DEVSS

    c1 = 0.1            # SUPG / SU
    c2 = 0.005                  # Artificial Diffusion
    c3 = 0.005

    Rey=Re



    # Define boundary FUNCTIONS
    epp = 0.1
    #(1+epp*sin(t))* #Time Dependence
    u_exact = Expression(('4.0*x[1]*(1.0-x[1])','0'), degree=2) # Velocity
    tau_vec_exact = Expression(('8*(1.0-betav)*We*(1.0-2.0*x[1])*(1.0-2.0*x[1])',
                                '4*(1.0-betav)*(1.0-2.0*x[1])',
                                '0.0'), degree=2, t=0.0, betav=betav, We=We)
    p_exact = Expression('8*(1.0-x[0])', degree=2, t=0.0, pi=pi, epp=epp) # Pressure



    """# Force Terms (TIME DEPENDENT PROBLEM)
    F_u = Expression(('epp*cos(t)*x[1]*(1-x[1])','0.0'), degree=2, pi=pi, Re=Re, t=t, epp=epp)
    F_tau_vec = Expression(('0.0',
                        '0.0',
                        '0.0'), degree=2, pi=pi, Re=Re, t=t, epp=epp, We=We, betav=betav)
    F_tau = as_matrix([[F_tau_vec[0], F_tau_vec[1]],
                      [F_tau_vec[1], F_tau_vec[2]]])"""   

    # Interpolate Stabilisation Functions
    h = CellSize(mesh)
    h_k = project(h/mesh.hmax(), Qt)


    u_l2 = interpolate(u_exact, V)
    tau_vec_l2 = interpolate(tau_vec_exact, Z)
    p_l2 = interpolate(p_exact, Q)
    #D_exact = interpolate(Dincomp(u_l2), Z)

    # Initial conditions
    u0 = interpolate(u0,V)
    u0.vector()[:] = u_l2.vector().array()
    tau0_vec.vector()[:] = tau_vec_l2.vector().array() 
    p0.vector()[:] = p_l2.vector().array()
    #D0.vector()[:] = D_exact.vector().array()

    tau0 = as_matrix([[tau0_vec[0], tau0_vec[1]],
                      [tau0_vec[1], tau0_vec[2]]])        # Stress 

    U = 0.5*(u + u0)   
    U12 = 0.5*(u1 + u0)   



    # Define boundaries
    inflow  = 'near(x[0], 0)'
    outflow = 'near(x[0], 1)'
    walls   = 'near(x[1], 0) || near(x[1], 1)'


    u_exact = Expression(('4.0*x[1]*(1.0-x[1])','0'), degree=2) # Velocity
    tau_vec_exact = Expression(('8.0*(1.0-betav)*We*(1.0-2.0*x[1])*(1.0-2.0*x[1])',
                                '4.0*(1.0-betav)*(1.0-2.0*x[1])',
                                '0.0'), degree=2, t=0.0, betav=betav, We=We)
    p_exact = Expression('8.0*(1.0-x[0])', degree=2, t=0.0) # Pressure

    # Define boundary conditions
    u_bound1  = DirichletBC(W.sub(0), u_exact, inflow)  # Dirichlet Boundary conditions for Velocity
    u_bound2 =  DirichletBC(W.sub(0), u_exact, outflow) 
    bcu_noslip  = DirichletBC(W.sub(0), Constant((0, 0)), walls)
    bcp_inflow  = DirichletBC(Q, Constant(8), inflow)
    bcp_outflow = DirichletBC(Q, Constant(0), outflow)
    bctau_inflow = DirichletBC(Z, tau_vec_exact, outflow) 
    bcu = [bcu_noslip]
    bcp = [bcp_inflow, bcp_outflow]
    bctau = [bctau_inflow]

    n   = FacetNormal(mesh)

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

    #ftau=File("Incompressible Viscoelastic Flow Results/Paraview/Stress_th"+str(th)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"/stress "+str(t)+".pvd")
    #fv=File("Incompressible Viscoelastic Flow Results/Paraview/Velocity_th"+str(th)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"/velocity "+str(t)+".pvd")

    # Time-stepping
    tau_step = project(tau1_vec - tau0_vec, Z)
    stop_crit = norm(tau_step.vector(), 'l2')

    t = 0.0
    iter = 0            # iteration counter
    maxiter = 50000000000
    while t < Tf + DOLFIN_EPS and iter < maxiter:
        iter += 1
        print"t = %s,  Iteration = %d, Convergence Failures = %s, Loop = %s" %(t, iter, jj, j)

        #tau_step = project(tau1_vec - tau0_vec, Z)
        #stop_crit = norm(tau_step, 'L2')
        #print stop_crit

        #Update Boundary Condidtions and Exact Solution
        #u_exact.t=t
        #tau_vec_exact.t=t
        #p_exact.t=t

        #F_u.t = t
        #F_tau_vec.t=t

        #u_l2 = interpolate(u_exact, V)
        #tau_vec_l2 = interpolate(tau_vec_exact, Z)
        #p_l2 = interpolate(p_exact, Q)
        
        #u_bound  = DirichletBC(W.sub(0), u_exact, inflow)  # Dirichlet Boundary conditions for Velocity 
        #tau_bound  =  DirichletBC(Z, tau_vec_exact, inflow)  # Dirichlet Boundary conditions for Stress
        #p_bound = DirichletBC(Q, p_exact, no_slip) # Dirichlet Boundary conditions for Pressure
        #bcu = [u_bound, no_sl]
        #bcp = [p_bound]
        #bctau = [tau_bound]




        # Update Stabilisation (Copy and Paste Stabilisation Technique from above)
        F1R = dot(u1,grad(tau1)) - dot(grad(u1),tau1) - dot(tau1,tgrad(u1))
        F1R_vec = as_vector([F1R[0,0], F1R[1,0], F1R[1,1]])
        res1 = ((1.0 + We/dt)*tau1_vec + We*F1R_vec) - We/dt*tau0_vec -2*(1-betav)*D12_vec  
        res_test = project(res1,Zd)
        res_orth = project(res1-res_test,Z)                                # Project the residual!!!!!!!!!!!!
        tau_stab = as_matrix([[res_orth[0]*tau_vec[0], res_orth[1]*tau_vec[1]],
                              [res_orth[1]*tau_vec[1], res_orth[2]*tau_vec[2]]])
        Rt_stab = as_matrix([[res_orth[0]*Rt_vec[0], res_orth[1]*Rt_vec[1]],
                              [res_orth[1]*Rt_vec[1], res_orth[2]*Rt_vec[2]]]) 
        res_orth_norm_sq = project(inner(res_orth,res_orth),Qt)
        res_orth_norm = np.power(res_orth_norm_sq,0.5)
        kapp = project(res_orth_norm, Qt)


        #resv = (Re/dt)*rho1*(u1-u0) + rho1*grad(u1)*u1 + 0.5*(grad(p1) + grad(p0)) - div(tau0) - betav*div(grad(u1))        
        #resv_test = project(resv, Vd)
        #resv_orth = project(resv - resv_test, V)
        #resv_orth_norm_sq = project(inner(resv_orth,resv_orth), Qt)     # Project residual norm onto discontinuous space
        #resv_orth_norm = np.power(resv_orth_norm_sq, 0.5)
        #kappv = project(resv_orth_norm, Qt)


        LPSl_stress = inner(kapp*h*c1*grad(tau),grad(Rt))*dx + inner(kapp*h*c2*div(tau),div(Rt))*dx  # Stress Stabilisation
        #LPSl_vel = inner(kappv*c3*h_k*Dincomp(u),Dincomp(v))*dx   




        if iter > 1:
            w0.assign(w1)
            p0.assign(p1)
            tau0_vec.assign(tau1_vec)
            (u0, D0_vec)=w0.split()
   
        DEVSSr_u12 = 2*(1-betav)*inner(D0,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS

        U = 0.5*(u + u0)              
        # VELOCITY HALF STEP
        lhsFu12 = Re*rho0*(2.0*(u - u0) / dt + conv*dot(u0, nabla_grad(u0)))
        Fu12 = dot(lhsFu12, v)*dx + \
               + inner(sigma(U, p0, tau0), Dincomp(v))*dx \
               + dot(p0*n, v)*ds - dot(betav*nabla_grad(U)*n, v)*ds \
               - dot(tau0*n, v)*ds \
               +  inner(D-Dincomp(u),R)*dx
        a1 = lhs(Fu12)
        L1 = rhs(Fu12)

            #DEVSS Stabilisation
        a1+= th*DEVSSl_u12                     
        L1+= th*DEVSSr_u12 

        A1 = assemble(a1)
        b1= assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]
        solve(A1, w12.vector(), b1, "bicgstab", "default")
        end()

        (u12, D12_vec)=w12.split()

        DEVSSr_u1 = 2.0*(1-betav)*inner(D12,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS

        # STRESS Half Step
        F12 = dot(u12,grad(tau)) - dot(grad(u12),tau) - dot(tau,tgrad(u12)) # Convection/Deformation Terms
        lhs_tau12 = (We/dt+1.0)*tau + We*F12                             # Left Hand Side
        rhs_tau12= (We/dt)*tau0 + 2.0*(1.0-betav)*Dincomp(u0) #+ F_tau                    # Right Hand Side

        a3 = inner(lhs_tau12,Rt)*dx                                 # Weak Form
        L3 = inner(rhs_tau12,Rt)*dx

        a3 += LPSl_stress             # SUPG Stabilisation LHS
        L3 += 0             # SUPG / SU Stabilisation RHS
        A3=assemble(a3)
        b3=assemble(L3)
        [bc.apply(A3, b3) for bc in bctau]
        solve(A3, tau12_vec.vector(), b3, "bicgstab", "default")
        end()
        
        #Predicted U* Equation
        Fus = Re*dot((u - u0) / dt, v)*dx + \
              Re*conv*dot(dot(u12, nabla_grad(u12)), v)*dx \
              + inner(sigma(U, p0, tau12), Dincomp(v))*dx \
              + dot(p0*n, v)*ds - dot(betav*nabla_grad(U)*n, v)*ds - dot(tau0*n, v)*ds \
              +  inner(D-Dincomp(u),R)*dx     
              
        a2= lhs(Fus)
        L2= rhs(Fus)

        A2 = assemble(a2)
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcu]
        solve(A2, ws.vector(), b2, "bicgstab", "default")
        end()

        (us, Ds_vec) = ws.split()


        #PRESSURE CORRECTION
        a5=inner(grad(p),grad(q))*dx 
        L5=inner(grad(p0),grad(q))*dx - (Re/dt)*div(us)*q*dx
        A5 = assemble(a5)
        b5 = assemble(L5)
        [bc.apply(A5, b5) for bc in bcp]
        #[bc.apply(p1.vector()) for bc in bcp]
        solve(A5, p1.vector(), b5, "bicgstab", "default")
        end()
        
        #VELOCITY UPDATE
        lhs_u1 = (Re/dt)*u                                          # Left Hand Side
        rhs_u1 = (Re/dt)*us                                         # Right Hand Side

        a7=inner(lhs_u1,v)*dx + inner(D-Dincomp(u),R)*dx  # Weak Form
        L7=inner(rhs_u1,v)*dx + 0.5*inner(p1-p0,div(v))*dx 

            #DEVSS Stabilisation
        a7+= th*DEVSSl_u1                      
        L7+= th*DEVSSr_u1

        A7 = assemble(a7)
        b7 = assemble(L7)
        solve(A7, w1.vector(), b7, "bicgstab", "default")
        end()

        (u1, D1_vec) = w1.split()
        U12 = 0.5*(u1 + u0)

        # Stress Full Step
        F1 = dot(u1,grad(tau)) - dot(grad(u1),tau) - dot(tau,tgrad(u1)) # Convection/Deformation Terms t^{n+1}
        F12 = dot(U12,grad(tau12)) - dot(grad(U12),tau12) - dot(tau12,tgrad(U12)) # Convection/Deformation Terms t^{n+1/2}
        lhs_tau1 = (We/dt+1.0)*tau  +  We*F12                            # Left Hand Side
        rhs_tau1= (We/dt)*tau0 + 2.0*(1.0-betav)*Dincomp(U12) #+  F_tau       # Right Hand Side

        A = inner(lhs_tau1,Rt)*dx - inner(rhs_tau1,Rt)*dx
        a4 = lhs(A)
        L4 = rhs(A) 

            # SUPG / SU / LPS Stabilisation (User Choose One)

        a4 += LPSl_stress  # [SUPGl4, SUl4, LPSl_stab, LPSl_stress, diff_stab, 0]
        L4 += 0  # [SUPGr4, SUr4, LPSr_stab, LPS_res_stab, 0]   


        A4=assemble(a4)                                     # Assemble System
        b4=assemble(L4)
        [bc.apply(A4, b4) for bc in bctau]
        solve(A4, tau1_vec.vector(), b4, "bicgstab", "default")
        end()


     

        # Record Error Data 
        err = project(h*kapp,Qt)            # Relative Size of LPS Error Term
        x.append(t)
        ee.append(norm(err.vector(),'linf'))
        ek.append(norm(tau1_vec.vector(),'linf'))


        # Break Loop if code is diverging

        if norm(w1.vector(), 'linf') > 10E5 or np.isnan(sum(w1.vector().array())):
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
            T0=Function(Q)       # Temperature Field t=t^n
            T1=Function(Q)       # Temperature Field t=t^n+1
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
        #if t>0.0:
            #plot(tau1_vec[0], title="Normal Stress", rescale=True)
            #plot(p1, title="Pressure", rescale=True)
            #plot(u1, title="Velocity", rescale=True, mode = "auto")

        # Move to next time step
        t += dt

    # Plot Error Control Data
    """plt.figure(0)
    plt.plot(x, ee, 'r-', label=r'$\kappa$')
    plt.plot(x, ek, 'b-', label=r'$||\tau||$')
    plt.legend(loc='best')
    plt.xlabel('time(s)')
    plt.ylabel('$||\cdot||_{\infty}$')
    plt.savefig("Incompressible Viscoelastic Flow Results/Error/LPSError_controlRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
    plt.clf()
    plt.close()"""


    # Convergence in Mesh Size
    u1_vec = interpolate(u1,V)
    u_error = u_l2.vector() - u1_vec.vector()
    tau_error = tau_vec_l2.vector() - tau1_vec.vector()
    p_error = p_l2.vector() - p1.vector()
    
    u_errfun = project(u_l2 - u1_vec, V)
    tau_errfun = project(tau_vec_l2 - tau1_vec, Z)
    p_errfun = project(p_l2 - p1, Q)

    u_err_l2 = norm(u_error,'l2')
    u_err_L2 = norm(u_errfun, 'L2')
    u_err_linf = norm(u_error , 'linf')
    u_err_Linf = norm(u_errfun, 'H1')

    tau_err_l2 = norm(tau_error,'l2')
    tau_err_L2 = norm(tau_errfun, 'L2')
    tau_err_linf = norm(tau_error,'linf')
    tau_err_Linf = norm(tau_errfun, 'H1')

    p_err_l2 = norm(p_error,'l2')
    p_err_L2 = norm(p_errfun, 'L2')
    p_err_linf = norm(p_error,'linf')
    p_err_Linf = norm(p_errfun, 'H1')

    m_size.append(np.log(1.0/mesh.hmin()))
    u_errl2.append(np.log(u_err_l2))  
    u_errL2.append(np.log(u_err_L2)) 
    u_errlinf.append(np.log(u_err_linf)) 
    u_errLinf.append(np.log(u_err_Linf)) 

    tau_errl2.append(np.log(tau_err_l2))  
    tau_errL2.append(np.log(tau_err_L2)) 
    tau_errlinf.append(np.log(tau_err_linf)) 
    tau_errLinf.append(np.log(tau_err_Linf)) 

    p_errl2.append(np.log(p_err_l2))  
    p_errL2.append(np.log(p_err_L2)) 
    p_errlinf.append(np.log(p_err_linf)) 
    p_errLinf.append(np.log(p_err_Linf)) 

 
    


    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E5 and j==4 or j==3:
        # Velocity Norms
        plt.figure(0)
        #plt.plot(m_size, u_errl2, 'r-', label=r'$log||e_u||$')
        plt.plot(m_size, u_errL2, 'b-', label=r'$log||e_u||_{L2}$')
        plt.plot(m_size, u_errlinf, 'c-', label=r'$log||e_u||_{l_\infty}$')
        plt.plot(m_size, u_errLinf, 'm-', label=r'$log||e_u||_{H_1}$')
        plt.legend(loc='best')
        plt.xlabel('$log(1/h)$')
        plt.ylabel('$log||e_u||$')
        plt.savefig("Incompressible Viscoelastic Flow Results/Error/vel_convergenceTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+"DEVSS"+str(th)+".png")
        plt.clf()
        # Pressure Norms
        plt.figure(1)
        #plt.plot(m_size, tau_errl2, 'r-', label=r'$||e_p||$')
        plt.plot(m_size, tau_errL2, 'b-', label=r'$||e_p||_{L2}$')
        plt.plot(m_size, tau_errlinf, 'c-', label=r'$||e_p||_{l_\infty}$')
        plt.plot(m_size, tau_errLinf, 'm-', label=r'$||e_p||_{H_1}$')
        plt.legend(loc='best')
        plt.xlabel('$log(1/h)$')
        plt.ylabel('$log||e_{\tau}||$')
        plt.savefig("Incompressible Viscoelastic Flow Results/Error/pressure_convergenceTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+"DEVSS"+str(th)+".png")
        plt.clf()
        # Stress Norms
        plt.figure(2)
        #plt.plot(m_size, p_errl2, 'r-', label=r'$||e_\tau||$')
        plt.plot(m_size, p_errL2, 'b-', label=r'$||e_\tau||_{L2}$')
        plt.plot(m_size, p_errlinf, 'c-', label=r'$||e_\tau||_{l_\infty}$')
        plt.plot(m_size, p_errLinf, 'm-', label=r'$||e_\tau||_{H_1}$')
        plt.legend(loc='best')
        plt.xlabel('$log(1/h)$')
        plt.ylabel('$log||e_p||$')
        plt.savefig("Incompressible Viscoelastic Flow Results/Error/stress_convergenceTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+"DEVSS"+str(th)+".png")
        plt.clf()





    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E5 and j==1 or j==4 or j==3:

        # Plot Stress/Normal Stress Difference
        tau_e = project(tau_vec_exact - tau1_vec, Zc)
        tau_xxe = project(tau_e[0],Q)
        mplot(tau_xxe)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/tau_xx_errorRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"t="+str(t)+"DEVSS"+str(th)+".png")
        plt.clf() 
        tau_xye = project(tau_e[1],Q)
        mplot(tau_xye)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/tau_xy_errorRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"t="+str(t)+"DEVSS"+str(th)+".png")
        plt.clf() 
        tau_yye = project(tau_e[2],Q)
        mplot(tau_yye)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/tau_yy_errorRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"t="+str(t)+"DEVSS"+str(th)+".png")
        plt.clf() 
        """tau_xx = project(tau1_vec[0],Q)
        mplot(tau_xx)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/tau_xxRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"t="+str(t)+"DEVSS"+str(th)+".png")
        plt.clf() 
        tau_xy = project(tau1_vec[1],Q)
        mplot(tau_xy)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/tau_xyRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"t="+str(t)+"DEVSS"+str(th)+".png")
        plt.clf() 
        tau_yy = project(tau1_vec[2],Q)
        mplot(tau_yy)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/tau_yyRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"t="+str(t)+"DEVSS"+str(th)+".png")
        plt.clf() 
        tau_xx_exact = project(tau_vec_exact[0],Q)
        mplot(tau_xx_exact)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/tau_xx_exactRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"t="+str(t)+"DEVSS"+str(th)+".png")
        plt.clf() 
        tau_xy_exact = project(tau_vec_exact[1],Q)
        mplot(tau_xy_exact)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/tau_xy_exactRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"t="+str(t)+"DEVSS"+str(th)+".png")
        plt.clf() 
        tau_yy_exact = project(tau_vec_exact[2],Q)
        mplot(tau_yy_exact)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/tau_yy_exactRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"t="+str(t)+"DEVSS"+str(th)+".png")
        plt.clf() 
        #N1=project(tau1[0,0]-tau1[1,1],Q)
        #mplot(N1)
        #plt.colorbar()
        #plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/FirstNormalStressDifferenceRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
        #plt.clf()"""

    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E4 and j==1 or j==3 or j==4:
        # Plot Velocity Components
        u_e = project(u_exact - u1, V)
        uxe=project(u_e[0],Q)
        mplot(uxe)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/u_xerrorRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"t="+str(t)+"DEVSS"+str(th)+".png")
        plt.clf()
        uye = project(u_e[1],Q)
        mplot(uye)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/u_yerrorRe="+str(Rey)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"t="+str(t)+"DEVSS"+str(th)+".png")
        plt.clf()

    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E5 and j==1 or j==4:


        # Matlab Plot of the Solution at t=Tf
        #p1=mu_0*(L/U)*p1  #Dimensionalised Pressure
        #p1=project(p1,Q)
        mplot(p1)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/PressureRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(c1)+".png")
        plt.clf()


        #Plot Contours USING MATPLOTLIB
        # Scalar Function code


        x = Expression('x[0]', degree=2)  #GET X-COORDINATES LIST
        y = Expression('x[1]', degree=2)  #GET Y-COORDINATES LIST
        pvals = p1.vector().array() # GET SOLUTION p= p(x,y) list
        Tvals = T1.vector().array() # GET SOLUTION T= T(x,y) list
        xyvals = mesh.coordinates()     # CLEAN THIS UP!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        xvalsq = interpolate(x, Q)#xyvals[:,0]
        yvalsq= interpolate(y, Q)#xyvals[:,1]
        xvalsw = interpolate(x, Qt)#xyvals[:,0]
        yvalsw= interpolate(y, Qt)#xyvals[:,1]

        xvals = xvalsq.vector().array()
        yvals = yvalsq.vector().array()


        xx = np.linspace(0,1)
        yy = np.linspace(0,1)
        XX, YY = np.meshgrid(xx,yy)   # (x,y) coordinate data formatted so that it can be used by plt.contour()
        pp = mlab.griddata(xvals, yvals, pvals, xx, yy, interp='nn') # u(x,y) data so that it can be used by 


        plt.contour(XX, YY, pp, 25)
        plt.title('Pressure Contours')   # PRESSURE CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/PressureContoursRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"t="+str(t)+"DEVSS"+str(th)+".png")
        plt.clf()



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
                       density=2.5,              
                       color=speed,  
                       cmap=cm.gnuplot,                         # colour map
                       linewidth=0.75)                           # line thickness
                                                                # arrow size
        plt.colorbar()                                          # add colour bar on the right
        plt.title('Lid Driven Cavity Flow')
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/VelocityContoursRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"t="+str(t)+"Stabilisation"+str(th)+".png")   
        plt.clf()                                               # display the plot


    plt.close()



    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E5:
        Tf=T_f 
       


    # Reset Functions
    p0=Function(Q)       # Pressure Field t=t^n
    p1=Function(Q)       # Pressure Field t=t^n+1
    T0=Function(Qt)       # Temperature Field t=t^n
    T1=Function(Qt)       # Temperature Field t=t^n+1
    tau0_vec=Function(Ze)     # Stress Field (Vector) t=t^n
    tau12_vec=Function(Ze)    # Stress Field (Vector) t=t^n+1/2
    tau1_vec=Function(Ze)     # Stress Field (Vector) t=t^n+1
    w0= Function(W)
    w12= Function(W)
    ws= Function(W)
    w1= Function(W)
    (u0, D0_vec)=w0.split()
    (u12, D12_vec)=w0.split()
    (us, Ds_vec)=w0.split()
    (u1, D1_vec)=w0.split()
    D_proj_vec = Function(Zd)
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




