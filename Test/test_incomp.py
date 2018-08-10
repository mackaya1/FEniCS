"""Test for Convergence of Numerical Scheme"""
"""Ernesto Castillo - PHD Thesis 2016"""

"""Base Code for the Incompressible flow Numerical Scheme Test"""


""" EXACT SOLUTIONS

    u = 
    tau = 
    p =      """

from decimal import *
from dolfin import *
from mshr import *
from math import pi, sin, cos, sqrt, fabs
import numpy as np
from matplotlib.pyplot import cm, ion
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
    base_mesh= UnitSquareMesh(nx, ny)


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
    editor.open(mesh1,"triangle",2,2)
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
    h_min = mesh.hmin()

    mplot(base_mesh)
    plt.savefig("mesh_"+str(mm)+".eps")
    plt.clf()
    plt.close() 

    class No_slip(SubDomain):
          def inside(self, x, on_boundary):
              return True if on_boundary else False 

    class Bottom_Point(SubDomain):
          def inside(self, x, on_boundary):
              return True if near(x[0], 0.0, 2*h_min) and near(x[1], 0.0, 2*h_min) else False 

    class Top_Wall(SubDomain):
          def inside(self, x, on_boundary):
            return True if near(x[1], 1.0) else False  

    class Left_Wall(SubDomain):
          def inside(self, x, on_boundary):
            return True if near(x[0], 0.0) else False                                                                             

    no_slip = No_slip()
    bottom_point = Bottom_Point()
    top_wall = Top_Wall()
    left_wall = Left_Wall()


    # MARK SUBDOMAINS (Create mesh functions over the cell facets)
    sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    sub_domains.set_all(5)
    no_slip.mark(sub_domains, 0)
    bottom_point.mark(sub_domains, 1)
    top_wall.mark(sub_domains, 2)
    


    #plot(sub_domains, interactive=True)        # DO NOT USE WITH RAVEN
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
    Z_e = MixedElement(Z_c,Z_se)
    #Z_e = EnrichedElement(Z_c,Z_se)                 # Enriched Elements
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
        u_array = u.vector().get_local()
        u_max = np.max(np.abs(u_array))
        u_array /= u_max
        u.vector()[:] = u_array
        #u.vector().set_local(u_array)  # alternative
        return u

    def magnitude(u):
        return np.power((u[0]*u[0]+u[1]*u[1]), 0.5)

    def sigma(u, p, Tau):
        return 2*betav*Dcomp(u) - p*Identity(len(u)) + Tau

    def Fdef(u, Tau):
        return dot(u,grad(Tau)) - dot(grad(u),Tau) - dot(Tau,tgrad(u)) 

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

    def absolute(u):
        u_array = np.absolute(u.vector().get_local())
        u.vector()[:] = u_array
        return u
        

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
    T_f = 1.0
    Tf = T_f
    tol = 10E-6
    defpar = 1.0

    conv = 1                                      # Non-inertial Flow Parameter (Re=0)
    We = 0.1
    Re = 0.5
    Ma = 0.001
    betav = 1.0 - DOLFIN_EPS

    alph1 = 0.0
    alph2 = 10E-20
    alph3 = 10E-20
    th = 0 +(jjj)*0.25              # DEVSS

    c1 = 0.1           # SUPG / SU
    c2 = 0.05                  # Artificial Diffusion
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
    F_u = Expression(('Re*x[0]*(-exp(-t) + exp(-2*t)) - 2*((1.0-betav)/We)*exp(-t)',
                      '-Re*x[1]*(-exp(-t) + exp(-2*t)) - 2*((1.0-betav)/We)*exp(-t)'), degree=2, pi=pi, Re=Re, We=We, t=t, betav=betav)
    F_tau_vec = Expression(('(1.0-We)*x[0]*exp(-t)  - We*(exp(-2*t)*x[0] + 2.0*(exp(-t)))-1.0',
                            '(x[0]+x[1])*(exp(-t)-We*exp(-t)) + We*(x[0]-x[1])*exp(-2*t)',
                            '(1.0-We)*x[1]*exp(-t) + We*(exp(-2*t)*x[1]+ 2.0*(exp(-t)))-1.0'), degree=2, pi=pi, Re=Re, t=t, We=We, betav=betav)
    F_tau_vecl2 = interpolate(F_tau_vec, Z)
    F_tau = as_matrix([[F_tau_vec[0], F_tau_vec[1]],
                       [F_tau_vec[1], F_tau_vec[2]]])  



    # Interpolate Stabilisation Functions
    h = CellSize(mesh)
    h_k = project(h/mesh.hmax(), Qt)
    n = FacetNormal(mesh)

    u_l2 = interpolate(u_exact, V)
    tau_vec_l2 = interpolate(tau_vec_exact, Z)
    p_l2 = interpolate(p_exact, Q)

    # Initial conditions
    assign(w0.sub(0), u_l2)
    assign(tau0_vec, tau_vec_l2)

    tau0 = as_matrix([[tau0_vec[0], tau0_vec[1]],
                      [tau0_vec[1], tau0_vec[2]]])        # Stress


    I_vec = as_vector([1.0,0,1.0])


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


    Np= len(p0.vector().get_local())
    Nv= len(w0.vector().get_local())   
    Ntau= len(tau0_vec.vector().get_local())
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


    # Governing Equations
    




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

    solveru = KrylovSolver("bicgstab", "hypre_amg")
    solvertau = KrylovSolver("bicgstab", "hypre_amg")
    solverp = KrylovSolver("bicgstab", "hypre_amg")

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
    #tau_step = project(tau1_vec - tau0_vec, Z)
    #stop_crit = norm(tau_step.vector(), 'l2')

    t = 0.0
    iter = 0            # iteration counter
    maxiter = 10000000
    while t < Tf + DOLFIN_EPS and iter < maxiter:
        t += dt
        iter += 1
        print"t = %s,  Iteration = %d, Convergence Failures = %s, Loop = %s" %(t, iter, jj, j)

        #tau_step = project(tau1_vec - tau0_vec, Z)
        #stop_crit = norm(tau_step, 'L2')
        #print stop_crit

        #Update Boundary Condidtions and Exact Solution
        u_exact.t=t
        tau_vec_exact.t=t
        p_exact.t=t

        F_u.t = t
        F_tau_vec.t = t
        F_tau = as_matrix([[F_tau_vec[0], F_tau_vec[1]],
                           [F_tau_vec[1], F_tau_vec[2]]])  

        F_test = project(tau1_vec[1] , Q)

        u_l2 = interpolate(u_exact, V)
        tau_vec_l2 = interpolate(tau_vec_exact, Z)
        p_l2 = interpolate(p_exact, Q)
        
        u_bound  = DirichletBC(W.sub(0), u_exact, no_slip)  # Dirichlet Boundary conditions for Velocity 
        tau_bound  =  DirichletBC(Z, tau_vec_exact, no_slip)  # Dirichlet Boundary conditions for Stress
        p_bound = DirichletBC(Q, p_exact, no_slip) # Dirichlet Boundary conditions for Pressure
        bcu = [u_bound]
        bcp = [p_bound]
        bctau = [tau_bound]




        # Update Stabilisation (Copy and Paste Stabilisation Technique from above)
        F1R = Fdef(u1, tau1)  #Compute the residual in the STRESS EQUATION
        F1R_vec = as_vector([F1R[0,0], F1R[1,0], F1R[1,1]])
        Dincomp1_vec = as_vector([Dincomp(u1)[0,0], Dincomp(u1)[1,0], Dincomp(u1)[1,1]])
        #restau0 = We/dt*(tau1_vec-tau0_vec) + We*F1R_vec + tau1_vec - I_vec
        restau0 = tau1_vec - tau_vec_l2
        res_test = project(restau0, Zd)
        res_orth = project(restau0-res_test, Z)                                
        res_orth_norm_sq = project(inner(restau0,restau0), Qt)     # Project residual norm onto discontinuous space
        res_orth_norm = np.power(res_orth_norm_sq, 0.5)
        kapp = project(res_orth_norm, Qt)
        kapp = absolute(kapp)
        LPSl_stress = inner(kapp*h*c1*grad(tau),grad(Rt))*dx + inner(kapp*h*c2*div(tau),div(Rt))*dx  # Stress Stabilisation

        # Update SU Term
        alpha_supg = h/(magnitude(u1)+0.0000000001)
        SU = inner(dot(u1, grad(tau)), alpha_supg*dot(u1,grad(Rt)))*dx  

                
        U12 = 0.5*(u1 + u0)    
        # Update Solutions
        if iter > 1:
            w0.assign(w1)
            T0.assign(T1)
            p0.assign(p1)
            tau0_vec.assign(tau1_vec)

        #if iter > 10:
            #bcu = [noslip, noslip_s, freeslip, drive, fully_developed]             

        (u0, D0_vec)=w0.split()  

        D0 = as_matrix([[D0_vec[0], D0_vec[1]],
                        [D0_vec[1], D0_vec[2]]])                    #DEVSS STABILISATION
        DEVSSr_u12 = 2*(1-betav)*inner(D0,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS


        U = 0.5*(u + u0)     

        

        # VELOCITY HALF STEP (ALTERNATIVE)
        lhsFu12 = Re*(2.0*(u - u0) / dt + conv*dot(u0, nabla_grad(u0)))
        Fu12 = dot(lhsFu12, v)*dx \
               + inner(2.0*betav*Dincomp(u0), Dincomp(v))*dx - ((1. - betav)/We)*inner(div(tau0-Identity(len(u))), v)*dx + inner(grad(p0),v)*dx - inner(F_u,v)*dx\
               + inner(D-Dincomp(u),R)*dx 

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

        (u12, D12_vec) = w12.split()
        D12 = as_matrix([[D12_vec[0], D12_vec[1]],
                        [D12_vec[1], D12_vec[2]]])
        DEVSSr_u1 = 2*(1-betav)*inner(D12,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS



        # Stress Half Step
        lhs_tau12 = (2.0*We/dt + 1.0)*tau  +  We*Fdef(u12,tau)                            # Left Hand Side
        rhs_tau12= (2.0*We/dt)*tau0 + Identity(len(u)) + F_tau

        A = inner(lhs_tau12,Rt)*dx - inner(rhs_tau12,Rt)*dx
        a3 = lhs(A)
        L3 = rhs(A) 

            # SUPG / SU / LPS Stabilisation (User Choose One)

        a3 += LPSl_stress  # [SUPGl4, SUl4, LPSl_stab, LPSl_stress, diff_stab, 0]
        L3 += 0  # [SUPGr4, SUr4, LPSr_stab, LPS_res_stab, 0]   


        A3=assemble(a3)                                     # Assemble System
        b3=assemble(L3)
        [bc.apply(A3, b3) for bc in bctau]
        solvertau.solve(A3, tau12_vec.vector(), b3)
        end()
        

        #Predicted U* Equation (ALTERNATIVE)
        lhsFus = Re*((u - u0)/dt + conv*dot(u12, nabla_grad(U)))
        Fus = dot(lhsFus, v)*dx + \
               + inner(2.0*betav*Dincomp(U), Dincomp(v))*dx - ((1. - betav)/(We+DOLFIN_EPS))*inner(div(tau12-Identity(len(u))), v)*dx + inner(grad(p0),v)*dx - inner(F_u,v)*dx\
               + inner(D-Dincomp(u),R)*dx   
              
        a2= lhs(Fus)
        L2= rhs(Fus)


            # Stabilisation
        a2+= th*DEVSSl_u1   #[th*DEVSSl_u12]                     
        L2+= th*DEVSSr_u1    #[th*DEVSSr_u12]

        A2 = assemble(a2)        
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcu]
        solve(A2, ws.vector(), b2, "bicgstab", "default")
        end()
        (us, Ds_vec) = ws.split()

        F_u_V = interpolate(F_u, V)
        divFu = project(div(us), Q)

        #PRESSURE CORRECTION
        a5=inner(grad(p),grad(q))*dx 
        L5=inner(grad(p0),grad(q))*dx - 0.5*(Re/dt)*div(us)*q*dx #+ inner(div(F_u_V),q)*dx
        A5 = assemble(a5)
        b5 = assemble(L5)
        [bc.apply(A5, b5) for bc in bcp]
        #[bc.apply(p1.vector()) for bc in bcp]
        solve(A5, p1.vector(), b5, "cg", prec)
        end()
        
        #Velocity Update
        lhs_u1 = (Re/dt)*u                                          # Left Hand Side
        rhs_u1 = (Re/dt)*us                                         # Right Hand Side

        a7=inner(lhs_u1,v)*dx + inner(D-Dincomp(u),R)*dx                                           # Weak Form
        L7=inner(rhs_u1,v)*dx - 0.5*inner(grad(p1-p0),v)*dx  

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
        rhs_tau1= (We/dt)*tau0 + Identity(len(u)) + F_tau

        A = inner(lhs_tau1,Rt)*dx - inner(rhs_tau1,Rt)*dx
        a4 = lhs(A)
        L4 = rhs(A) 

            # SUPG / SU / LPS Stabilisation (User Choose One)

        a4 += SU + LPSl_stress  # [SUPGl4, SUl4, LPSl_stab, LPSl_stress, diff_stab, 0]
        L4 += 0  # [SUPGr4, SUr4, LPSr_stab, LPS_res_stab, 0]   


        A4=assemble(a4)                                     # Assemble System
        b4=assemble(L4)
        [bc.apply(A4, b4) for bc in bctau]
        solvertau.solve(A4, tau1_vec.vector(), b4)
        end()

     

        # Record Covergence Data 
        #u1_vec = interpolate(u1,V)

        if iter > 5:
            u_errfun = project(u1 - u_l2, V)
            tau_errfun = project(tau1_vec - tau_vec_l2, Z)
            p_errfun = project(p1 - p_l2, Q)

            u_err_L2_list.append(norm(u_errfun,'L2'))
            u_err_H1_list.append(norm(u_errfun, 'H1'))

            tau_err_L2_list.append(norm(tau_errfun,'L2'))
            tau_err_H1_list.append(norm(tau_errfun, 'H1'))

            p_err_L2_list.append(norm(p_errfun, 'L2'))
            p_err_H1_list.append(norm(p_errfun, 'H1')) 


        # Break Loop if code is diverging

        if norm(w1.vector(), 'linf') > 10E5 or np.isnan(sum(w1.vector().get_local())):
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

        #U_x = project(ws[0],Q)
        #U_y = project(u_exact[1],Q)
        #tau_xx = project(tau1_vec[0],Q)



        # Plot solution
        """if t> 0.01:
            tau_e = project(tau_vec_exact - tau1_vec, Zc)
            tau_xxe = project(tau_e[0],Q)
            plt.close()
            mplot(tau_xxe)
            plt.colorbar()
            plt.show(block=False)"""




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


    # RECORD ERROR NORMS

    # Square of spatial norms at each time step
    u_err_L2_list_sq = np.power(u_err_L2_list, 2)
    u_err_H1_list_sq = np.power(u_err_H1_list, 2)
    tau_err_L2_list_sq = np.power(tau_err_L2_list, 2)
    tau_err_H1_list_sq = np.power(tau_err_H1_list, 2)
    p_err_L2_list_sq = np.power(p_err_L2_list, 2)
    p_err_H1_list_sq = np.power(p_err_H1_list, 2)

    # Velocity error norms
    u_err_0_0_sq = trapezium(u_err_L2_list_sq, dt)      # l2 norm (time) of L2 (space) norms of the error function 
    u_err_0_0 = np.power(u_err_0_0_sq, 0.5)
    u_err_1_0_sq = trapezium(u_err_H1_list_sq, dt)          # l2 norm (time) of H1 (space) norms of the error function
    u_err_1_0 = np.power(u_err_1_0_sq, 0.5)
    u_err_0_inf = max(u_err_L2_list)                 # Supremum (time) of L2 (space) norms of the error function 
    u_err_1_inf = max(u_err_H1_list)                 # Supremum (time) of H1 (space) norms of the error function 

    tau_err_0_0_sq = trapezium(tau_err_L2_list_sq, dt)      # l2 norm (time) of L2 (space) norms of the error function 
    tau_err_0_0 = np.power(tau_err_0_0_sq, 0.5)
    tau_err_1_0_sq = trapezium(tau_err_H1_list_sq, dt)          # l2 norm (time) of H1 (space) norms of the error function
    tau_err_1_0 = np.power(tau_err_1_0_sq, 0.5)        
    tau_err_0_inf = max(tau_err_L2_list)                 
    tau_err_1_inf = max(tau_err_H1_list) 

    p_err_0_0_sq = trapezium(p_err_L2_list_sq, dt)      # l2 norm (time) of L2 (space) norms of the error function 
    p_err_0_0 = np.power(p_err_0_0_sq, 0.5)
    p_err_1_0_sq = trapezium(p_err_H1_list_sq, dt)          # l2 norm (time) of H1 (space) norms of the error function
    p_err_1_0 = np.power(p_err_1_0_sq, 0.5)         
    p_err_0_inf = max(p_err_L2_list)                 
    p_err_1_inf = max(p_err_H1_list) 

    m_size.append(np.log(1.0/mesh.hmin()))
    u_errl2.append(np.log(u_err_0_inf))  
    u_errL2.append(np.log(u_err_0_0 )) 
    u_errlinf.append(np.log(u_err_0_inf)) 
    u_errLinf.append(np.log(u_err_1_0)) 

    tau_errl2.append(np.log(tau_err_0_0))  
    tau_errL2.append(np.log(tau_err_1_0)) 
    tau_errlinf.append(np.log(tau_err_0_inf)) 
    tau_errLinf.append(np.log(tau_err_1_0)) 

    p_errl2.append(np.log(p_err_0_inf))  
    p_errL2.append(np.log(p_err_0_0)) 
    p_errlinf.append(np.log(p_err_0_inf)) 
    p_errLinf.append(np.log(p_err_1_0)) 


    h_squared_x = [2.5, 3.5]
    h_squared_y = [-6.5, -8.5]

    # Error Data
    if j==4 or j==1:
        with open("Convergence Results.txt", "a") as text_file:
             text_file.write("Re="+str(Rey*conv)+", We="+str(We)+", t="+str(t)+", mm="+str(mm)+'\n')  
             text_file.write("Velocity: u_errl2="+str(u_errl2)+'\n'+", u_errL2="+str(u_errL2)+'\n'+", u_errlinf="+str(u_errlinf)+'\n'+", u_errLinf="+str(u_errLinf)+'\n')  
             text_file.write("Pressure: p_errl2="+str(p_errl2)+'\n'+", p_errL2="+str(p_errL2)+'\n'+", p_errlinf="+str(p_errlinf)+'\n'+", p_errLinf="+str(p_errLinf)+'\n')
             text_file.write("Stress: u_errl2="+str(tau_errl2)+'\n'+", tau_errL2="+str(tau_errL2)+'\n'+", tau_errlinf="+str(tau_errlinf)+'\n'+", tau_errLinf="+str(tau_errLinf)+'\n'+'\n')     


    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E5 and j==loopend or j==1 or j==2:
        # Velocity Norms
        plt.figure(0)
        plt.plot(h_squared_x, h_squared_y, color='black', label=r'$O(h^2)$', linewidth='2.0' )
        plt.plot(m_size, u_errl2, 'r-*', label=r'$||e_{\rho}||_{0,\infty}$')
        plt.plot(m_size, u_errL2, 'b:*', label=r'$||e_{\rho}||_{0,0}$')
        plt.plot(m_size, u_errlinf, 'c--o', label=r'$||e_{\rho}||_{1,\infty}$')
        plt.plot(m_size, u_errLinf, 'm-o', label=r'$||e_{\rho}||_{1,0}$')
        plt.legend(loc = 'best')
        plt.xlabel('$log(1/h)$')
        plt.ylabel(r'$log||e_{\rho}||$')
        plt.savefig("Incompressible Viscoelastic Flow Results/Error/density_convergenceTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+"DEVSS"+str(th)+".png")
        plt.clf()
        # Pressure Norms
        plt.figure(1)
        plt.plot(h_squared_x, h_squared_y, color='black', label=r'$O(h^2)$', linewidth='2.0' )
        plt.plot(m_size, tau_errl2, 'r*-', label=r'$||e_{\tau}||_{0,\infty}$')
        plt.plot(m_size, tau_errL2, 'b*-', label=r'$||e_{\tau}||_{0,0}$')
        plt.plot(m_size, tau_errlinf, 'c*-', label=r'$||e_{\tau}||_{1,\infty}$')
        plt.plot(m_size, tau_errLinf, 'm*-', label=r'$||e_{\tau}||_{1,0}$')
        plt.legend(loc = 'best')
        plt.xlabel('$log(1/h)$')
        plt.ylabel(r'$log||e_{\tau}||$')
        plt.savefig("Incompressible Viscoelastic Flow Results/Error/stress_convergenceTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+"DEVSS"+str(th)+".png")
        plt.clf()
        # Stress Norms
        plt.figure(2)
        plt.plot(h_squared_x, h_squared_y, color='black', label=r'$O(h^2)$', linewidth='2.0' )
        plt.plot(m_size, p_errl2, 'r-*', label=r'$||e_{\rho}||_{0,\infty}$')
        plt.plot(m_size, p_errL2, 'b*-', label=r'$||e_{\rho}||_{0,0}$')
        plt.plot(m_size, p_errlinf, 'c*-', label=r'$||e_{\rho}||_{1,\infty}$')
        plt.plot(m_size, p_errLinf, 'm*-', label=r'$||e_{\rho}||_{1,0}$')
        plt.legend(loc='best')
        plt.xlabel('$log(1/h)$')
        plt.ylabel(r'$log||e_{\rho}||$')
        plt.savefig("Incompressible Viscoelastic Flow Results/Error/pressure_convergenceTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+"DEVSS"+str(th)+".png")
        plt.clf()





    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E5 and j==loopend or j==1 or j==2:

        # Plot Stress/Normal Stress Difference
        tau_e = project(tau_vec_exact - tau1_vec, Zc)
        tau_xxe = project(tau_e[0],Q)
        mplot(tau_xxe)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/tau_xx_errorRe="\
                    +str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"t="+str(t)+"DEVSS"+str(th)+".png")
        plt.clf() 
        tau_xye = project(tau_e[1],Q)
        mplot(tau_xye)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/tau_xy_errorRe="\
                    +str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"t="+str(t)+"DEVSS"+str(th)+".png")
        plt.clf() 
        tau_yye = project(tau_e[2],Q)
        mplot(tau_yye)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/tau_yy_errorRe="\
                    +str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"t="+str(t)+"DEVSS"+str(th)+".png")
        plt.clf() 
        #N1=project(tau1[0,0]-tau1[1,1],Q)
        #mplot(N1)
        #plt.colorbar()
        #plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/FirstNormalStressDifferenceRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
        #plt.clf()

    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E4 and j==loopend or j==1 or j==2:
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

    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E5 and j==4 or j==1 or j==2:


        # Matlab Plot of the Solution at t=Tf
        mplot(p1)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/PressureRe="+str(Rey*conv)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(c1)+".png")
        plt.clf()


        #Plot Contours USING MATPLOTLIB
        # Scalar Function code


        x = Expression('x[0]', degree=2)  #GET X-COORDINATES LIST
        y = Expression('x[1]', degree=2)  #GET Y-COORDINATES LIST
        pvals = p1.vector().get_local() # GET SOLUTION p= p(x,y) list
        Tvals = T1.vector().get_local() # GET SOLUTION T= T(x,y) list
        xyvals = mesh.coordinates()     # CLEAN THIS UP!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        xvalsq = interpolate(x, Q)#xyvals[:,0]
        yvalsq= interpolate(y, Q)#xyvals[:,1]
        xvalsw = interpolate(x, Qt)#xyvals[:,0]
        yvalsw= interpolate(y, Qt)#xyvals[:,1]

        xvals = xvalsq.vector().get_local()
        yvals = yvalsq.vector().get_local()


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
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/VelocityContoursRe="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"mesh="+str(mm)+"t="+str(t)+"Stabilisation"+str(th)+".png")   
        plt.clf()                                               # display the plot


    plt.close()



    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E5:
        Tf=T_f 
       

    if jjj==4:
        quit()

    if j==loopend:
        jjj+=1
        j=0
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

    # Reset Functions
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




