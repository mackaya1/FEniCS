

from __future__ import print_function
from fenics import *
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt

T = 3.0           # final time
num_steps = 500    # number of time steps
dt = T / num_steps # time step size
betav = 0.75             # kinematic viscosity
Re = 0.5            # density
We = 0.25
conv = 1

# Create mesh and define function spaces
mesh = UnitSquareMesh(30, 30)
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

family = "CG"; dfamily = "DG"; rich = "Bubble"
shape = "triangle"; order = 2

Z_c = VectorElement(family, mesh.ufl_cell(),  order, 3)
Z_s = VectorElement(dfamily, mesh.ufl_cell(),  order-1, 3)
Z_se = VectorElement(rich, mesh.ufl_cell(),  order+1, 3)
Z_d = VectorElement(dfamily, mesh.ufl_cell(),  order-2, 3)  # Stress Elements

Z = FunctionSpace(mesh,Z_s)
Zd = FunctionSpace(mesh,Z_d)

# Define boundaries
inflow  = 'near(x[0], 0)'
outflow = 'near(x[0], 1)'
walls   = 'near(x[1], 0) || near(x[1], 1)'


u_exact = Expression(('4.0*x[1]*(1.0-x[1])','0'), degree=2) # Velocity
tau_vec_exact = Expression(('8*(1.0-betav)*We*(1.0-2.0*x[1])*(1.0-2.0*x[1])',
                            '4*(1.0-betav)*(1.0-2.0*x[1])',
                            '0.0'), degree=2, t=0.0, betav=betav, We=We)
p_exact = Expression('8*(1.0-x[0])', degree=2, t=0.0) # Pressure

# Define boundary conditions
bcu_inflow  = DirichletBC(V, u_exact, inflow)
bcu_noslip  = DirichletBC(V, Constant((0, 0)), walls)
bcp_inflow  = DirichletBC(Q, Constant(8), inflow)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bctau_inflow = DirichletBC(Z, tau_vec_exact, outflow) 
bcu = [ bcu_noslip]
bcp = [bcp_outflow, bcp_inflow]     #bcp_inflow
bctau = [bctau_inflow]

# Define expressions used in variational forms

n   = FacetNormal(mesh)
f   = Constant((0, 0))
k   = Constant(dt)
Rey = Re
Re  = Constant(Re)
Wi = We
We = Constant(We)
beta = betav
betav = Constant(betav)
th = 0.0



# Define strain-rate tensor
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p, Tau):
    return 2*betav*epsilon(u) - p*Identity(len(u)) + Tau

def  tgrad (w):
    """ Returns  transpose  gradient """
    return  transpose(grad(w))
def Dincomp (w):
    """ Returns 2* the  rate of  strain  tensor """
    return (grad(w) + tgrad(w))/2
def Dcomp (w):
    """ Returns 2* the  rate of  strain  tensor """
    return ((grad(w) + tgrad(w))-(2.0/3)*div(w)*I)/2

def Fdef(u, Tau):
    return We*(dot(u,grad(Tau)) - dot(grad(u),Tau) - dot(Tau,tgrad(u)) + div(u)*Tau)

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
j=0
x = list()
err_0 = list()
err_1 = list()
while j < 2:
    j+=1

    if j==1:
        th=0.0
    if j==2:
        th=1.0
    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)

    # Define functions for solutions at previous and current time steps
    u_n = Function(V)
    u_  = Function(V)
    p_n = Function(Q)
    p_  = Function(Q)
    tau_vec = TrialFunction(Z)
    Rt_vec = TestFunction(Z)


    tau0_vec=Function(Z)     # Stress Field (Vector) t=t^n
    tau12_vec=Function(Z)    # Stress Field (Vector) t=t^n+1/2
    tau1_vec=Function(Z)     # Stress Field (Vector) t=t^n+1
    Rt = as_matrix([[Rt_vec[0], Rt_vec[1]],
                     [Rt_vec[1], Rt_vec[2]]])        # DEVSS Space

    tau = as_matrix([[tau_vec[0], tau_vec[1]],
                     [tau_vec[1], tau_vec[2]]])  

    tau0 = as_matrix([[tau0_vec[0], tau0_vec[1]],
                      [tau0_vec[1], tau0_vec[2]]])        # Stress 

    tau12 = as_matrix([[tau12_vec[0], tau12_vec[1]],
                       [tau12_vec[1], tau12_vec[2]]]) 

    tau1 = as_matrix([[tau1_vec[0], tau1_vec[1]],
                      [tau1_vec[1], tau1_vec[2]]]) 


    U   = 0.5*(u_n + u)
    U12 = 0.5*(u_n + u_)  
    Dincompu_vec = as_vector([Dincomp(u_n)[0,0], Dincomp(u_n)[1,0], Dincomp(u_n)[1,1]])
    D0_vec = project(Dincompu_vec,Zd)
    D0 = as_matrix([[D0_vec[0], D0_vec[1]],
                    [D0_vec[1], D0_vec[2]]])        #DEVSS STABILISATION
     
    DEVSSl_u = 2*(1-betav)*inner(Dincomp(u),Dincomp(v))*dx    
    DEVSSr_u = 2*(1-betav)*inner(D0,Dincomp(v))*dx 

    # Define variational problem for Velocity Half Step
    F0 = 2.0*Re*dot((u - u_n) / k, v)*dx + \
         Re*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
       + inner(sigma(U, p_n, tau0), epsilon(v))*dx \
       + dot(p_n*n, v)*ds - dot(betav*nabla_grad(U)*n, v)*ds \
       - dot(tau0*n, v)*ds 
    a0 = lhs(F0)
    L0 = rhs(F0)

    # Define variational problem for Velocity step 1
    lhsFus = Re*((u - u_n)/dt + conv*dot(u_n, nabla_grad(u_n)))
    F1 = dot(lhsFus, v)*dx + \
       + inner(sigma(U, p_n, tau0), epsilon(v))*dx \
       + dot(p_n*n, v)*ds - dot(betav*nabla_grad(U)*n, v)*ds \
       - dot(tau0*n, v)*ds 
    a1 = lhs(F1)
    L1 = rhs(F1)

    a1 += th*DEVSSl_u
    L1 += th*DEVSSr_u

    # Define variational problem for stress Half Step
    """lhs_tau12 = (2.0*We/dt+1.0)*tau  +  Fdef(u_,tau)                        # Left Hand Side
    rhs_tau12= (2.0*We/dt)*tau0 + 2.0*(1.0-betav)*epsilon(U12) #+  F_tau       # Right Hand Side

    F4 = inner(lhs_tau12,Rt)*dx - inner(rhs_tau12,Rt)*dx
    a4 = lhs(F4)
    L4 = rhs(F4)""" 

    # Define variational problem for step 2
    a2 = dot(nabla_grad(p), nabla_grad(q))*dx
    L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

    # Define variational problem for velocity correction step
    a3 = dot(u, v)*dx
    L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

    # Define variational problem for stress update
    lhs_tau1 = (We/dt+1.0)*tau  +  Fdef(u_,tau)                         # Left Hand Side
    rhs_tau1= (We/dt)*tau0 + 2.0*(1.0-betav)*epsilon(U12) #+  F_tau       # Right Hand Side

    F5 = inner(lhs_tau1,Rt)*dx - inner(rhs_tau1,Rt)*dx
    a5 = lhs(F5)
    L5 = rhs(F5) 

    # Assemble matrices
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)
    #A4 = assemble(a4)
    A5 = assemble(a5)

    # Apply boundary conditions to matrices
    #[bc.apply(A1) for bc in bcu]
    #[bc.apply(A2) for bc in bcp]
    #[bc.apply(A4) for bc in bctau]

    # Time-stepping
    t = 0
    for i in range(num_steps):

        # Update current time
        t += dt
        U12 = 0.5*(u_n + u_)        
        D0_vec = project(Dincompu_vec,Zd)
        D0 = as_matrix([[D0_vec[0], D0_vec[1]],
                        [D0_vec[1], D0_vec[2]]])        #DEVSS STABILISATION  

        DEVSSr_u = 2*(1-betav)*inner(D0,Dincomp(v))*dx 
        L1 = rhs(F1)
        L1 += th*DEVSSr_u
        # Step 1: Tentative velocity step
        A1 = assemble(a1)
        b1 = assemble(L1)
        [bc.apply(A1,b1) for bc in bcu]
        solve(A1, u_.vector(), b1)

        # Step 2: Stress Half Step
        """A4 = assemble(a4)
        b4 = assemble(L4)
        [bc.apply(A4,b4) for bc in bctau]
        solve(A4, tau12_vec.vector(), b4)"""

        # Step 2: Pressure correction step
        b2 = assemble(L2)
        [bc.apply(A2,b2) for bc in bcp]
        solve(A2, p_.vector(), b2)

        # Step 3: Velocity correction step
        b3 = assemble(L3)
        solve(A3, u_.vector(), b3)

        # Step 5: Stress Correction
        A5 = assemble(a5)    
        b5 = assemble(L5)
        [bc.apply(A5,b5) for bc in bctau]
        solve(A5, tau1_vec.vector(), b5)

        # Plot solution
        #plot(u_)

        # Compute error
        u_e = Expression(('4*x[1]*(1.0 - x[1])', '0'), degree=2)
        u_e = interpolate(u_e, V)
        tau_vec_l2 = interpolate(tau_vec_exact, Z)
        u_errorinf = np.abs(u_e.vector().array() - u_.vector().array()).max()
        tau_error = np.abs(tau_vec_l2.vector().array() - tau1_vec.vector().array()).max()
        print('t = %.2f: u_error = %.3g, tau_error = %.3g' % (t, u_errorinf, tau_error))
        print('max u:', u_.vector().array().max())

        if j==1:
           x.append(t)
           err_0.append(u_errorinf)
        if j==2:
           err_1.append(u_errorinf)
        
        plot(u_)        

        # Update previous solution
        u_n.assign(u_)
        p_n.assign(p_)
        tau0_vec.assign(tau1_vec)

# Plot Convergence
plt.figure(0)
plt.plot(x, err_0, 'r-', label=r'No Stabilisation')
plt.plot(x, err_1, 'b--', label=r'DEVSS')
plt.legend(loc='best')
plt.xlabel('time(s)')
plt.ylabel('$||u-e_e||_{\infty}$')
plt.savefig("Incompressible Viscoelastic Flow Results/Error/DEVSSConvergenceRe="+str(Rey)+"We="+str(Wi)+"b="+str(beta)+"dt="+str(dt)+".png")
plt.clf()
plt.close()



# Hold plot
#interactive()
