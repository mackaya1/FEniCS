"""Inompressible Lid Driven Cavity Problem for an COMPRESSIBLE Oldroyd-B Fluid"""
"""Solution Method: Finite Element Method using DOLFIN (FEniCS)"""

from Sphere_base import *  # Import Base Code for LDC Problem

dt = 2*mesh.hmin()**2  #Time Stepping  
T_f = 15.0
Tf = T_f
loopend = 4
j = 0
jj = 0
jjj = 0
tol = 10E-5
defpar = 1.0

conv = 0                                      # Non-inertial Flow Parameter (Re=0)
We = 0.1 #0.01
Re = 1.0
Ma = 0.0005
betav = 0.5  #
c0 = 1./Ma

alph1 = 0.0
c1 = 0.01
c2 = 0.01
th = 1.0             # DEVSS

b = 5. # Maximal chain extension
lambda_d = 0.0 # dissipative parameter
corr = 1.0   #Drag Correction

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



    # Comparing different dissipative constants Numbers (We=0.1,0.2,0.3,0.4,0.5) at Re=__
    
    if jjj==0:
        We = 0.65
    if jjj==1:
        We = 1.0
    if jjj==3:
        We = 1.5

    betav = 0.5
    if j==1:
        betav= 1.0 - DOLFIN_EPS
        We=0.01
        lambda_d = 0.
    if j==2:
       lambda_d = 0.
    elif j==3:
       lambda_d = 0.1
    elif j==4:
       lambda_d = 0.2

    """if j==1:
       betav = 0.5
       We=0.01
       lambda_d = 0
       b=1000.0
    if j==2:
        b=10.0
    if j==3:
        b=20.0
    if j==4:
        b=50.0
    if j==5:
        b=100.0"""






    # Comparing different WEISSENBERG Numbers (We=0.1,0.2,0.3,0.4,0.5) at Re=__
    """betav=0.5
    if j==1:
       betav = 1.0 - DOLFIN_EPS
       We = 0.0001
    elif j==2:
       We = 0.1
    elif j==3:
       We = 0.25
    elif j==4:
       We = 0.5
    elif j==5:
       We = 1.0"""


    # Comparing different REYNOLDS NUMBERS Numbers (Re=0,5,10,25,50) at We=0.5
    """conv=1                                      # Non-inertial Flow Parameter (Re=0)
    We=0.5
    if j==1:
       conv=1
       Re=0.1
    elif j==2:
       Re=0.5
    elif j==3:
       Re=1.0
    elif j==4:
       Re=2.0
    elif j==5:
       Re=5.0"""


    # Comparing Effect of DEVSS/ SUPG Stabilisation Parameter
    """alph = 0.125
    th=10E-16
    c1=alph*h_ska    #SUPG Stabilisation
    We=0.5
    conv=10E-15
    Re=1
    if j==1:
        th=0
    elif j==2:
        th=0.1*(1.0-betav)
    elif j==3:
        th=0.5*(1.0-betav)"""



    # Comparing Effect of Diffusion Stabilisation Parameter
    """c1=h_ka     #SUPG Stabilisation
    th=0.1*(1.0-betav)          #DEVSS Stabilisation
    We=0.5
    conv=10E-15
    Re=1
    if j==1:
        c2=10E-6*h_ka
    elif j==2:
        c2=rampd*0.1*h_ka"""

    # Comparing the Effect of SUPG Stabilisation
    """th=10E-16        #DEVSS Stabilisation
    c2=10E-6*h_ka    #Diffusion Stabilisation
    We=0.5
    Re=10
    if j==1:
        c1=h_ka*10E-10
    elif j==2:
        c1=0.1*h_ka
    elif j==3:
        c1=h_ka"""
    
        

    # Continuation in Reynolds/Weissenberg Number Number (Re-->20Re/We-->20We)
    Ret=Expression('Re*(1.0+19.0*0.5*(1.0+tanh(0.7*t-4.0)))', t=0.0, Re=Re, degree=2)
    Rey=Re
    Wet=Expression('(We/100)*(1.0+99.0*0.5*(1.0+tanh(0.7*t-5.0)))', t=0.0, We=We, degree=2)


    print '############# TIME SCALE ############'
    print 'Timestep size (s):', dt
    print 'Finish Time (s):', Tf

    print '############# Scalings & Nondimensional Parameters ############'
    print 'Channel Length (m):', B
    print 'Characteristic Length:', L
    print 'Shpere Radius / Channel Width:', beta_ratio
    print 'Reynolds Number:', Rey
    print 'Non-inertial parameter:', conv
    print 'Weissenberg Number:', We
    print 'Dissipation Factor:', lambda_d
    print 'Viscosity Ratio:', betav
    print 'Diffusion Number:' ,Di
    print 'Viscous Heating Number:', Vh
    print 'Maximal chain extension:', b 
    print 'Dissipative parameter:', lambda_d 

    Np= len(p0.vector().get_local())
    Nv= len(w0.vector().get_local())   
    Ntau= len(tau0_vec.vector().get_local())
    dof= 3*Nv+2*Ntau+Np
    print '############# Discrete Space Characteristics ############'
    print 'Degree of Elements', order
    print 'Mesh: %s x %s' %(mm, mm)
    print('Size of Pressure Space = %d ' % Np)
    print('Size of Velocity Space = %d ' % len(u_test.vector().get_local()))
    print('Size of Velocity/DEVSS Space = %d ' % Nv)
    print('Size of Stress Space = %d ' % Ntau)
    print('Degrees of Freedom = %d ' % dof)
    print 'Number of Cells:', mesh.num_cells()
    print 'Number of Vertices:', mesh.num_vertices()
    print 'Minimum Cell Diamter:', mesh.hmin()
    print 'Maximum Cell Diamter:', mesh.hmax()
    print '############# Stabilisation Parameters ############'
    print 'DEVSS Momentum Term:', th

    print 'Loop:', jjj, '-', j    

    

    # Initial Density Field
    rho_initial = Expression('1.0', degree=1)
    rho_initial_guess = project(1.0, Q)
    rho0.assign(rho_initial_guess)


    # Initial Conformation Tensor
    I_vec = Expression(('1.0','0.0','1.0'), degree=2)
    initial_guess_conform = project(I_vec, Zc)
    assign(tau0_vec, initial_guess_conform)         # Initial guess for conformation tensor is Identity matrix

    tau0 = as_matrix([[tau0_vec[0], tau0_vec[1]],
                      [tau0_vec[1], tau0_vec[2]]])        # Stress 

    # Initial Temperature Field
    T_initial_guess = project(T_0, Q)
    T0.assign(T_initial_guess)



    # DEVSS Stabilisation
    
    DEVSSl_u12 = 2.*(1.-betav)*inner(Dincomp(u),Dincomp(v))*dx    
    DEVSSr_u12 = 2.*inner(D0,Dincomp(v))*dx   
    DEVSSl_u1 = 2.*(1.-betav)*inner(Dincomp(u),Dincomp(v))*dx    
    DEVSSr_u1 = 2.*inner(D12,Dincomp(v))*dx 



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

    #fp = File("Paraview_Results/Pressure Results Re="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"lambda"+str(lambda_d)+"/pressure "+str(t)+".pvd")
    #fN1 =File("Paraview_Results/N1 Results Re="+str(Re*conv)+"We="+str(We)+"b="+str(betav)+"lambda"+str(lambda_d)+"/N1 "+str(t)+".pvd") 

    #Lists for Energy Values
    x=list()
    ee=list()
    ek=list()
    #ftau=File("Incompressible Viscoelastic Flow Results/Paraview/Stress_th"+str(th)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"/stress "+str(t)+".pvd")
    #fv=File("Incompressible Viscoelastic Flow Results/Paraview/Velocity_th"+str(th)+"Re="+str(Re)+"We="+str(We)+"b="+str(betav)+"/velocity "+str(t)+".pvd")

    # Time-stepping
    t = 0.0
    iter = 0            # iteration counter
    maxiter = 10000000
    frames = int((Tf/dt)/1000)
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
        dissipation = We*0.5*(phi_def(u1, lambda_d)-1.)*(tau1*Dincomp(u1) + Dincomp(u1)*tau1) 
        diss_vec = as_vector([dissipation[0,0], dissipation[1,0], dissipation[1,1]])
        restau0 = We/dt*(tau1_vec-tau0_vec) + We*F1R_vec + fene_func(tau1, b)*tau1_vec - diss_vec - I_vec #- diss_vec 
        res_test = project(restau0, Zd)
        res_orth = project(restau0-res_test, Zc)                                
        res_orth_norm_sq = project(inner(res_orth,res_orth), Qt)     # Project residual norm onto discontinuous space
        res_orth_norm = np.power(res_orth_norm_sq, 0.5)
        kapp = project(res_orth_norm, Qt)
        kapp = absolute(kapp)
        LPSl_stress = inner(kapp*h*c1*grad(tau),grad(Rt))*dx + inner(kapp*h*c2*div(tau),div(Rt))*dx  # Stress Stabilisation

        # Update SU Term
        alpha_supg = h/(magnitude(u1)+0.000001)
        SU = inner(dot(u1, grad(tau)), alpha_supg*dot(u1,grad(Rt)))*dx
                
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
        DEVSSr_u12 = 2.*(1.-betav)*inner(D0,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS
        DEVSSr_u1 = 2.*(1.-betav)*inner(D0,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS


        U = 0.5*(u + u0)     



        # VELOCITY HALF STEP (ALTERNATIVE)
        """lhsFu12 = Re*(2.0*(u - u0) / dt + conv*dot(u0, nabla_grad(u0)))
        Fu12 = dot(lhsFu12, v)*dx \
               + inner(2.0*betav*Dincomp(u0), Dincomp(v))*dx - ((1.-betav)/(We+DOLFIN_EPS))*inner(div(phi_def(u0, lambda_d)*fene_func(tau0, b)*tau0-Identity(len(u))), (v))*dx  + inner(grad(p0),v)*dx\
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
                        [D12_vec[1], D12_vec[2]]])"""


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
               + inner(fene_sigma(U, p1, tau1, b, lambda_d), Dincomp(v))*dx\
               + dot(p0*n, v)*ds - betav*(dot(Dincomp(U)*n, v)*ds) - ((1.0-betav)/We)*dot(tau0*n, v)*ds\
               + inner(D-Dincomp(u),R)*dx   
              
        a2= lhs(Fus)
        L2= rhs(Fus)""" 



        #Predicted U* Equation (ALTERNATIVE)
        lhsFus = Re*((u - u0)/dt + conv*dot(u0, nabla_grad(U)))
        Fus = dot(lhsFus, v)*dx + \
               + inner(2.0*betav*Dincomp(U), Dincomp(v))*dx - ((1.-betav)/(We+DOLFIN_EPS))*inner( div( (fene_func(tau0, b)*tau0 - Identity(len(u)))*phi_def(u0, lambda_d) ), v )*dx + inner(grad(p0),v)*dx\
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

        a7+= th*DEVSSGl_u1  #[th*DEVSSl_u1]                                                #DEVSS Stabilisation
        L7+= th*DEVSSGr_u1   #[th*DEVSSr_u1] 

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
        lhs_tau1 = (We/dt)*tau + fene_func(tau0, b)*tau +  We*Fdef(u1,tau) - We*0.5*(phi_def(u1, lambda_d)-1.)*(tau*Dincomp(u1) + Dincomp(u1)*tau)            # Left Hand Side 
        rhs_tau1= (We/dt)*tau0  + Identity(len(u)) 

        Astress = inner(lhs_tau1,Rt)*dx - inner(rhs_tau1,Rt)*dx
        a4 = lhs(Astress)
        L4 = rhs(Astress) 

            # SUPG / SU / LPS Stabilisation (User Choose One)

        a4 += LPSl_stress + SU # [SUPGl4, SUl4, LPSl_stab, LPSl_stress, diff_stab, 0]
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
        E_e=assemble((fene_func(tau1, b)*(tau1[0,0]+tau1[1,1])-2.)*dx)

        # Drag Calculations
        
        tau_drag = dot(fene_sigma(u1, p1, tau1, b, lambda_d),Constant((1.0,0.0)))
        drag_correction = 6.*pi*r_a #corr                             # REDO AND REMOVE
        F_drag = assemble(2.*pi*r_a*r_a*inner(n,tau_drag)*SIN_THETA*ds(4))/corr

        # Alternative Formulation
        #tau_drag = (fene_sigma(u1, p1, tau1, b, lambda_d)[0,0]*COS_THETA + fene_sigma(u1, p1, tau1, b, lambda_d)[1,0]*SIN_THETA)*SIN_THETA
        #F_drag = -2.*pi*r_a*r_a*assemble(tau_drag*ds(4))/corr



        #print tau_drag.vector().array().max()     
        #print F_drag       
        # Record Elastic & Kinetic Energy Values (Method 1)
        if j==1:
           x1.append(t)
           ek1.append(E_k)
           ee1.append(E_e)
           fd1.append(0.5*(1.0+tanh(8*(t-0.5)))) # Newtonian Reference line (F_drag)
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

        
        
        # Save Plot to Paraview Folder
        """if j==1 or j==2 or j==5:
            if iter % frames == 0:
               first_normal_stress = project(tau1[0,0]-tau1[1,1], Q)
               fp << p1
               fN1 << first_normal_stress""" 


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
            # Update Stabilisation
            c1 = c1*2.0

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

            # Dirichlet Boundary Conditions  (LID DRIVEN CAVITY)
            in_vel.t = 0.0
            in_stress.t = 0.0
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

    mplot(kapp)
    plt.colorbar()
    plt.savefig("kappa.png")
    plt.clf()
    f_corr = File("Paraview_Results/Stabilisation_Re="+str(Re*conv)+"We="+str(We)+"Ma="+str(Ma)+"b="+str(betav)+"L"+str(b)+"/kappa "+str(t)+".pvd")
    f_corr << kapp
    #quit()

    # Data on Stability Measure
    with open("Drag.txt", "a") as text_file:
         text_file.write("Re="+str(Rey*conv)+", We="+str(We)+"beta"+str(betav)+", L="+str(b)+", lambda="+str(lambda_d)+"beta_s"+str(rat)+", t="+str(t)+", Drag Factor="+str(F_drag)+'\n')

    # Data on pressure min/max 
    with open("Min-Max-Pressure.txt", "a") as text_file:
         text_file.write("Re="+str(Rey*conv)+", We="+str(We)+"lambda="+str(lambda_d)+", p_min="+str(min(p1.vector().get_local()))+", p_max="+str(max(p1.vector().get_local()))+'\n')

    # Data on normal stress difference min/max 
    with open("Min-Max-Stress.txt", "a") as text_file:
         text_file.write("Re="+str(Rey*conv)+" We="+str(We)+" lambda="+str(lambda_d)+" t="+str(t)+", N1_min="+str(min(N1.vector().get_local()))+" N1_max="+str(max(N1.vector().get_local()))+'\n')

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



        #Plot Kinetic and elasic Energies for different Weissenberg numbers at Re=0 (METHOD 2)
    """if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E7 and j==5 or j==1 or j==3 or j==2:
        # Kinetic Energy
        plt.figure(0)
        plt.plot(x1, ek1, 'r-', label=r'$We=0$')
        plt.plot(x2, ek2, 'b-', label=r'$We=0.1$')
        plt.plot(x3, ek3, 'c-', label=r'$We=0.25$')
        plt.plot(x4, ek4, 'm-', label=r'$We=0.5$')
        plt.plot(x5, ek5, 'g-', label=r'$We=1.0$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_k')
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/KineticEnergyTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
        plt.clf()
        # Elastic Energy
        plt.figure(1)
        plt.plot(x1, ee1, 'r-', label=r'$We=0$')
        plt.plot(x2, ee2, 'b-', label=r'$We=0.1$')
        plt.plot(x3, ee3, 'c-', label=r'$We=0.25$')
        plt.plot(x4, ee4, 'm-', label=r'$We=0.5$')
        plt.plot(x5, ee5, 'g-', label=r'$We=1.0$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_e')
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/ElasticEnergyTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
        plt.clf()
        # Drag
        plt.figure(2)
        plt.plot(x1, fd1, 'r-', label=r'$We=0$')
        plt.plot(x2, fd2, 'b-', label=r'$We=0.1$')
        plt.plot(x3, fd3, 'c-', label=r'$We=0.25$')
        plt.plot(x4, fd4, 'm-', label=r'$We=0.5$')
        plt.plot(x5, fd5, 'g-', label=r'$We=1.0$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_e')
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/DragTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
        plt.clf()
        plt.close()"""

    """font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 16}"""



    nt1 = list()
    for x in x1:
        nt1.append(1.0)
    if j==1:
        corr = F_drag

        #Plot Kinetic and elasic Energies for different Dissipative Constants at Re=1 (METHOD 2)
    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E10 and j==loopend or j==1:
        # Kinetic Energy
        plt.figure(0)        
        plt.plot(x2, ek2, 'r-', label=r'$\lambda_D=0$')
        plt.plot(x3, ek3, 'b--', label=r'$\lambda_D=0.1$')
        plt.plot(x4, ek4, 'c:', label=r'$\lambda_D=0.2$')
        #plt.plot(x4, ek4, 'm-', label=r'$\lambda_D=0.15$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('$E_k$')
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/newKineticEnergyRe="+str(Rey*conv)+"We="+str(We)+"lambda="+str(lambda_d)+"b="+str(betav)+"dt="+str(dt)+"beta"+str(rat)+".png")
        plt.clf()
        # Elastic Energy
        plt.figure(1)
        plt.plot(x2, ee2, 'r-', label=r'$\lambda_D=0$')
        plt.plot(x3, ee3, 'b--', label=r'$\lambda_D=0.1$')
        plt.plot(x4, ee4, 'c:', label=r'$\lambda_D=0.2$')
        #plt.plot(x4, ee4, 'm-', label=r'$\lambda_D=0.15$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('$E_e$')
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/newElasticEnergyRe="+str(Rey*conv)+"We="+str(We)+"lambda="+str(lambda_d)+"b="+str(betav)+"dt="+str(dt)+"beta"+str(rat)+".png")
        plt.clf()
        # Drag
        plt.figure(2)
        plt.plot(x1, fd1, 'r--', label=r'Newtonian')
        plt.plot(x2, fd2, 'b-', label=r'$\lambda_D=0$')
        plt.plot(x3, fd3, 'c--', label=r'$\lambda_D=0.1$')
        plt.plot(x4, fd4, 'm:', label=r'$\lambda_D=0.2$')
        #plt.plot(x4, fd4, 'm-', label=r'$\lambda_D=0.15$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('$K/K_N$')
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/newDragRe="+str(Rey*conv)+"We="+str(We)+"lambda="+str(lambda_d)+"b="+str(betav)+"dt="+str(dt)+"beta"+str(rat)+".png")
        plt.clf()
        plt.close()


        #Plot Kinetic and elasic Energies for different Dissipative Constants at Re=1 (METHOD 2)
    """if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E10 and j==loopend or j==1 or j==3 or j==loopend:
        # Kinetic Energy
        plt.figure(0)        
        plt.plot(x2, ek2, 'r-', label=r'$b=10$')
        plt.plot(x3, ek3, 'b-', label=r'$b=20$')
        plt.plot(x4, ek4, 'c-', label=r'$b=50$')
        plt.plot(x5, ek5, 'm-', label=r'$b=100$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('$E_k$')
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/KineticEnergyRe="+str(Rey*conv)+"We="+str(We)+"lambda="+str(lambda_d)+"b="+str(betav)+"dt="+str(dt)+"beta"+str(rat)+".png")
        plt.clf()
        # Elastic Energy
        plt.figure(1)
        plt.plot(x2, ee2, 'r-', label=r'$b=10$')
        plt.plot(x3, ee3, 'b-', label=r'$b=20$')
        plt.plot(x4, ee4, 'c-', label=r'$b=50$')
        plt.plot(x5, ee5, 'm-', label=r'$b=100$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('$E_e$')
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/ElasticEnergyRe="+str(Rey*conv)+"We="+str(We)+"lambda="+str(lambda_d)+"b="+str(betav)+"dt="+str(dt)+"beta"+str(rat)+".png")
        plt.clf()
        # Drag
        plt.figure(2)
        plt.plot(x1, fd1, 'r--', label=r'Newtonian')
        plt.plot(x2, fd2, 'b-', label=r'$b=10$')
        plt.plot(x3, fd3, 'c-', label=r'$b=20$')
        plt.plot(x4, fd4, 'm-', label=r'$b=50$')
        plt.plot(x5, fd5, 'm-', label=r'$b=100$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('$K/K_N$')
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/DragRe="+str(Rey*conv)+"We="+str(We)+"lambda="+str(lambda_d)+"b="+str(betav)+"dt="+str(dt)+"beta"+str(rat)+".png")
        plt.clf()
        plt.close()"""
        

        #Plot Kinetic and elasic Energies for different Reynolds numbers at We = 0.5 (METHOD 2)
    """if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E5 and j==5 or j==1 or j==3:
        # Kinetic Energy
        plt.figure(0)
        plt.plot(x1, ek1, 'r-', label=r'$Re=0.1$')
        plt.plot(x2, ek2, 'b-', label=r'$Re=0.5$')
        plt.plot(x3, ek3, 'c-', label=r'$Re=1.0$')
        plt.plot(x4, ek4, 'm-', label=r'$Re=2.0$')
        plt.plot(x5, ek5, 'g-', label=r'$Re=5.0$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_k')
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/KineticEnergyTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
        plt.clf()
        # Elastic Energy
        plt.figure(1)
        plt.plot(x1, ee1, 'r-', label=r'$Re=0.1$')
        plt.plot(x2, ee2, 'b-', label=r'$Re=0.5$')
        plt.plot(x3, ee3, 'c-', label=r'$Re=1.0$')
        plt.plot(x4, ee4, 'm-', label=r'$Re=2.0$')
        plt.plot(x5, ee5, 'g-', label=r'$Re=5.0$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_e')
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/ElasticEnergyTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
        plt.clf()
        # Drag
        plt.figure(2)
        plt.plot(x1, fd1, 'r-', label=r'$Re=0.1$')
        plt.plot(x2, fd2, 'b-', label=r'$Re=0.5$')
        plt.plot(x3, fd3, 'c-', label=r'$Re=1.0$')
        plt.plot(x4, fd4, 'm-', label=r'$Re=2.0$')
        plt.plot(x5, fd5, 'g-', label=r'$Re=5.0$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_e')
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/newDragTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
        plt.clf()
        plt.close()"""


        # Comparing Kinetic & Elastic Energies for different Stablisation parameters
    """if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E5 and abs(E_k) < 10 and j==3 or j==1:
        # Kinetic Energy
        plt.figure(0)
        plt.plot(x1, ek1, 'r-', label=r'$\theta=0$')
        plt.plot(x2, ek2, 'b-', label=r'$\theta=(1-\beta)/10$')
        plt.plot(x3, ek3, 'c-', label=r'$\theta=1-\beta$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_k')
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/newKineticEnergyTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"We="+str(We)+"dt="+str(dt)+".png")
        plt.clf()
        # Elastic Energy
        plt.figure(1)
        plt.plot(x1, ee1, 'r-', label=r'$\theta=0$')
        plt.plot(x2, ee2, 'b-', label=r'$\theta=(1-\beta)/10$')
        plt.plot(x3, ee3, 'c-', label=r'$\theta=\beta$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_e')
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/newElasticEnergyTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"We="+str(We)+"dt="+str(dt)+".png")
        plt.clf()"""





    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E10 and j==1 or j==loopend:

        fv = File("Paraview_Results/Velocity Results Re="+str(Re*conv)+"We="+str(We)+"Ma="+str(Ma)+"b="+str(betav)+"L"+str(b)+"/velocity "+str(t)+".pvd")
        fv_x = File("Paraview_Results/Velocity Results Re="+str(Re*conv)+"We="+str(We)+"Ma="+str(Ma)+"b="+str(betav)+"L"+str(b)+"/u_x "+str(t)+".pvd")
        fv_y = File("Paraview_Results/Velocity Results Re="+str(Re*conv)+"We="+str(We)+"Ma="+str(Ma)+"b="+str(betav)+"L"+str(b)+"/u_y "+str(t)+".pvd")
        fmom = File("Paraview_Results/Velocity Results Re="+str(Re*conv)+"We="+str(We)+"Ma="+str(Ma)+"b="+str(betav)+"L"+str(b)+"/mom "+str(t)+".pvd")
        fp = File("Paraview_Results/Pressure Results Re="+str(Re*conv)+"We="+str(We)+"Ma="+str(Ma)+"b="+str(betav)+"L"+str(b)+"/pressure "+str(t)+".pvd")
        ftau_xx = File("Paraview_Results/Stress Results Re="+str(Re*conv)+"We="+str(We)+"Ma="+str(Ma)+"b="+str(betav)+"L"+str(b)+"/tua_xx "+str(t)+"lambda_d"+str(lambda_d)+".pvd")
        ftau_xy = File("Paraview_Results/Stress Results Re="+str(Re*conv)+"We="+str(We)+"Ma="+str(Ma)+"b="+str(betav)+"L"+str(b)+"/tau_xy "+str(t)+"lambda_d"+str(lambda_d)+".pvd")
        ftau_yy = File("Paraview_Results/Stress Results Re="+str(Re*conv)+"We="+str(We)+"Ma="+str(Ma)+"b="+str(betav)+"L"+str(b)+"/tau_yy "+str(t)+"lambda_d"+str(lambda_d)+".pvd")
        f_N1 = File("Paraview_Results/Stress Results Re="+str(Re*conv)+"We="+str(We)+"Ma="+str(Ma)+"b="+str(betav)+"L"+str(b)+"/N1"+str(t)+"lambda_d"+str(lambda_d)+".pvd")

        # Plot Stress/Normal Stress Difference
        tau_xx=project(tau1[0,0],Q)
        ftau_xx << tau_xx
        mplot(tau_xx)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/newtau_xxRe="+str(Rey*conv)+"We="+str(We)+"lambda="+str(lambda_d)+"b="+str(betav)+"t="+str(t)+"beta"+str(rat)+".png")
        plt.clf() 
        tau_xy=project(tau1[1,0],Q)
        ftau_xy << tau_xy
        mplot(tau_xy)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/newtau_xyRe="+str(Rey*conv)+"We="+str(We)+"lambda="+str(lambda_d)+"b="+str(betav)+"t="+str(t)+"beta"+str(rat)+".png")
        plt.clf() 
        tau_yy=project(tau1[1,1],Q)
        ftau_yy << tau_yy
        mplot(tau_yy)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/newtau_yyRe="+str(Rey*conv)+"lambda="+str(lambda_d)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"beta"+str(rat)+".png")
        plt.clf() 
        N1=project(tau1[0,0]-tau1[1,1],Q)
        mplot(N1)
        f_N1 << N1
        #plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/newFirstNormalStressDifferenceRe="+str(Rey*conv)+"lambda="+str(lambda_d)+"We="+str(We)+\
                    "beta"+str(rat)+"b="+str(betav)+"t="+str(t)+".png")
        plt.clf()

 
       # Plot Velocity Components
        ux=project(u1[0],Q)
        fv_x << ux
        mplot(ux)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/newu_xRe="+str(Rey)+"We="+str(We)+"lambda="+str(lambda_d)+"b="+str(betav)+"t="+str(t)+".tif")
        plt.clf()
        uy=project(u1[1],Q)
        fv_y << uy
        mplot(uy)
        #plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/newu_yRe="+str(Rey)+"We="+str(We)+"lambda="+str(lambda_d)+"b="+str(betav)+"t="+str(t)+".tif")
        plt.clf()
        fp << p1
        mplot(p1)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/newPressureRe="+str(Re*conv)+"We="+str(We)+"lambda="+str(lambda_d)+"b="+str(betav)+"t="+str(t)+".tif")
        plt.clf()



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


        #plt.contour(XX, YY, pp, 25)
        #plt.title('Pressure Contours')   # PRESSURE CONTOUR PLOT
        #plt.colorbar() 
        #plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/newPressureContoursRe="+str(Re*conv)+"We="+str(We)+"lambda="+str(lambda_d)+"b="+str(betav)+"t="+str(t)+".png")
        #plt.clf()


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
                       density=1,              
                       color=speed,  
                       cmap=cm.gnuplot,                         # colour map
                       linewidth=0.8)                           # line thickness
                                                                # arrow size
        plt.colorbar()                                          # add colour bar on the right
        plt.title('Flow Past a Sphere')
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots-Contours/newVelocityContoursRe="+str(Re*conv)+"We="+str(We)+"lambda="+str(lambda_d)+"b="+str(betav)+"t="+str(t)+".png")   
        plt.clf()                                             # display the plot


    plt.close()

    if dt < tol:
       j=loopend+1
       break

    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E7:
        Tf=T_f 

    if jjj==3:
        quit()
    
    if j==loopend:
        jjj+=1
        j=0
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


    if jj == 0: 
        # Calculate Stress Residual 
        F1R = Fdef(u1, tau1)  
        F1R_vec = as_vector([F1R[0,0], F1R[1,0], F1R[1,1]])
        restau0 =  We*F1R_vec + tau1_vec - I_vec + We/dt*(tau1_vec-tau0_vec) 
        res_test = inner(restau0,restau0)                            

        kapp = project(res_test, Qt) # Error Function
        norm_kapp = normalize_solution(kapp) # normalised error function

        ratio = 0.25/(1*err_count + 1.0) # Proportion of cells that we want to refine
        tau_average = project((tau1_vec[0]+tau1_vec[1]+tau1_vec[2])/3.0 , Qt)
        error_rat = project(kapp/(tau_average + 0.000001) , Qt)
        error_rat = absolute(error_rat)

        jj=1 

        if error_rat.vector().get_local().max() > 0.01 and err_count < 2:
           err_count+=1
           mesh = adaptive_refinement(mesh, norm_kapp, ratio)
           mplot(norm_kapp)
           plt.colorbar()
           plt.savefig("adaptive-error-function.eps")
           plt.clf()
           mplot(mesh)
           plt.savefig("adaptive-mesh.eps")
           plt.clf()
           jj=0
           conv_failure = 0
            
        j = 0
        dt = 5*mesh.hmin()**2
        Tf = T_f
        th = 0.5
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
        tau_zz_list = list()
        tau_zr_list = list()
        tau_rr_list = list()


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

    # Dirichlet Boundary Conditions  (LID DRIVEN CAVITY)
    in_vel.t = 0.0
    in_stress.t = 0.0
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
 

