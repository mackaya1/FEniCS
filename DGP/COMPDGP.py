"""COMPRESSIBLE AIRFLOW (Double Glazing) PROBLEM"""
"""Inompressible Double Glazing Problem """

from DGP_base import *  # Import Base Code for LDC Problem

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


# Experiment Run Time
dt = 0.001  #Timestep

T_f = 0.5
Tf = T_f 
tol = 0.0001

tol = 0.0001
U = 1
betav = 0.99    
Ra = 10000                           #Rayleigh Number
Pr = 1
We = 0.01                          #Weisenberg NUmber
Vh = 0.005
T_0 = 300
T_h = 350
Bi = 0.0
Ma = 0.001
al = 0.01

c1 = 0.05
c2 = 0.001
th = 0.5              # DEVSS
C = 200.


# Loop Experiments
loopend = 1
j=0
j_2=0
jj=0
j=0

while j < loopend:
    j+=1
    t=0.0
    


    # Comparing different WEISSENBERG Numbers (We=0.1,0.2,0.3,0.4,0.5) at Re=__
    #betav=0.5
    if j==1:
       We=0.1
    elif j==2:
       We=0.25
    elif j==3:
       We=0.5
    elif j==4:
       We=1.0
    elif j==5:
       We=2.0

    print '############# TIME SCALE ############'
    print 'Timestep size (s):', dt
    print 'Finish Time (s):', Tf

    print '############# Scalings & Nondimensional Parameters ############'
    print 'Characteristic Length (m):', L
    print 'Characteristic Velocity (m/s):', 1.0
    print 'Rayleigh Number:', Ra
    print 'Prandtl Number:', Pr
    print 'Weissenberg Number:', We
    print 'Viscosity Ratio:', betav
    print 'Diffusion Number:' ,Di
    print 'Viscous Heating Number:', Vh

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

    #quit()

    phi0 = project((T0+C)/(T_0+C)*(T0/T_0)**(3/2),Q)


    # Initial Conformation Tensor
    I_vec = Expression(('1.0','0.0','1.0'), degree=2)
    initial_guess_conform = project(I_vec, Zc)
    assign(tau0_vec, initial_guess_conform)         # Initial guess for conformation tensor is Identity matrix


    # Initial Density Field
    rho_initial = Expression('1.0', degree=1)
    rho_initial_guess = project(1.0, Q)
    rho0.assign(rho_initial_guess)


    # Initial Temperature Field
    T_initial_guess = project(T_0, Q)
    T0.assign(T_initial_guess)





    #Define Variable Parameters, Strain Rate and other tensors
    thetal = (T)/(T_h-T_0)
    thetar = (T_0)/(T_h-T_0)
    thetar = project(thetar,Q)
    theta0 = (T0-T_0)/(T_h-T_0)
    theta1 = (T1-T_0)/(T_h-T_0)

    #Redefine C0 as a temperature dependent Variable


    # TAYLOR GALERKIN METHOD (COMPRESSIBLE VISCOELASTIC)

    # Weak Formulation 



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



    # FEM Solution Convergence Plot
    x=list()
    y=list()
    z=list()
    zz=list()


    # Set up Krylov Solver 

    # Use amg preconditioner if available
    prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

    # Use nonzero guesses - essential for CG with non-symmetric BC
    parameters['krylov_solver']['nonzero_initial_guess'] = True
    parameters['krylov_solver']['monitor_convergence'] = False
    
    solveru = KrylovSolver("bicgstab", "default")
    solvertau = KrylovSolver("bicgstab", "default")
    solverp = KrylovSolver("bicgstab", "default")


     # Time-stepping
    t = 0.0
    iter = 0            # iteration counter
    maxiter = 100000
    while t < Tf + DOLFIN_EPS and iter < maxiter:
        #tic = time.time()
        iter += 1
        print"t = %s,  Iteration = %d, Convergence Failures = %s, Loop = %s" %(t, iter, jj, j)

        # Set Function timestep
        ramped_T.t = t
        #Wet.t = t

        # Update Stabilisation (Copy and Paste Stabilisation Technique from above)
        F1R = Fdefcom(u1, tau1)  #Compute the residual in the STRESS EQUATION
        F1R_vec = as_vector([F1R[0,0], F1R[1,0], F1R[1,1]])
        Dcomp1_vec = as_vector([Dcomp(u1)[0,0], Dcomp(u1)[1,0], Dcomp(u1)[1,1]])
        restau0 = We/dt*(tau1_vec-tau0_vec) + We*F1R_vec + tau1_vec - I_vec
        res_test = project(restau0, Zd)
        res_orth = project(restau0-res_test, Zc)                                
        res_orth_norm_sq = project(inner(res_orth,res_orth), Qt)     # Project residual norm onto discontinuous space
        res_orth_norm = np.power(res_orth_norm_sq, 0.5)
        kapp = project(res_orth_norm, Qt)
        LPSl_stress = inner(kapp*h*c1*grad(tau),grad(Rt))*dx + inner(kapp*h*c2*div(tau),div(Rt))*dx  # Stress Stabilisation

        U12 = 0.5*(u1 + u0)  
  
        # Update Solutions
        if iter > 1:
            w0.assign(w1)
            T0.assign(T1)
            rho0.assign(rho1)
            p0.assign(p1)
            tau0_vec.assign(tau1_vec)

        # Use previous solutions to update nonisthermal paramters
        phi0 = project((T0+C)/(T_0+C)*(T0/T_0)**(3/2),Q)   
        theta0 = (T0-T_0)/(T_h-T_0)            
        therm = (1.0+al*theta0)
        therm_inv = project(1.0/therm, Q)
        c0 = c0*(1.0+al*T0/T_0) 
        theta1 = (T1-T_0)/(T_h-T_0)

        (u0, D0_vec) = w0.split()

        D0 = as_matrix([[D0_vec[0], D0_vec[1]],
                        [D0_vec[1], D0_vec[2]]])                    #DEVSS STABILISATION
        DEVSSr_u12 = 2*(1-betav)*inner(D0,Dincomp(v))*dx            # Update DEVSS Stabilisation RHS

        #print therm_inv.vector().get_local().min()

      

        U = 0.5*(u + u0)              
        # VELOCITY HALF STEP
        Du12Dt = rho0*(2.0*(u - u0) / dt + dot(u0, nabla_grad(u0)))
        Fu12 = dot(Du12Dt, v)*dx + \
               + inner(sigmacom(U, p0, tau0), Dincomp(v))*dx + Ra*Pr*inner(rho0*f,v)*dx \
               + dot(p0*n, v)*ds - betav*Pr*(dot(nabla_grad(U)*n, v)*ds + (1.0/3)*dot(div(U)*n,v)*ds)\
               - (Pr*(1-betav)/We)*dot(tau0*n, v)*ds\
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

        """# STRESS Half Step
        F12 = dot(u12,grad(tau)) - dot(grad(u12),tau) - dot(tau,tgrad(u12)) # Convection/Deformation Terms
        lhs_tau12 = (We/dt+1.0)*tau + We*F12                             # Left Hand Side
        rhs_tau12= (We/dt)*tau0 + 2.0*(1.0-betav)*Dincomp(u0)                     # Right Hand Side

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
        lhsFus = rho0*((u - u0)/dt + dot(u12, nabla_grad(U)))
        Fus = dot(lhsFus, v)*dx + \
              + inner(sigmacom(U, p0, tau0), Dincomp(v))*dx + Ra*Pr*inner(rho0*f,v)*dx \
              + dot(p0*n, v)*ds - betav*Pr*(dot(nabla_grad(U)*n, v)*ds + (1.0/3)*dot(div(U)*n,v)*ds) \
              - (Pr*(1.0-betav)/We)*dot(tau0*n, v)*ds\
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
        
        #Continuity Equation 1
        
        lhs_p_1 = (Ma*Ma/(dt))*p
        rhs_p_1 = (Ma*Ma/(dt))*p0 - therm*dot(grad(rho0),us) - therm*rho0*div(us)

        lhs_p_2 = therm*0.5*dt*grad(p)
        rhs_p_2 = therm*0.5*dt*grad(p0)
        
        a5=inner(lhs_p_1,q)*dx + inner(lhs_p_2,grad(q))*dx   
        L5=inner(rhs_p_1,q)*dx + inner(rhs_p_2,grad(q))*dx

        A5 = assemble(a5)
        b5 = assemble(L5)
        [bc.apply(A5, b5) for bc in bcp]
        #[bc.apply(p1.vector()) for bc in bcp]
        solve(A5, p1.vector(), b5, "bicgstab", "default")
        end()

        #Continuity Equation 2
        rho1 = rho0 + (Ma*Ma)*therm_inv*(p1-p0)
        rho1 = project(rho1,Q)
        #A6 = assemble(a6)
        #b6 = assemble(L6)
        #[bc.apply(A6, b6) for bc in bcp]
        #solve(A6, rho1.vector(), b6, "bicgstab", "default")
        #end()




        #Velocity Update

        lhs_u1 = (1./dt)*rho1*u                                          # Left Hand Side
        rhs_u1 = (1./dt)*rho0*us                                         # Right Hand Side

        a7=inner(lhs_u1,v)*dx +inner(D-Dincomp(u),R)*dx  # Weak Form
        L7=inner(rhs_u1,v)*dx + 0.5*(inner(p1-p0,div(v))*dx + dot(p1*n, v)*ds) 

        a7+= 0                      #DEVSS Stabilisation
        L7+= 0 

        A7 = assemble(a7)
        b7 = assemble(L7)
        [bc.apply(A7, b7) for bc in bcu]
        solve(A7, w1.vector(), b7, "bicgstab", "default")
        end()

        (u1, D1_vec) = w1.split()


        # Stress Full Step
        lhs_tau1 = ((We/dt)+1.0)*tau  +  We*Fdefcom(u1,tau)                            # Left Hand Side
        rhs_tau1= (We/dt)*tau0 + Identity(len(u)) 

        A = inner(lhs_tau1,Rt)*dx - inner(rhs_tau1,Rt)*dx
        a4 = lhs(A)
        L4 = rhs(A) 

            # SUPG / SU / LPS Stabilisation (User Choose One)

        a4 += LPSl_stress  # [SUPGl4, SUl4, LPSl_stab, LPSl_stress, diff_stab, 0]
        L4 += 0  # [SUPGr4, SUr4, LPSr_stab, LPS_res_stab, 0]   


        A4=assemble(a4)                                     # Assemble System
        b4=assemble(L4)
        #[bc.apply(A4, b4) for bc in bctau]
        #solvertau.solve(A4, tau1_vec.vector(), b4)
        #end()

        # Temperature Update (FIRST ORDER)
        gamdot = inner(sigmacom(u1, p1, tau1), grad(u1))
        lhs_theta1 = (1.0/dt)*rho0*thetal + rho0*dot(u1,grad(thetal))
        rhs_theta1 = (1.0/dt)*rho0*thetar + rho0*dot(u1,grad(thetar)) + (1.0/dt)*rho0*theta0 + Vh*gamdot
        a8 = inner(lhs_theta1,r)*dx + inner(grad(thetal),grad(r))*dx 
        L8 = inner(rhs_theta1,r)*dx + inner(grad(thetar),grad(r))*dx #+ Bi*inner(grad(theta0),n*r)*ds(3) 

        A8=assemble(a8)                                     # Assemble System
        b8=assemble(L8)
        [bc.apply(A8, b8) for bc in bcT]
        solve(A8, T1.vector(), b8, "bicgstab", "default")
        end()


        # Energy Calculations
        E_k=assemble(0.5*rho1*dot(u1,u1)*dx)
        E_e=assemble((tau1[0,0]+tau1[1,1]-2.0)*dx)

        
        
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

       
        t += dt
        #elapsed = time.time() - tic
        #print elapsed

    # Minimum of stream function (Eye of Rotation)
    u1 = project(u1, V)
    psi = comp_stream_function(rho1, u1)
    psi_min = min(psi.vector().get_local())
    min_loc = min_location(psi)
    with open("Stream-Function.txt", "a") as text_file:
         text_file.write("Ra="+str(Ra)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+"----- psi_min="+str(psi_min)+"---"+str(min_loc)+'\n')

    # Data on Kinetic/Elastic Energies
    with open("ConformEnergy.txt", "a") as text_file:
         text_file.write("Ra="+str(Ra)+", We="+str(We)+", Ma="+str(Ma)+", t="+str(t)+", E_k="+str(E_k)+", E_e="+str(E_e)+'\n')



    if j==3:
        peakEk1 = max(ek1)
        peakEk2 = max(ek2)
        peakEk3 = max(ek3)
        with open("ConformEnergy.txt", "a") as text_file:
             text_file.write("Ra="+str(Ra)+", We="+str(We)+", Ma="+str(Ma)+"-------Peak Kinetic Energy: "+str(peakEk3)+"Incomp Kinetic En"+str(peakEk1)+'\n')

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
        plt.savefig("Compressible Viscoelastic Flow Results/Stability-Convergence/Mesh_KineticEnergyRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
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
        plt.savefig("Incompressible Viscoelastic Flow Results/Stability-Convergence/Mesh_ElasticEnergyRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
        plt.clf()"""

        #Plot Kinetic and elasic Energies for different REYNOLDS numbers at constant Weissenberg Number    
    """if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E6 and j==5:
        # Kinetic Energy
        plt.figure(0)
        plt.plot(x1, ek1, 'r-', label=r'$Ra=0$')
        plt.plot(x2, ek2, 'b-', label=r'$Ra=5$')
        plt.plot(x3, ek3, 'c-', label=r'$Ra=10$')
        plt.plot(x4, ek4, 'm-', label=r'$Ra=25$')
        plt.plot(x5, ek5, 'g-', label=r'$Ra=50$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_k')
        plt.savefig("Compressible Viscoelastic Flow Results/Energy/Fixed_We_KineticEnergyRa="+str(Ra+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
        # Elastic Energy
        plt.figure(1)
        plt.plot(x1, ee1, 'r-', label=r'$Ra=0$')
        plt.plot(x2, ee2, 'b-', label=r'$Ra=5$')
        plt.plot(x3, ee3, 'c-', label=r'$Ra=10$')
        plt.plot(x4, ee4, 'm-', label=r'$Ra=25$')
        plt.plot(x5, ee5, 'g-', label=r'$Ra=50$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_e')
        plt.savefig("Compressible Viscoelastic Flow Results/Energy/Fixed_We_ElasticEnergyRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"dt="+str(dt)+".png")
        plt.clf()"""

    plt.close()

        #Plot Kinetic and elasic Energies for different Weissenberg numbers at Re=0 (METHOD 2)
    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E6 and j==5 or j==1 or j==3:
        # Kinetic Energy
        plt.figure(0)
        plt.plot(x1, ek1, 'r-', label=r'$We=0.1$')
        plt.plot(x2, ek2, 'b-', label=r'$We=0.25$')
        plt.plot(x3, ek3, 'c-', label=r'$We=0.5$')
        plt.plot(x4, ek4, 'm-', label=r'$We=1.0$')
        plt.plot(x5, ek5, 'g-', label=r'$We=2.0$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_k')
        plt.savefig("Compressible Viscoelastic Flow Results/Energy/KineticEnergyTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
        plt.clf()
        # Elastic Energy
        plt.figure(1)
        plt.plot(x1, ee1, 'r-', label=r'$We=0.1$')
        plt.plot(x2, ee2, 'b-', label=r'$We=0.25$')
        plt.plot(x3, ee3, 'c-', label=r'$We=0.5$')
        plt.plot(x4, ee4, 'm-', label=r'$We=1.0$')
        plt.plot(x5, ee5, 'g-', label=r'$We=2.0$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_e')
        plt.savefig("Compressible Viscoelastic Flow Results/Energy/ElasticEnergyTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
        plt.clf()


    """if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E6 and j==1:
        # Kinetic Energy
        plt.figure(0)
        plt.plot(x1, ek1, 'r-', label=r'$We=0$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_k')
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/KineticEnergyTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
        plt.clf()
        # Elastic Energy
        plt.figure(1)
        plt.plot(x1, ee1, 'r-', label=r'$We=0$')
        plt.xlabel('time(s)')
        plt.ylabel('E_e')
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/ElasticEnergyTf="+str(Tf)+"Ra="+str(Ra)+"b="+str(betav)+"mesh="+str(mm)+"dt="+str(dt)+".png")
        plt.clf()"""

    plt.close()


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
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/KineticEnergyTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"We="+str(We)+"dt="+str(dt)+".png")
        plt.clf()
        # Elastic Energy
        plt.figure(1)
        plt.plot(x1, ee1, 'r-', label=r'$\theta=0$')
        plt.plot(x2, ee2, 'b-', label=r'$\theta=(1-\beta)/10$')
        plt.plot(x3, ee3, 'c-', label=r'$\theta=\beta$')
        plt.legend(loc='best')
        plt.xlabel('time(s)')
        plt.ylabel('E_e')
        plt.savefig("Incompressible Viscoelastic Flow Results/Energy/ElasticEnergyTf="+str(Tf)+"Re"+str(Rey*conv)+"b="+str(betav)+"We="+str(We)+"dt="+str(dt)+".png")
        plt.clf()"""


    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E6 and abs(E_k) < 10 and j==1 or j==5:

        # Plot Stress/Normal Stress Difference
        tau_xx=project(tau1[0,0],Q)
        mplot(tau_xx)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/tau_xxRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(th)+".png")
        plt.clf() 
        tau_xy=project(tau1[1,0],Q)
        mplot(tau_xy)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/tau_xyRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(th)+".png")
        plt.clf() 
        tau_yy=project(tau1[1,1],Q)
        mplot(tau_yy)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/tau_yyRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(th)+".png")
        plt.clf() 
        theta0 = project(theta0, Q)
        mplot(theta0)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/TemperatureRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
        plt.clf()

    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E6 and abs(E_k) < 10 and j==1 or j==5:
 
       # Plot Velocity Components
        ux=project(u1[0],Q)
        mplot(ux)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/u_xRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(c1)+".png")
        plt.clf()
        uy=project(u1[1],Q)
        mplot(uy)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/u_yRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+"Stabilisation"+str(c1)+".png")
        plt.clf()

    if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==5 or j==1:
        # Matlab Plot of the Solution at t=Tf
        rho1=project(rho1,Q)
        mplot(rho1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/DensityRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf() 
        mplot(p1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/PressureRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()
        mplot(T1)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/TemperatureRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()
        psi = stream_function(u1)
        mplot(psi)
        plt.colorbar()
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/stream_functionRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"Ma="+str(Ma)+"t="+str(t)+".png")
        plt.clf()

    if max(norm(tau1_vec.vector(),'linf'),norm(w1.vector(), 'linf')) < 10E6 and j==5 or j==1:
       #Plot Contours USING MATPLOTLIB
        # Scalar Function code


        x = Expression('x[0]', degree=2)     #GET X-COORDINATES LIST
        y = Expression('x[1]', degree=2)     #GET Y-COORDINATES LIST
        pvals = p1.vector().get_local()          # GET SOLUTION p= p(x,y) list
        Tvals = theta0.vector().get_local() 
        psiq = project(psi, Q)
        psivals = psiq.vector().get_local() 
        tauxx = project(tau1_vec[0], Q)
        tauxxvals = tauxx.vector().get_local()
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
        pp = mlab.griddata(xvals, yvals, pvals, xx, yy, interp='nn') 
        TT = mlab.griddata(xvals, yvals, Tvals, xx, yy, interp='nn')
        psps = mlab.griddata(xvals, yvals, psivals, xx, yy, interp='nn')  


        plt.contour(XX, YY, pp, 25)
        plt.title('Pressure Contours')   # PRESSURE CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/PressureContoursRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
        plt.clf()


        plt.contour(XX, YY, TT, 20)
        plt.title('Temperature Contours')   # PRESSURE CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/TemperatureContoursRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
        plt.clf()

        plt.contour(XX, YY, psps, 15)
        plt.title('Streamline Contours')   # PRESSURE CONTOUR PLOT
        plt.colorbar() 
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/StreamlineContoursRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")
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
        plt.title('Double Glazing Problem')
        plt.savefig("Compressible Viscoelastic Flow Results/Plots-Contours/VelocityContoursRa="+str(Ra)+"We="+str(We)+"b="+str(betav)+"t="+str(t)+".png")   
        plt.clf()                                           # display the plot


    plt.close()



    if dt < tol:
       j=loopend+1
       break

    if max(norm(w1.vector(),'linf'),norm(p1.vector(), 'linf')) < 10E6 and abs(E_k) < 10:
        Tf=T_f    

    if j==5:
        quit()
        j_2+=1
        Ra=2*Ra
        T_f = 5
        Tf = T_f
        j=0
        jj=0
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

    #Reset Functions
    rho0 = Function(Q)
    p0=Function(Q)       # Pressure Field t=t^n
    p1=Function(Q)       # Pressure Field t=t^n+1
    T0=Function(Q)       # Temperature Field t=t^n
    T1=Function(Q)       # Temperature Field t=t^n+1

    tau0_vec=Function(Zc)     # Stress Field (Vector) t=t^n
    tau12_vec=Function(Zc)    # Stress Field (Vector) t=t^n+1/2
    tau1_vec=Function(Zc)     # Stress Field (Vector) t=t^n+1

    w0= Function(W)
    w12= Function(W)
    ws= Function(W)
    w1= Function(W)

    (u0, D0_vec) = w0.split()
    (u12, D12_vec) = w12.split()
    (u1, D1_vec) = w1.split()
    (us, Ds_vec) = ws.split()
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

        



