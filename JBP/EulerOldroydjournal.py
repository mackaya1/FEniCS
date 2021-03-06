
""" Journal Bearing Lubrication"""
"""Solution to the Momentum and Energy equation for an Oldroyd-B Fluid """
"""Numerical Scheme: ADAPTED CHORINS PROJECTION METHOD"""
"""Stabilisation: Not Used """


from decimal import *
from dolfin import *
from mshr import *
from math import pi, sin, cos, sqrt
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.tri as tri
import matplotlib.mlab as mlab


#Set Decimal precision
#decimal.getcontext(  ).prec = 10

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
x1=0.0
y1=0.0
x2=0.015
y2=0.0

c0=Circle(Point(0.0,0.0), r_a, 600)
c1=Circle(Point(x1,y1), r_a, 600)
c2=Circle(Point(x2,y2), r_b, 600)

c3=Circle(Point(x1,y1), 0.99*r_a, 256)

ex=x2-x1
ey=y2-y1
ec=np.sqrt(ex**2+ey**2)
c=r_b-r_a

if c <= 0.0:
   print("ERROR! Journal radius greater than bearing radius")
   quit()

# Create mesh
cyl0=c2-c0
cyl=c2-c1

mesh0 = generate_mesh(cyl0, 15) #Initial Meshing (FENICS Mesh generator)
mesh = generate_mesh(cyl, 25)
meshc= generate_mesh(c3, 25)

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

#plot(cell_domains, interactive=False)
#plot(mesh, interactive=False)



#print('Number of Cells:', mesh.num_cells())
#print('Number of Vertices:', mesh.num_vertices())

#Define Boundaries 
                                                                          
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
#plot(mesh0, interactive=False)
#plot(mesh1, interactive=False)
plot(sub_domains, interactive=False)


# Define function spaces (P2-P1)
d=2
V = VectorFunctionSpace(mesh, "CG", d)
Q = FunctionSpace(mesh, "CG", d)
W = TensorFunctionSpace(mesh, "CG", d)


dt = 0.001  #Time Stepping  

w_j = 300.0 

loopend=9
j=0
jj=0
tol=10E-6
while j < loopend:
    j+=1

    # Define trial and test functions
    u = TrialFunction(V)
    p = TrialFunction(Q)
    T = TrialFunction(Q)
    S = TrialFunction(W)
    R = TestFunction(W)
    v = TestFunction(V)
    q = TestFunction(Q)

    # Set parameter values
    hm = mesh.hmin()
    hM = mesh.hmax()
    #dt = 0.005                          # Time Stepping [s]
    Tf = 3.0                             # Stopping Time [s] 
    rho = 1000.0                         # Density [kg/m^3]
    cp = 4000.0                          # Heat Capacity [J/(g.K)]
    mu1 = (49.99/1.0)*3*10E-2             # Solvent Viscosity [Pa.s]
    mu2 = (0.01/1.0)*3*10E-2              # Polymeric Viscosity [Pa.s]
    lambda1 = (1.0/3)*10E-4              # Relaxation Time [s]
    #w_j = 100.0                         # Angular VELOCITY (inner bearing) [rad/s]
    kappa = 2.0                          # Thermal condusctivity  
    heatt= 2.0                           # Heat Transfer Coefficient
    mu = mu1+mu2                         # Total Viscosity 
    beta = mu1/mu                        # Viscosity Ratio (Viscoelastic Fluid)

    # Define coefficients 
    k = Constant(dt)
    alpha = 1.0/(rho*cp)
    ecc = (ec)/(c)

    #Nondimensional Parameters

    Re = rho*w_j*r_a*c/mu     #Reynolds Number 
    We = lambda1*w_j*r_a/c    #Weissenberg Number 
    De = lambda1*w_j          #Deborah Number 
    tstar = r_a*w_j/c         #Nondimensional Time
    eta = We/k
    #Print Parameter Values

    print'Minimum Cell Diameter:', hm
    print'Eccentricity:' ,ec
    print'Radius DIfference:',c
    print'Eccentricity Ratio:',ecc
    print'Reynolds Number:',Re
    print'Weissenberg Number:',We
    print'Viscosity Ratio:', beta
    print'Nondimensional Second:',tstar
    #quit()




    #Define inner cylinder velocity (TIME DEPENDENT)
    td= Constant('5')
    e = Constant('6')
    w = Expression(('(0.5*(1.0+tanh(e*t-2.5)))*(x[1]-y1)/r_a' , '-(0.5*(1.0+tanh(e*t-2.5)))*(x[0]-x1)/r_a' ), d=d, degree=d, w_j=w_j , r_a=r_a, x1=x1, y1=y1, e=e , t=0.0)

    # Define unit Normal/tangent Vector at inner and outer Boundary (Method 2)
    n0 =  Expression(('(x[0]-x1)/r_a' , '(x[1]-y1)/r_a' ), d=d, degree=d, r_a=r_a, x1=x1, y1=y1)
    n1 =  Expression(('(x[0]-x2)/r_b' , '(x[1]-y2)/r_b' ), d=d, degree=d, r_b=r_b, x2=x2, y2=y2)
    t0 =  Expression(('(x[1]-y1)/r_a' , '-(x[0]-x1)/r_a' ), d=d, degree=d, r_a=r_a, x1=x1, y1=y1)
    t1 =  Expression(('(x[1]-y2)/r_b' , '-(x[0]-x2)/r_b' ), d=d, degree=d, r_b=r_b, x2=x2, y2=y2)

    # Define boundary conditions
    spin =  DirichletBC(V, w, omega0)  #The inner cylinder will be rotated with constant angular velocity w_a
    noslip  = DirichletBC(V, (0.0, 0.0), omega1) #The outer cylinder remains fixed with zero velocity 
    temp0 =  DirichletBC(Q, 0.0, omega0)    #Dirichlet Boundary Condition on the inner bearing 


    #Collect Boundary Conditions
    bcu = [spin, noslip]
    bcp = []
    bcT = [temp0]
    bcS = []

    # Create functions
    u0 = Function(V)
    us = Function(V)
    u1 = Function(V)
    p1 = Function(Q)
    T0 = Function(Q)
    T1 = Function(Q)
    S0 = Function(W)
    S1 = Function(W)


    #Define Strain Rate and other tensors
    sr = 0.5*(grad(u0) + transpose(grad(u0)))
    gamdots = inner(sr,grad(u0))
    gamdotp = inner(S1, grad(u0))
    F = (grad(u0)*S0 + S0*transpose(grad(u0)))
    F1 = (grad(u1)*S + S*transpose(grad(u1)))

    #CHORIN'S PROJECTION METHOD (adapted for viscoelastic stress)

    # Tentative velocity step
    a1 = inner(u, v)*dx  
    L1 = inner(u0,v)*dx - k*inner(grad(u0)*u0, v)*dx -(k/Re)*inner(S0,grad(v))*dx- 0.5*(k/Re)*beta*inner(grad(u0), grad(v))*dx

    # Pressure update 
    a2 = (k/Re)*inner(grad(p), grad(q))*dx 
    L2 = -div(u1)*q*dx

    # Velocity update
    a3 = inner(u, v)*dx  + 0.5*(k/Re)*beta*inner(grad(u), grad(v))*dx
    L3 = inner(u1, v)*dx + (k/Re)*inner(grad(p1), v)*dx

    # Stress Update
    a4 = (eta + 1.0)*inner(S,R)*dx + We*inner(dot(u1,grad(S)),R)*dx - We*inner(F1, R)*dx
    L4 =  eta*inner(S0,R)*dx + (2.0)*(1.0-beta)*inner(sr,R)*dx 

    # Temperature Update
    a5 = inner(T,q)*dx + k*kappa*alpha*inner(grad(T),grad(q))*dx + k*inner(inner(u1,grad(T)),q)*dx
    L5 = inner(T0,q)*dx + k*alpha*(heatt*(inner(grad(T0),n1*q)*ds(1)) + mu*inner(gamdots,q)*dx + inner(gamdotp,q)*dx) #Neumann Condition on the outer Bearing is encoded in the weak formulation


    # Assemble matrices
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)
    A4 = assemble(a4)
    A5 = assemble(a5)

    n0v = interpolate(n0, V)
    I = Expression((('1.0','0.0'),
                    ('0.0','1.0')), degree=2)

    boundary_parts = FacetFunction("size_t", mesh)
    omega0.mark(boundary_parts,0)
    omega1.mark(boundary_parts,1)
    ds = Measure("ds")[boundary_parts]

    dimenp=(mu*r_a*w_j)/(c)
    dimenu=1/(r_a*w_j)

    dimensr=mu*r_a*w_j/(c)

    #print(dimenp, dimensr)
    #quit()

    sigma0 = (dimenp*(-p1*I+2*beta*sr) + S0)*t0
    sigma1 = (dimenp*(-p1*I+2*beta*sr) + S0)*t1

    omegaf0 = dimenp*p1*I*n0  #Nomral component of the stress 
    omegaf1 = dimenp*p1*I*n1


    innerforcex = -inner(Constant((1.0, 0.0)), omegaf0)*ds(0)
    innerforcey = -inner(Constant((0.0, 1.0)), omegaf0)*ds(0)  # Formulation see Li, Phillips 1999 eq.(5.1)-(5.2)
    outerforcex = -inner(Constant((1.0, 0.0)), omegaf1)*ds(1)
    outerforcey = -inner(Constant((0.0, 1.0)), omegaf1)*ds(1)
    innertorque = -r_a*inner(n0, sigma0)*ds(0)
    outertorque = -r_b*inner(n1, sigma1)*ds(1)
    torque = innertorque+outertorque


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
    zz=list()
    xx=list()
    yy=list()
    zzz=list()
    m=list()
    n=list()
    l=list()

    # Time-stepping
    iter = 0            # iteration counter
    maxiter = 1000000
    t = 0.0
    u_err = norm(u1.vector()-u0.vector())
    print(u_err)
    while t < Tf + DOLFIN_EPS and iter < maxiter:
        iter += 1
        print"t = %s,  Iteration = %d, Convergence Failures = %s, Loop = %s" %(t, iter, jj, j)
        u_err = norm(u1.vector()-u0.vector())

        # Compute tentative velocity step
        #begin("Computing tentative velocity")
        b1 = assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]
        solve(A1, us.vector(), b1, "bicgstab", "default")
        end()

        # Pressure correction
        #begin("Computing pressure correction")
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcp]
        [bc.apply(p1.vector()) for bc in bcp]
        solve(A2, p1.vector(), b2, "bicgstab", prec)
        end()

        # Velocity correction
        #begin("Computing velocity correction")
        b3 = assemble(L3)
        [bc.apply(A3, b3) for bc in bcu]
        solve(A3, u1.vector(), b3, "bicgstab", "default")
        end()
      
        # Stress correction
        # begin("Computing velocity correction")
        #b4 = assemble(L4)
        #[bc.apply(A4, b4) for bc in bcS]
        #solve(A4, S1.vector(), b4, "bicgstab", "default")
        #end()

        # Temperature correction
        # begin("Computing temperature correction")
        b5 = assemble(L5)
        [bc.apply(A5, b5) for bc in bcT]
        solve(A5, T1.vector(), b5, "bicgstab", "default")
        end()

        # Calcultate Force on the bearing
        #print("Normal Force on inner bearing: ",  (assemble(innerforcex), assemble(innerforcey)))
        #print("Torque on inner bearing: ", assemble(torque))

        # Record Torque Data 
        x.append(t)
        y.append(assemble(torque))
        z.append(assemble(innerforcey))
        zz.append(assemble(innerforcex))
        m.append(lambda1*0.5*(1.0+tanh(6*t-2.5))*w_j*r_a/c)
        n.append(lambda1*(0.5*1.0+0.5*tanh(6*t-2.5))*w_j)
        l.append(norm(u1.vector()-u0.vector()))


        # Record Timestep Error Data 
        oo= S1.vector()-S0.vector()
        #hh= p1.vector()-p0.vector()
        mm=u1.vector()-u0.vector()
        
        #if iter > 1:
           #xx.append(t)
           #yy.append(norm(hh,'linf')/norm(p1.vector(),'linf'))
           #yy.append(norm(oo,'linf')/norm(S1.vector(),'linf'))
           #zzz.append(norm(mm,'linf')/norm(u1.vector()))


        if max(norm(T1.vector(), 'linf'),norm(p1.vector(), 'linf')) > 10E6 or np.isnan(sum(T1.vector().array())):
            print 'FE Solution Diverging'   #Print message 
            with open("Incompressible Stability.txt", "a") as text_file:
                 text_file.write("Iteration:"+str(j)+"--- Re="+str(Re)+", We="+str(We)+", t="+str(t)+", dt="+str(dt)+'\n')
            dt=dt/2                         #Use Smaller timestep 
            j-=1                            #Extend loop
            jj+= 1                          # Convergence Failures
            break    
        
        # Plot solution (HIGHER ITERATIONS)

        plot(p1, title="Pressure", rescale=True)
        #plot(u1, title="Velocity", rescale=True, mode = "auto")
        #plot(T1, title="Temperature", rescale=True)


        #Update Velocity Boundary Condition
        w.t = t
        u0.assign(u1)
        T0.assign(T1)
        S0.assign(S1)     
        
        # Move to next time step

        t += dt

    # Impulse data (Trapezium Rule)
    yarr=np.asarray(y)
    impy=sum(yarr)
    zarr=np.asarray(z)
    impz=sum(zarr)
    print('Angular Impulse:', dt*impy)
    print('Normal Impulse:', dt*impz)

    if max(norm(T1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) > 10E6:
        # Plot Convergence Data
        fig1=plt.figure()
        plt.plot(xx, yy, 'r-', label='Extra Stress Timestep Error')
        plt.xlabel('time(s)')
        plt.ylabel('||tau1-tau0||/||tau1||')
        plt.savefig("Incompressible Viscoelastic Flow Results/Code-Stability/Euler Extra Stress Timestep Error We="+str(We)+"Re="+str(Re)+"t="+str(t)+".png")
        plt.clf()
        plt.plot(xx, zzz, 'g-', label='Velocity Field Timestep Error')
        plt.xlabel('time(s)')
        plt.ylabel('||u1-u0||/||u1||')
        plt.savefig("Incompressible Viscoelastic Flow Results/Code-Stability/Euler Velocity Covergence Timestep Error We="+str(We)+"Re="+str(Re)+"t="+str(t)+".png")
        plt.clf()

    if max(norm(T1.vector(),'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) < 10E6:

        # Matlab Plot of the Solution at t=Tf
        mplot(p1)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots/Pressure We="+str(We)+"Re="+str(Re)+"t="+str(t)+".png")
        plt.clf()
        mplot(T1)
        plt.colorbar()
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots/Temperature We="+str(We)+"Re="+str(Re)+"t="+str(t)+".png")
        plt.clf()

        # Plot Torque Data
        plt.plot(x, zz, 'b-', label='Horizontal Force')
        plt.xlabel('time(s)')
        plt.ylabel('Horizontal Load Force (N)')
        plt.savefig("Incompressible Viscoelastic Flow Results/Torque-Force/Euler Horizontal Load We="+str(We)+"Re="+str(Re)+"t="+str(t)+".png")
        plt.clf()
        plt.plot(x, y, 'r-', label='Torque')
        plt.xlabel('time(s)')
        plt.ylabel('Torque')
        plt.savefig("Incompressible Viscoelastic Flow Results/Torque-Force/Euler Torque We="+str(We)+"Re="+str(Re)+"t="+str(t)+".png")
        plt.clf()
        plt.plot(x, z, 'b-', label='Vertical Force')
        plt.xlabel('time(s)')
        plt.ylabel('Vertical Load Force (N)')
        plt.savefig("Incompressible Viscoelastic Flow Results/Torque-Force/Euler Vertical Load We="+str(We)+"Re="+str(Re)+"t="+str(t)+".png")
        plt.clf()


        #quit()

        #Plot PRESSURE Contours USING MATPLOTLIB
        # Scalar Function code

        #Set Values for inner domain as -infty

        x = Expression('x[0]', d=d, degree=d)  #GET X-COORDINATES LIST
        y = Expression('x[1]', d=d, degree=d)  #GET Y-COORDINATES LIST
        meshc= generate_mesh(c3, 35)
        Q1=FunctionSpace(meshc, "CG", 2)
        pj=Expression('0', d=d, degree=d) #Expression for the 'pressure' in the domian
        pjq=interpolate(pj, Q1)
        pjvals=pjq.vector().array()

        xyvals=meshc.coordinates()
        xqalsv = interpolate(x, Q1)
        yqalsv= interpolate(y, Q1)

        xqals= xqalsv.vector().array()
        yqals= yqalsv.vector().array()

        x = Expression('x[0]', d=d, degree=d)  #GET X-COORDINATES LIST
        y = Expression('x[1]', d=d, degree=d)  #GET Y-COORDINATES LIST
        pvals = p1.vector().array() # GET SOLUTION u= u(x,y) list
        xyvals = mesh.coordinates()     # CLEAN THIS UP!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
        uu = mlab.griddata(xvals, yvals, pvals, xx, yy, interp='nn') # u(x,y) data so that it can be used by 

        plt.contour(XX, YY, uu, 20)
        plt.colorbar()
        plt.title('Pressure Contours')   # PRESSURE CONTOUR PLOT
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots/Pressure Contours We="+str(We)+"Re="+str(Re)+"t="+str(t)+".png")
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

        plt.contour(XX, YY, TT, 20)
        plt.colorbar()
        plt.title('Temperature Contours')   # TEMPERATURE CONTOUR PLOT
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots/Temperature Contours We="+str(We)+"Re="+str(Re)+"t="+str(t)+".png")
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
        #print(xyvalsv.vector().array())
        for i in range(len(vjq.vector().array())/2):  
            qq.append(xyvalsv.vector().array()[2*i+1])
            rr.append(xyvalsv.vector().array()[2*i])

        xvalsj = np.asarray(rr)
        yvalsj = np.asarray(qq)


        g=list()
        h=list()
        n= mesh.num_vertices()
        #print(u1.vector().array())               # u is the FEM SOLUTION VECTOR IN FUNCTION SPACE 
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

        #Determine Speed 



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
                       density=4,              
                       color=speed/speed.max(),  
                       cmap=cm.gnuplot,                         # colour map
                       linewidth=1.0)       # line thickness
        plt.colorbar()                  # add colour bar on the right
        plt.title('Journal Bearing Problem')
        plt.savefig("Incompressible Viscoelastic Flow Results/Plots/Velocity Contours We="+str(We)+"Re="+str(Re)+"t="+str(t)+".png")   
        plt.clf()                                                                          # display the plot


    plt.close()


    if dt < tol:
       j=loopend+1
       break


    # Update Control Variables 
    if max(norm(u1.vector(), 'linf'),norm(p1.vector(), 'linf'),norm(u1.vector(), 'linf')) < 10E6 and np.isfinite(sum(u1.vector().array())):
        with open("Compressible Stability.txt", "a") as text_file:
             text_file.write("Solution Converges Re"+str(Re)+", We="+str(We)+", dt="+str(dt)+'\n')
        dt = 0.064  #Time Stepping                        #Go back to Original Timestep
        #gammah=10E0
        jj=0
        if w_j==300.0:
            w_j=2.0*w_j
            lambda1=1.0*c/(r_a*w_j)
        elif w_j==150.0:
            w_j=2.0*w_j
            lambda1=1.0*c/(r_a*w_j)
        elif w_j==50.0:
            w_j=3.0*w_j
            lambda1=1.0*c/(r_a*w_j)
        elif w_j==25.0:
            w_j=2.0*w_j
            lambda1=1.0*c/(r_a*w_j)
        elif w_j==10.0:
            w_j=2.5*w_j
            lambda1=1.0*c/(r_a*w_j)
        elif w_j==5.0:
            w_j=2.0*w_j
            lambda1=1.0*c/(r_a*w_j)
        elif w_j==2.5:
            w_j=2.0*w_j
            lambda1=1.0*c/(r_a*w_j)
        elif w_j==1.0:
            w_j=2.5*w_j
            lambda1=1.0*c/(r_a*w_j)
        elif w_j==0.5:
            w_j=2.0*w_j
            lambda1=1.0*c/(r_a*w_j)
        elif w_j==0.25:
            w_j=2.0*w_j
            lambda1=1.0*c/(r_a*w_j)





