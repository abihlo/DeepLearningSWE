from firedrake import *
import numpy as np
import os
import matplotlib.pyplot as pl
import funcs

def swe(N=200,M=100,T=1.,M_length=1,degree=1,inum=1,bnum=1):

    #Hacky parameter to avoid issue with .at() on periodic meshes
    eps = 1e-12
    
    #Set up finite element mesh and spaces
    mesh = PeriodicIntervalMesh(M,M_length)
    U = FunctionSpace(mesh,"CG",degree)
    W = MixedFunctionSpace((U,U))
  
    #Specify quadrature degree (to be 'exact')
    quad_degree = 2*degree + 4
    dxx = dx(degree=quad_degree)
    dSS = dS(degree=quad_degree)
    
    #Set up initial conditions
    w0 = Function(W)
    u0, h0 = w0.split()
    t = 0.
    t_dump = 0.
    x = SpatialCoordinate(U.mesh())
    u_tmp, h_tmp = funcs.initial_conditions(x[0],t,num=inum)
    u0.assign(project(u_tmp,W.sub(0)))
    h0.assign(project(h_tmp,W.sub(1)))

    #Set up bottom topography
    beta = project(funcs.bottom(x[0],num=bnum),U)

    #define the timestep
    timestep = float(T)/N
        
    #Build the weak form
    phi, psi = TestFunctions(W)
    w1 = Function(W)
    w1.assign(w0)
    u1, h1 = split(w1)
    u0, h0 = split(w0)
    ut = (u1-u0)/timestep
    ht = (h1-h0)/timestep
    umid = 0.5*(u1+u0)
    hmid = 0.5*(h1+h0)

    F1 = (ut + hmid.dx(0) + 0.5*(umid**2).dx(0)) * phi * dxx
    F2 = (ht + (hmid*umid).dx(0) + (beta*umid).dx(0)) * psi * dxx
    F = F1 + F2

    #Set up solver
    uprob = NonlinearVariationalProblem(F, w1)
    tol = 1e-6
    sparameters = {'mat_type': 'aij',
                   'ksp_type': 'preonly',
                   'pc_type': 'lu',
                   'snes_rtol': 1e-50,
                   'snes_atol' : tol,
                   'snes_stol': tol,
                   'ksp_atol': tol,
                   'ksp_rtol': tol,
                   'ksp_divtol': 1e4}
    usolver = NonlinearVariationalSolver(uprob, solver_parameters=sparameters)


    #Split form for UFL
    u0, h0 = w0.split()
    u1, h1 = w1.split()

    #Initialise output files
    ufile = File('tmp/1du.pvd')
    vfile = File('tmp/1dh.pvd')
    File('tmp/1dbeta.pvd').write(beta)

    #Initialise solution vectors
    u_vec = []
    h_vec = []

    #Define spatial vector
    x_vec = mesh.coordinates.dat.data
    
    #Initialise temporal vector
    t_vec = [0]

    #Set up hacky shift vector to avoid hitting periodic boundary
    eps_vec = eps*np.ones_like(x_vec)
    eps_vec[-1] = - eps_vec[-1]
    
    #Append initial data
    u_vec.append([np.array(u0.at(x_vec+eps_vec))])
    h_vec.append([np.array(h0.at(x_vec+eps_vec))])
    #Save beta array
    beta_vec = np.array(beta.at(x_vec+eps_vec))
    
    counter = 0
    #Loop over time
    while (t < T - 0.5*timestep):
        t += timestep
        counter +=1
        #solve
        usolver.solve()
        w0.assign(w1)
        #Save solution
        t_vec.append(t)
        try:
            u_vec.append([np.array(u1.at(x_vec+eps_vec))])
            h_vec.append([np.array(h1.at(x_vec+eps_vec))])
        except:
            u_vec.append([np.array(u1.at(x_vec+eps_vec))])
            h_vec.append([np.array(h1.at(x_vec+eps_vec))])

        #write solution
        t_dump = t_dump + timestep
        if t_dump>= 0.05:
            t_dump = 0.
            ufile.write(u1,time=t)
            vfile.write(h1,time=t)



    #Convert to numpy arrays where required
    u_vec = np.array(u_vec)
    h_vec = np.array(h_vec)
    bshape = u_vec.shape
    gshape = (bshape[0],bshape[-1])
    u_vec = u_vec.reshape(gshape)
    h_vec = h_vec.reshape(gshape)
            
    #Save data as numpy arrays
    dir_name = 'output/1d_initial%d_bottom%d/' % (inum, bnum)
    os.makedirs(dir_name,exist_ok=True)
    np.save(dir_name+'t.npy',t_vec)
    np.save(dir_name+'x.npy',x_vec)
    np.save(dir_name+'u.npy',u_vec)
    np.save(dir_name+'h.npy',h_vec)
    np.save(dir_name+'beta.npy',beta_vec)

    print('u.shape', u_vec.shape)
    print('h.shape', h_vec.shape)
    print('beta.shape', beta_vec.shape)
    
    return t_vec, x_vec, u_vec, h_vec, beta_vec
        

if __name__=="__main__":
    swe(N=1000,M=500,inum=1,bnum=1)
