import dolfin as dl
import numpy as np
import scipy.sparse as sp
from math import sqrt
import time # measurements of time
#from memory_profiler import memory_usage, profile

dl.parameters["form_compiler"]["optimize"] = True
dl.parameters["form_compiler"]["cpp_optimize"] = True
dl.parameters["form_compiler"]["cpp_optimize_flags"] = '-Ofast'
dl.parameters["form_compiler"]["representation"] = "uflacs"
dl.parameters["form_compiler"]["quadrature_degree"] = 2

####################################################################
# handling boundary conditions
####################################################################

def getbcsValues(bcs):
    """
    to store a list of the values on the boundary
    """
    bcsValues = []
    
    for b in bcs:
        bcsValues.append(b.value())

    return bcsValues


def homogenizeBCs(bcs):
    """
    to make all bcs homogenized
    """
    
    for b in bcs:
        b.homogenize()

    return


def applyBCs(bcs, R):
    """
    to apply bcs to the Vector/Matrix
    """

    for b in bcs:
        b.apply(R)

    return 


def setbcsValues(bcs, bcsValues):
    """
    to set back the boundary conditions
    """
    N = len(bcs); N1 = len(bcsValues);

    if (N != N1): 
        print("length of bcsValues does not match bcs") 
        exit(0)
    
    else:
        for i in range(N):
            b = bcs[i]
            b.set_value(bcsValues[i])

    return

####################################################################
# handling assembles
####################################################################
#@profile
def assembleVec(F, bcs, RVec, R):
    dl.assemble(F, tensor = RVec)
    applyBCs(bcs, RVec)
    R[:] = RVec[:]

#@profile
def assembleCSR(J, bcs, KMat, KCSR):
    dl.assemble(J, tensor = KMat)
    applyBCs(bcs, KMat)
    row, col, val = KMat.mat().getValuesCSR()
    KCSR.data = val; KCSR.indices = col; KCSR.indptr = row


def calC(t, cmin, cmax):

    if t<0.: t=0.

    c = 2. * sqrt(t)
    if (c<cmin): c=cmin
    if (c>cmax): c=cmax

    return c


def printInfo(error, t, c, tol,
              eps, qdot, qdotdot, 
              nIters, nPrint, 
              info_force, info): 
    
    ## printing control
    if nIters % nPrint == 1:
        #print('\t------------------------------------')
        if info_force == True:
            print(('  DR Iteration %d: Max force = %g (tol = %g)' +
                   ' Max velocity = %g') % (nIters, error, tol, 
                                            np.max(np.absolute(qdot))))
        if info == True: 
            print('Damping t: ',t, );
            print('Damping coefficient: ', c)
            print('Max epsilon: ',np.max(eps))
            print('Max acceleration: ',np.max(np.absolute(qdotdot)))

####################################################################
# solver
####################################################################

#@profile
def DynamicRelaxSolve(F, u, bcs, J, 
                      # default parameters
                      tol = 1e-8, nKMat = 50, nPrint = 1000, 
                      info = False, info_force = True):

    # parameters not to change
    cmin  = 1e-3; cmax  = 3.9; h_tilde=1.1; h=1.;
    # initialize all arrays
    N = u.vector().local_size(); #print("--------num of DOF's: %d-----------" % N)
    #initialize displacements, velocities and accelerations
    q, qdot, qdotdot = np.zeros(N), np.zeros(N), np.zeros(N)
    #initialize displacements, velocities and accelerations from a previous time step
    q_old, qdot_old, qdotdot_old = np.zeros(N), np.zeros(N), np.zeros(N)
    #initialize the M, eps, R_old arrays
    eps, M, R, R_old = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    # initialize tensors
    RVec, KMat, KCSR = dl.PETScVector(), dl.PETScMatrix(), sp.csr_matrix((N, N), dtype='float')
    # save the boundary value data
    bcsValues = getbcsValues(bcs)
    # make sure that the bcs is set on the solution
    applyBCs(bcs, u.vector())
    # homogenize the boundary conditions
    homogenizeBCs(bcs)

    # start to solve the problem
    assembleVec(F, bcs, RVec, R)
    assembleCSR(J, bcs, KMat, KCSR)
    M[:] = h_tilde*h_tilde/4. * np.array(np.absolute(KCSR).sum(axis = 1)).squeeze()
    q[:] = u.vector()
    qdot[:] = - h/2. * R / M
    # set the counters for iterations and 
    nIters, iKMat = 0, 0; error = 1.0;

    timeZ = time.time() #Measurement of loop time.
    
    while error > tol:
        
        print(f"error = {error}")
        
        # marching forward
        q_old[:] = q[:]; R_old[:] = R[:]
        q[:] += h*qdot; u.vector()[:] = q[:]
        assembleVec(F, bcs, RVec, R)
        nIters += 1; iKMat += 1; error = np.max(np.absolute(R))
        
        # damping calculation
        S0 = np.dot((R - R_old)/h,  qdot); t = S0 / np.einsum('i,i,i', qdot,M,qdot)
        c = calC(t, cmin, cmax)

        # determine whether to recal KMat
        eps = h_tilde*h_tilde/4. * np.absolute(
                np.divide((qdotdot - qdotdot_old), (q - q_old),
                out = np.zeros_like( (qdotdot - qdotdot_old) ),
                where = (q - q_old)!=0))
        
        # calculating the jacobian matrix
        if ((np.max(eps) > 1) and (iKMat > nKMat)): #SPR JAN max --> min
            if info==True: print('\tRecalculating the tangent matrix: ', nIters)
            iKMat = 0; assembleCSR(J, bcs, KMat, KCSR)
            M[:] = h_tilde*h_tilde/4. * np.array(np.absolute(KCSR).sum(axis = 1)).squeeze()

        #compute new velocities and accelerations
        qdot_old[:] = qdot[:]; qdotdot_old[:] = qdotdot[:];
        qdot = (2.- c*h)/(2 + c*h) * qdot_old - 2.*h/(2.+c*h)* R / M
        qdotdot = qdot - qdot_old 
            
        # output on screen
        printInfo(error, t, c, tol,
                  eps, qdot, qdotdot,
                  nIters, nPrint,
                  info_force, info)


    # set the boundary value back
    setbcsValues(bcs, bcsValues)

    # free space for all the relevant arrays
    del q, qdot, qdotdot, q_old, qdot_old, qdotdot_old, eps, M, R_old
    del RVec, KMat, KCSR
    del bcsValues

    # check if converged
    convergence = True
    if np.isnan(np.max(np.absolute(R))):
        convergence = False

    # print final info
    
    if convergence:
        print("  DRSolve finished in %d iterations and %fs" % \
              (nIters, time.time() - timeZ))
    else:
        print("  FAILED to converged")

    return nIters, convergence