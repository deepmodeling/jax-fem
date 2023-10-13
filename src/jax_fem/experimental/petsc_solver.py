import numpy as onp
import petsc4py
petsc4py.init()
from petsc4py import PETSc

n = 10 # Size of vector
x = PETSc.Vec().createSeq(n) # Faster way to create a sequential vector.

x.setValues(range(n), range(n))

print(x.getArray())
print(x.getValues(3))
print(x.getValues([1, 2]))

x.setValues(range(n), range(n))
x.shift(1)
print(x.getArray())
x.shift(-1)
print(x.getArray())

x.setValues(range(n), range(n))

print(x.sum())
print(x.min())
print(x.max())

print(x.dot(x)) # dot product with self
 
print ('2-norm =', x.norm())
print ('Infinity-norm =', x.norm(PETSc.NormType.NORM_INFINITY))

m, n = 4, 4 # size of the matrix
A = PETSc.Mat().createAIJ([m, n]) # AIJ represents sparse matrix
A.setUp()
A.assemble()

print(A.getValues(range(m), range(n)))
print(A.getValues(range(2), range(1)))

A.setValue(1, 1, -1)
A.setValue(0, 0, -2)
A.setValue(2, 2, -5)
A.setValue(3, 3, 6)                        
# A.setValues([0, 1], [2, 3], [1, 1, 1, 1])

# A.setValuesIJV([0, 1], [2, 3], [1, 1])

A.assemble()

A.zeroRows([0, 1])
 
print(A.getValues(range(m), range(n)))


exit()

print(A.getSize())
B = A.copy()
B.transpose()
print(A.getSize(), B.getSize())
print(B.getValues(range(4), range(4)))

C = A.matMult(B)
print(C.getValues(range(m), range(n)))

x = PETSc.Vec().createSeq(4) # making the x vector
x.set(1) # assigning value 1 to all the elements
y = PETSc.Vec().createSeq(4) # Put answer here.
A.mult(e, y) # A*e = y
print(y.getArray())


print("Matrix A: ")
print(A.getValues(range(m), range(n))) # printing the matrix A defined above

b = PETSc.Vec().createSeq(4) # creating a vector
b.setValues(range(4), [10, 5, 3, 6]) # assigning values to the vector

print('\\n Vector b: ')
print(b.getArray()) # printing the vector 

x = PETSc.Vec().createSeq(4) # create the solution vector x

ksp = PETSc.KSP().create() # creating a KSP object named ksp
ksp.setOperators(A)

# Allow for solver choice to be set from command line with -ksp_type <solver>.
ksp.setFromOptions()
print ('\\n Solving with:', ksp.getType()) # prints the type of solver

# Solve!
ksp.solve(b, x) 

print('\\n Solution vector x: ')
print(x.getArray())
