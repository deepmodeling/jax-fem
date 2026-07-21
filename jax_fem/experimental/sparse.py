import scipy
import numpy as onp
import jax
import jax.numpy as np
from jax.experimental.sparse import BCOO
from jax_fem.experiments import sparsejac


from jax.config import config
config.update("jax_enable_x64", True)


def exp1():
    I = onp.array([3, 2, 2, 1, 0])
    J = onp.array([3, 3, 3, 1, 2])
    V = onp.array([10., 1., 2., 7., 9.])
    A_sp = scipy.sparse.csc_array((V, (I, J)), shape=(4, 4))
    print(A_sp)
    A_sp.sort_indices()
    print(A_sp)
    print(A_sp.todense())


def exp2():
    M = np.array([[0., 2., 0.], [1., 0., 4.]])
    M_sp = BCOO.fromdense(M)
    R = M_sp + M_sp
    print(M_sp.data)
    print(R.sum_duplicates().data)


def exp3():
    I = onp.array([0,3,1,0])
    J = onp.array([0,3,1,2])
    V = onp.array([4.,5,7,9])
    A = scipy.sparse.coo_matrix((V,(I,J)),shape=(4,4))

    A_sp = BCOO.from_scipy_sparse(A)

    print(A_sp.data)

    fn = lambda x: x**2
    sparsity = BCOO.fromdense(np.eye(10000))
    x = jax.random.uniform(jax.random.PRNGKey(0), shape=(10000,))

    sparse_fn = jax.jit(sparsejac.jacrev(fn, sparsity))

    print(f"Finished JIT")

    J = sparse_fn(x)
    print(J)


if __name__=="__main__":
    exp1()
