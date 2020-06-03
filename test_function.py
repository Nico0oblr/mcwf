#!/usr/bin/env python3
import numpy as np
from mcwf import *
import pytest
import scipy
from scipy.sparse.linalg import norm
from scipy.sparse.linalg import expm as sc_expm
import time
from scipy.sparse import random
from scipy import stats

sites = 8
projector = HubbardProjector_sp(sites, sites // 2, sites - sites // 2)
mat = -0.05j * Hubbard_light_matter_sp(20, sites, 0.2, 1.0, 8.0, True, projector)
dimension = mat.shape[0]
niter = 10

print("dimension: {}".format(dimension))
print("sparsity: {}".format(mat.getnnz() / (mat.shape[0] * mat.shape[1])))
vec = random(dimension, 1, density = 1.0) \
    + 1.0j * random(dimension, 1, density = 1.0)
vec /= np.linalg.norm(vec.todense())

start = time.time()
scpres = scipy.sparse.linalg.expm_multiply(mat, vec).todense()
mid = time.time()
expres = expm_multiply_simple(mat, vec.todense(), 1.0).todense()
end = time.time()
alt = exp_krylov_alt(mat, vec.todense(), niter);
after_end = time.time()
arres = exp_krylov(mat, vec.todense(), niter)
aafter_end = time.time()
# Just consistently make them column vectors
scpres = scpres.reshape((scpres.size, 1))
expres = expres.reshape((expres.size, 1))
alt = alt.reshape((alt.size, 1))
arres = arres.reshape((arres.size, 1))

print("=====TIMES=====")
print("KrylovV1: {}".format(after_end - end))
print("KrylovV2: {}".format(aafter_end - after_end))
print("Mine: {}".format(end - mid))
print("Scipy: {}".format(mid - start))

print("=====SHAPES=====")
print(scpres.shape)
print(expres.shape)
print(alt.shape)
print(arres.shape)

print("=====RESULTS=====")
print("MTHDIFF: {}".format(np.linalg.norm(scpres - expres)))
print("KRDiffV1: {}".format(np.linalg.norm(alt - expres)))
print("KRDiffV2: {}".format(np.linalg.norm(arres - expres)))
print("KRDiffSYM: {}".format(np.linalg.norm(arres - alt)))

if dimension > 8000:
    exit(0)
print("=====Convergence=====")
#exit(0)
iteration = ArnoldiIteration(mat, niter, niter, vec.todense())
print("Arnoldi convergence: {}"
      .format(np.linalg.norm(iteration.V() @ iteration.H() -
                             mat.todense() @ iteration.V())))
print("Arnoldi convergence: {}"
      .format(np.linalg.norm(iteration.V() @ iteration.H()
                             @ iteration.V().T.conj() - mat.todense())))
print("Should be zero: {}"
      .format(np.linalg.norm(iteration.H() - iteration.V().T.conj()
                             @ mat.todense() @ iteration.V())))
x = iteration.eigenvectors().T.conj() @ iteration.eigenvectors()
x[np.abs(x) < 1e-10] = 0
print("Should be zero {}".format(np.linalg.norm(x - np.diag(np.diagonal(x)))))
print("=====MISC=====")

print("Arnoldi Norm {}".format(np.linalg.norm(iteration.V() @ iteration.H()
                                              @ iteration.V().T.conj())))
print("Arnoldi Norm {}".format(np.linalg.norm(iteration.H())))
print(iteration.H().shape)

exit(0)
import matplotlib.pyplot as plt
n, bins, patches = plt.hist(np.abs(np.linalg.eig(mat.todense())[0]), bins = 50, density = False, alpha = 0.5)
plt.hist(np.abs(np.linalg.eig(iteration.H())[0]), bins = bins, density = False, alpha = 0.5)
plt.show()
