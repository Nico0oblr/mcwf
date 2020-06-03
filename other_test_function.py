#!/usr/bin/env python3

import numpy as np
from mcwf import *
import pytest
import scipy
from scipy.sparse.linalg import norm
from scipy.sparse import rand
from scipy.sparse.linalg import expm as sc_expm
import time

#dimension = 200
#density = 0.1
#A = rand(dimension, dimension, density = density)
#B = rand(dimension, dimension, density = density)
#vec = rand(dimension * dimension, 1, density = 1.0)

#start = time.time()
#kron = tensor_identity(A, dimension) @ tensor_identity_LHS(B, dimension)
#mid = time.time()
#kron_alt = kroneckerApply(A, B, vec.todense())
#end = time.time()
#kroneckerApplyLazy(A, B, vec.todense())
#after_end = time.time()

#kron_vec = kron @ vec
#kron_alt = kron_alt.reshape((kron_alt.shape[0], 1))
#kron_vec = kron_vec.reshape((kron_vec.shape[0], 1))

#print(np.linalg.norm(kron_vec - kron_alt))
#print(mid - start)
#print(end - mid)
#print(after_end - end)

sites  = 4
niter = 20
photon_dimension = 20

from scipy.sparse import random
projector = HubbardProjector_sp(sites, sites // 2, sites - sites // 2)
mat = -0.01j * Hubbard_light_matter_sp(photon_dimension,
                                       sites, 0.2, 1.0, 8.0, True, projector)
elec_dim = projector.shape[0]
print("sparsity: {}".format(mat.getnnz() / (mat.shape[0] * mat.shape[1])))
dimension = projector.shape[0] * photon_dimension
vec = random(dimension, 1, density = 1.0) \
    + 1.0j * random(dimension, 1, density = 1.0)

start1 = time.time()
iteration1 = ArnoldiIteration(mat, niter, niter, vec.todense())
end1 = time.time()
mat = -0.01j * Hubbard_light_matter_Operator(niter, sites, 0.2, 1.0, 8.0, True, projector)
mat = mat + kroneckerOperator_IDRHS(numberOperator_sp(photon_dimension), elec_dim)
start2 = time.time()

iteration2 = ArnoldiIterationOperator(mat, niter, niter, vec.todense())
end2 = time.time()

print(np.linalg.norm(iteration1.H() - iteration2.H()))
print(np.linalg.norm(iteration1.V() - iteration2.V()))
print(end1 - start1)
print(end2 - start2)

dimension = 20
sites = 6
temperature = 0.0
coupling = 0.2
hopping = 1.0
hubbardU = 8.0 * hopping
frequency = 6.0
laser_frequency = frequency
laser_amplitude = 0.1
gamma = 0.0
periodic = True
t = 2.0
dt = 0.1
set_num_threads(4)

# Hubbard model
projector = HubbardProjector_sp(sites, sites // 2, sites // 2)
hubbard_proj = tensor_identity_LHS(projector, dimension);
light_matter = Hubbard_light_matter_Operator(dimension, sites, coupling, hopping, hubbardU, periodic, projector);
elec_dim = projector.shape[0]
system_dimension = elec_dim * dimension
# Observables
photon_number = tensor_identity(numberOperator_sp(dimension), elec_dim)
photon_displacement = tensor_identity(creationOperator_sp(dimension) + annihilationOperator_sp(dimension), elec_dim)
single_spin = tensor_identity_LHS(projector @ n_th_subsystem_sp(HubbardOperators.n_up() - HubbardOperators.n_down(), 0, sites) @ projector.getH(), dimension)
state = np.random.rand(system_dimension)
system = drivenCavity(temperature, frequency, laser_frequency, gamma, laser_amplitude, dimension, elec_dim)
system.system_hamiltonian().add(light_matter)

start = time.time()
for i in range(20):
    system.system_hamiltonian().propagate(i, 0.01, state)
end = time.time()
print(end - start)

recorder2 = StateObservableRecorder([np.identity(system_dimension, dtype = np.complex), light_matter.eval(), photon_number, single_spin])
state_distro = coherent_photon_state(temperature, dimension)
state_distro += HubbardNeelState_sp(sites, projector)
start = time.time()
Solvers.direct_closed_observable(system.system_hamiltonian(), state_distro.draw(), t, dt, recorder2)
end = time.time()
print(end - start)
