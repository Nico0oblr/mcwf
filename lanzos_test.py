#!/usr/bin/env python3

import numpy as np
import scipy
import scipy.sparse
import matplotlib.pyplot as plt
from mcwf import *
import time
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 22})

sites = 7
hopping = 1.0
hubbardU = 8.0 * hopping
gamma = 0.0
periodic = True
set_num_threads(4)
niter = 75

def vshape(vec):
    return vec.reshape((vec.size, 1))

def fix_state_gauge(state):
    state /= state[0]
    state /= np.linalg.norm(state)
    return state

#np.random.seed(int(time.time()))
np.random.seed(0)
# Hubbard model
projector = HubbardProjector_sp(sites, sites // 2, sites - (sites // 2))
hubbard = Hubbard_hamiltonian_sp(sites, hopping, hubbardU, periodic, projector)
print(hubbard.shape)
elec_dim = projector.shape[0]
#initial_state = HubbardNeelState_sp(sites, projector).draw()
initial_state = np.random.rand(elec_dim)
initial_state /= np.linalg.norm(initial_state)

start = time.time()
ground_energy, ground_state = find_groundstate(hubbard, niter)
mid = time.time()
real_eival, real_eivec = np.linalg.eig(hubbard.todense())
end = time.time()

real_eival = np.real_if_close(real_eival)
real_args = real_eival.argsort()
real_eival = real_eival[real_args]; real_eivec = real_eivec[:,real_args]
real_ground_state = vshape(real_eivec[:,0])
real_ground_energy = real_eival[0]
ground_state = vshape(ground_state)

ground_state = fix_state_gauge(ground_state)
real_ground_state = fix_state_gauge(real_ground_state)

print(ground_state.shape)
print(real_ground_state.shape)

print("eival diff: {}".format(abs(ground_energy - real_ground_energy)))
print("eivec diff: {}".format(np.linalg.norm(ground_state - real_ground_state)))
print("Lanczos error: {}".format(np.linalg.norm(hubbard @ ground_state - ground_energy * ground_state)))
#print("Direct error: {}".format(np.linalg.norm(hubbard @ real_ground_state - real_ground_energy * real_ground_state)))
print("Lanczos duration: {}".format(mid - start))
print("Direct duration: {}".format(end - mid))
exit(0)

n, bins, patches = plt.hist(real_eival, color = "blue", bins = 150, alpha = 0.5, edgecolor='black', linewidth=0.2)
plt.hist(eival, color = "green", bins = bins, alpha = 0.5, edgecolor='black', linewidth=0.2)

plt.gca().set_yticks(list(range(int(np.min(n)),
                                int(np.max(n)) + 1)))
plt.grid()
#plt.plot(np.sort(real_eival))
#plt.plot(np.sort(eival))

plt.show()
