#!/usr/bin/env python3

import numpy as np
import scipy
import matplotlib.pyplot as plt
from mcwf import *
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 22})

dimension = 20
sites = 10
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
niter = 75

# Hubbard model
projector = HubbardProjector_sp(sites, sites // 2, sites - (sites // 2))
light_matter = Hubbard_light_matter_Operator(dimension, sites, coupling, hopping, hubbardU, periodic, projector)
elec_dim = projector.shape[0]
#state_distro = coherent_photon_state(temperature, dimension) + HubbardGroundState(sites, hopping, hubbardU, periodic, projector)
state_distro = HubbardGroundState(sites, hopping, hubbardU, periodic, projector)

projector_vector = HubbardProjector_basis(sites, sites // 2, sites - (sites // 2))
vec = state_distro.draw()
decomp = vector_to_matrix(state_distro.draw(), 4, projector_vector, projector.shape[1])
#real_decomp = vec.reshape(20, vec.size // 20)
#svd = np.linalg.svd(real_decomp, full_matrices = False)
#print(svd[1])
compute_svd(decomp)[1]
