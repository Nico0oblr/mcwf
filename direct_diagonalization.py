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
niter = 75
set_num_threads(4)

# Hubbard model
projector = HubbardProjector_sp(sites, sites // 2, sites - (sites // 2))
light_matter = Hubbard_light_matter_Operator(dimension, sites, coupling, hopping, hubbardU, periodic, projector);
elec_dim = projector.shape[0]
system_dimension = elec_dim * dimension

# Observables
#photon_number = kroneckerOperator_IDRHS(numberOperator_sp(dimension), elec_dim)
#photon_displacement = kroneckerOperator_IDRHS(creationOperator_sp(dimension) + annihilationOperator_sp(dimension), elec_dim)
#single_spin = kroneckerOperator_IDLHS(projector @ n_th_subsystem_sp(HubbardOperators.n_up() - HubbardOperators.n_down(), 0, sites) @ projector.getH(), dimension)
#opid = operatorize(scipy.sparse.identity(system_dimension, dtype = np.complex))

system = drivenCavity(temperature, frequency, laser_frequency, gamma, laser_amplitude, dimension, elec_dim)
system.system_hamiltonian().add(light_matter)
system_dimension = elec_dim * dimension
ground_energy, ground_state = find_groundstate(light_matter, niter)
