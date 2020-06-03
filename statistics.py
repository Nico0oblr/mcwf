#!/usr/bin/env python3

import numpy as np
import scipy
import matplotlib.pyplot as plt
from mcwf import *
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 22})

dimension = 20
sites = 2
temperature = 0.0
coupling = 0.2
hopping = 1.0
hubbardU = 8.0 * hopping
frequency = 6.0
laser_frequency = frequency
laser_amplitude = 0.1
gamma = 0.01
periodic = False
t = 200.0
dt = 0.1
runs = 2
set_num_threads(4)

# Hubbard model
projector = HubbardProjector_sp(sites, sites // 2, sites - (sites // 2))
light_matter = Hubbard_light_matter_Operator(dimension, sites, coupling, hopping, hubbardU, periodic, projector);
elec_dim = projector.shape[0]
system_dimension = elec_dim * dimension

system = drivenCavity(temperature, frequency, laser_frequency, gamma, laser_amplitude, dimension, elec_dim)
system.system_hamiltonian().add(light_matter)
system_dimension = elec_dim * dimension
ident = scipy.sparse.identity(system_dimension)
state_distro = coherent_photon_state(temperature, dimension) + HubbardGroundState(sites, hopping, hubbardU, periodic, projector)

recorder = MCWFCorrelationRecorderMixin(runs)
charge_densities = [kroneckerOperator_IDLHS(projector @ n_th_subsystem_sp(HubbardOperators.n_up() + HubbardOperators.n_down(), i, sites) @ projector.getH(), dimension) - operatorize(ident) for i in range(sites)]

for charge_density in charge_densities:
    recorder.push_back(charge_density)

import time
start = time.time()
Solvers.two_time_correlation(system, state_distro, 0.0, t, dt, runs, charge_densities[0].eval(), recorder)
mid = time.time()
print("Normal: {} seconds for {} steps".format(mid - start, int(t / dt)))
print("This implies {} seconds per step".format((mid - start) / (t / dt)))

plt.plot(recorder.Var2(0))
plt.plot(recorder.expval(1))
