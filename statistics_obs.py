#!/usr/bin/env python3

import pickle
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mcwf import *
import argparse
import uuid
import subprocess
import os

def metastore(filename, *args):
    all_dict = dict(globals(), **locals())
    arg_dict = {}
    for arg in args:
        arg_dict[arg] = all_dict[arg]
    this_content = None
    with open(os.path.basename(__file__), 'r') as f:
        this_content = f.read()
    label = subprocess.check_output(["git", "describe", "--always"]).strip()
    
    arg_dict["script"] = this_content
    arg_dict["git"] = label
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(arg_dict, f, pickle.HIGHEST_PROTOCOL)

def construct_parser():
    parser = argparse.ArgumentParser(prog='mcwf run for hubbard model')
    parser.add_argument('--sites', type = int, help = "sites in the hubbard model")
    parser.add_argument('--coupling', type = float, default = 0.2, help = "light matter coupling")
    parser.add_argument('--hubbardU', type = float, default = -8.0, help = "hubbard onsite term")
    parser.add_argument('--frequency', type = float, default = 6.0, help = "cavity frequency")
    parser.add_argument('--gamma', type = float, default = 0.05, help = "cavity leak")
    parser.add_argument('--laser_amplitude', type = float, default = 0.1, help = "laser amplitude")
    parser.add_argument('--dimension', type = int, default = 20, help = "photonic truncation dimension")
    parser.add_argument('--temperature', type = float, default = 0.0, help = "temperature of environment")
    parser.add_argument('--hopping', type = float, default = 1.0, help = "hopping energy of hubbard model")
    parser.add_argument('--num-threads', type = int, default = 1, help = "number of threads for mcwf procedure")
    parser.add_argument('--laser-frequency', type = float, help = "frequency of driving laser")
    parser.add_argument('-t', type = float, default = 200.0, help = "end time")
    parser.add_argument('--dt', type = float, default = 0.1, help = "time step")
    parser.add_argument('--runs', type = int, default = 100, help = "number of mcwf runs")
    parser.add_argument('--periodic', action='store_true', help = "periodic chain if provided, otherwise open")
    return parser

parser = construct_parser()
pargs = parser.parse_args()

if pargs.laser_frequency is None:
    pargs.laser_frequency = pargs.frequency
pargs.hubbardU *= pargs.hopping
pargs.frequency *= pargs.hopping

sites = pargs.sites
set_num_threads(pargs.num_threads)

projector = HubbardProjector_sp(sites, sites // 2, sites - (sites // 2))
light_matter = Hubbard_light_matter_Operator(pargs.dimension, sites, pargs.coupling, pargs.hopping, pargs.hubbardU, pargs.periodic, projector);
elec_dim = projector.shape[0]
system_dimension = elec_dim * pargs.dimension

system = drivenCavity(pargs.temperature, pargs.frequency, pargs.laser_frequency, pargs.gamma, pargs.laser_amplitude, pargs.dimension, elec_dim)
system.system_hamiltonian().add(light_matter)
system_dimension = elec_dim * pargs.dimension
ident = scipy.sparse.identity(system_dimension)
state_distro = coherent_photon_state(pargs.temperature, pargs.dimension) + HubbardGroundState(sites, pargs.hopping, pargs.hubbardU, pargs.periodic, projector)

recorder = MCWFObservableRecorder()
charge_densities = [kroneckerOperator_IDLHS(projector @ n_th_subsystem_sp(HubbardOperators.n_up() + HubbardOperators.n_down(), i, sites) @ projector.getH(), pargs.dimension) - operatorize(ident) for i in range(sites)]

for charge_density in charge_densities:
    recorder.push_back(charge_density)

Solvers.observable_calc(system, state_distro, t, dt, runs, recorder)
results = recorder.data()
unique_filename = str(uuid.uuid4())
metastore(unique_filename, "results", "pargs")
