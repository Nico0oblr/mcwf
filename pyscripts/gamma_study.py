#!/usr/bin/env python3

import pickle
def metastore(filename, *args):
    all_dict = dict(globals(), **locals())
    arg_dict = {}
    for arg in args:
        arg_dict[arg] = all_dict[arg]

    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(arg_dict, f, pickle.HIGHEST_PROTOCOL)
        
import sys
sys.path.insert(0, './')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mcwf import *
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 22})

def obtain_solver(solver_tag):
    tag_dict = {"mcwf": Solvers.observable_calc,
               "runge_kutta": Solvers.observable_kutta,
               "direct:": Solvers.observable_direct,
               "direct_closed": Solvers.direct_closed_observable,
               "mcwf_correlation": Solvers.two_time_correlation,
               "direct_correlation": Solvers.two_time_correlation_direct,
               "direct_closed_correlation": Solvers.direct_closed_two_time_correlation}
    return tag_dict[solver_tag]

def plot_distribution(dist, nbins = 200, cmap = plt.cm.viridis, log = False, ax = None, fig = None):
    if ax is None:
        fig, ax = plt.subplots()
    def histogram_from_runs(dat, nbins):
        out = []
        minn = np.min(dat)
        maxn = np.max(dat)
        for run in dat.T:
            hist, edges = np.histogram(run, bins=np.linspace(minn, maxn, nbins))
            out.append(hist)
        return minn, maxn, out
    minn, maxn, hists = histogram_from_runs(dist, nbins)
    hists = np.array(hists).T / runs
    norm = None
    if log:
        norm = matplotlib.colors.LogNorm(vmin = np.min(hists[hists > 0.0]), vmax = np.max(hists))
    else:
        norm = matplotlib.colors.Normalize(vmin = np.min(hists), vmax = np.max(hists))
    im = ax.imshow(hists, aspect = "auto", extent = (0.0, t, minn, maxn), cmap = cmap, 
                   interpolation="nearest", origin = "lower",
                    norm = norm)
    clb = fig.colorbar(im)
    return fig, ax, clb

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
periodic = True
t = 200.0
dt = 0.05
set_num_threads(4)

# Hubbard model
projector = HubbardProjector(sites, sites // 2, sites // 2);
hubbard_proj = tensor_identity_LHS(projector, dimension);  
light_matter = Hubbard_light_matter(dimension, sites, coupling, hopping, hubbardU, periodic);
light_matter = hubbard_proj * light_matter * hubbard_proj.getH()
elec_dim = projector.shape[0]
system_dimension = elec_dim * dimension
# Observables
photon_number = tensor_identity(numberOperator(dimension), elec_dim).toarray()
photon_displacement = tensor_identity(creationOperator(dimension) + annihilationOperator(dimension), elec_dim).toarray()
single_spin = tensor_identity_LHS(projector @ nth_subsystem(HubbardOperators.n_up() - HubbardOperators.n_down(), 0, sites) @ np.asmatrix(projector).getH(), dimension).toarray()

for gamma in np.linspace(0.0, 0.2, 20):
    runs = 1000
    #gamma = 0.1
    t = 200.0
    filename = "gamma_study{}".format(gamma).replace(".", "_")
    solver_tag = "mcwf"
    state_distro = coherent_photon_state(temperature, dimension)
    state_distro += HubbardNeelState(sites, projector)
    system = CavityLindbladian(frequency, laser_frequency, laser_amplitude, elec_dim, dimension, 
                               light_matter, dt, gamma, temperature)
    system.system_hamiltonian().set_order(3)
    recorder = MCWFObservableRecorder([np.identity(system_dimension, dtype = np.complex), light_matter, photon_number, single_spin, photon_displacement], runs)
    obtain_solver(solver_tag)(system, state_distro, t, dt, runs, recorder)
    metastore(filename, "system", "state_distro", "recorder", "solver_tag", "runs")