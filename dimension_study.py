def dimension_study(dimension):
    sites = 2
    temperature = 0.0
    coupling = 0.2
    hopping = 1.0
    hubbardU = 8.0 * hopping
    frequency = 6.0
    laser_frequency = frequency
    laser_amplitude = 0.1
    gamma = 0.0
    periodic = True
    time = 200
    #dt = 0.002

    projector = HubbardProjector(sites, sites // 2, sites // 2);
    hubbard_proj = tensor_identity_LHS(projector, dimension);  
    light_matter = Hubbard_light_matter(dimension, sites, coupling, hopping, hubbardU, periodic);
    light_matter = hubbard_proj * light_matter * hubbard_proj.getH()
    elec_dim = projector.shape[0]
    system = drivenCavity(temperature, frequency, laser_frequency, gamma, laser_amplitude, dimension, elec_dim)
    system.system_hamiltonian().add(light_matter)
    system_dimension = elec_dim * dimension
    state_distro = coherent_photon_state(temperature, dimension)
    #state_distro += HubbardNeelState(sites, projector)
    state_distro += DimerGroundState(hopping, hubbardU)

    photon_number = tensor_identity(numberOperator(dimension), elec_dim).toarray()
    photon_position = tensor_identity(creationOperator(dimension) + annihilationOperator(dimension), elec_dim).toarray()
    recorder = StateObservableRecorder([np.identity(system_dimension, dtype = np.complex), light_matter, 
                                        photon_number, photon_position])

    # This implies, that the norm of the Hamiltonian in the argument is 0.5
    # Then with a Taylor expansion up to 4th order
    #dt = 0.8 / np.linalg.norm(system.system_hamiltonian()(0.0))
    dt = 0.001
    Solvers.direct_closed_observable(system.system_hamiltonian(), state_distro.draw(), time, dt, recorder);
    return recorder

min_dimension = 1
max_dimension = 40

recorder_list = []
for dimension in range(min_dimension, max_dimension, 2):
    print(dimension)
    recorder = dimension_study(dimension)
    recorder_list.append(recorder)
    
datalist = []

for recorder in recorder_list:
    tmp = []
    tmp.append(recorder.expval(0))
    tmp.append(recorder.expval(1))
    tmp.append(recorder.expval(2))
    tmp.append(recorder.expval(3))
    datalist.append(tmp)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

min_dimension = 1
max_dimension = 40

cNorm  = matplotlib.colors.Normalize(vmin=min_dimension, vmax=max_dimension)
scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=plt.cm.jet)
matplotlib.rc('text', usetex=True)
plt.rcParams.update({'font.size': 22})

#energy_data = np.loadtxt("dimension_study_energy.csv", delimiter = ",")
#number_data = np.loadtxt("dimension_study_number.csv", delimiter = ",")
#displacement_data = np.loadtxt("dimension_study_displacement.csv", delimiter = ",")

fig1, ax1 = plt.subplots(ncols = 1, figsize = (5, 4))
fig2, ax2 = plt.subplots(ncols = 1, figsize = (5, 4))
fig3, ax3 = plt.subplots(ncols = 1, figsize = (5, 4))
size = energy_data.shape[1]
for i, dimension in enumerate(range(max_dimension, min_dimension, -2)):
    xf = np.linspace(0.0, 200.0, size)
    #ax1.plot(xf, recorder_list[-i-1].expval(0), color = scalarMap.to_rgba(dimension), alpha = 0.25)
    ax1.plot(xf, energy_data[-i-1], color = scalarMap.to_rgba(dimension), alpha = 0.25)
    ax2.plot(xf, displacement_data[-i-1], color = scalarMap.to_rgba(dimension), alpha = 0.25)
    ax3.plot(xf, number_data[-i-1], color = scalarMap.to_rgba(dimension), alpha = 0.25)
    ax1.set_xlim(left = 0.0, right = 200.0)
    ax2.set_xlim(left = 0.0, right = 200.0)    
    ax3.set_xlim(left = 0.0, right = 200.0)
    ax1.set_ylabel("$E_{\mathrm{el}}$")
    ax2.set_ylabel("$X$")
    ax3.set_ylabel("$N$")
    xlabel = "t"
    ax1.set_xlabel(xlabel)
    ax2.set_xlabel(xlabel)    
    ax3.set_xlabel(xlabel)

from mpl_toolkits.axes_grid1 import make_axes_locatable

for fig, ax in zip([fig1, fig2, fig2], [ax1, ax2, ax3]):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    clb = fig.colorbar(scalarMap, cax=cax, orientation='vertical')
    clb.ax.set_title('$N_{\mathrm{max}}$', pad = 15.0)
    
fig1.savefig("dimension_study_energy.pdf", bbox_inches = "tight")
fig2.savefig("dimension_study_number.pdf", bbox_inches = "tight")
fig3.savefig("dimension_study_displacement.pdf", bbox_inches = "tight")