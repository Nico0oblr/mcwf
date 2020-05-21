#include <fstream>
#include <omp.h>

#include "argparse.hpp"
#include "Common.hpp"
#include "Operators.hpp"
#include "HSpaceDistribution.hpp"
#include "Lindbladian.hpp"
#include "mcwf_functions.hpp"
#include "runge_kutta_solver.hpp"
#include "direct_solver.hpp"
#include "HubbardModel.hpp"
#include "toy_spin_model.hpp"
#include "direct_closed_solver.hpp"
#include "tests.hpp"

int main(int argc, char ** argv) {
  std::cout << "Starting program" << std::endl;
  superoperator_test(10);
  hubbard_tests();
  function_tests();
  /*System creation*/
  YamlOrCMD parser(argv, argc, "config.yaml");
  int num_threads = parser.parse<int>("num_threads");
  omp_set_dynamic(0);
  omp_set_num_threads(num_threads);
  // Photonic truncation
  int dimension = parser.parse<int>("dimension");
  PrecomputedOperators.precompute(dimension);
  // Number of spin sites
  int sites = parser.parse<int>("sites");
  // Number of realizations for mcwf
  int runs = parser.parse<int>("runs");
  // In photon occupations
  double temperature = parser.parse<double>("temperature");
  // Time step size
  double dt = parser.parse<double>("dt");
  double time = parser.parse<double>("time");
  // Cavity absorption rate
  double gamma = parser.parse<double>("gamma");
  // Light matter coupling
  double coupling = parser.parse<double>("coupling");
  double hubbardU = parser.parse<double>("hubbardU");
  double hopping = parser.parse<double>("hopping");
  double frequency = parser.parse<double>("frequency");
  double laser_frequency = parser.parse<double>("laser_frequency");
  // Laser amplitude
  double laser_amplitude = parser.parse<double>("laser_amplitude");
  bool periodic = parser.parse<bool>("periodic");
  double Jx = parser.parse<double>("Jx");
  double Jy = parser.parse<double>("Jy");
  double Jz = parser.parse<double>("Jz");

  std::string model = parser.parse<std::string>("model");
  mat_t light_matter;
  
  mat_t hubbard_projector = HubbardProjector(sites, sites / 2, sites / 2);
  mat_t hubbard_proj = tensor_identity_LHS(hubbard_projector, dimension);
  int elec_dim = hubbard_projector.rows();
  if (model == "hubbard") {
    light_matter = Hubbard_light_matter(dimension, sites, coupling,
					      hopping, hubbardU, periodic);
    print_matrix_dim(light_matter);
    print_matrix_dim(hubbard_proj);
    light_matter = (hubbard_proj * light_matter * hubbard_proj.adjoint()).eval();
  } else if (model == "heisenberg") {
    mat_t heisenberg = HeisenbergChain(sites, Jx, Jy, Jz, periodic);
    mat_t exchange = exchange_interaction_full(dimension, hubbardU, hopping,
					       frequency,coupling, dimension);
  
    light_matter = Eigen::kroneckerProduct(exchange, heisenberg);
    
  } else if (model == "none") {
    light_matter = mat_t::Zero(dimension, dimension);
    elec_dim = 1;
  } else if (model == "toy_spin") {
    mat_t heisenberg = HeisenbergChain(sites, Jx, Jy, Jz, periodic);
    light_matter = toy_modelize(dimension, heisenberg,
				hubbardU, hopping,
				frequency,coupling);
  }

  Lindbladian system = drivenCavity(temperature,
				    frequency - laser_frequency, gamma,
				    laser_amplitude, dimension);
  std::cout << system.m_system_hamiltonian << std::endl;
  if (sites > 0) {
    std::cout << "light_matter norm: " << light_matter.norm() << std::endl;
    system.add_subsystem(light_matter);
  }
  /*System created. Defining observable*/

  
  std::string observable_name = parser.parse<std::string>("observable");
  mat_t observable;
  if (observable_name == "photon_number") {
    observable = tensor_identity(numberOperator(dimension), elec_dim);
  } else if (observable_name == "photon_position") {
    mat_t A = annihilationOperator(dimension);
    mat_t A_t = creationOperator(dimension);
    observable = tensor_identity(A + A_t, elec_dim);
  } else if (observable_name == "spin_energy") {
    // observable = tensor_identity_LHS(heisenberg, dimension);
    observable = light_matter;
  } else if (observable_name == "total_spin") {
    observable = tensor_identity_LHS(0.5 * pauli_z_total(sites), dimension);
  } else if (observable_name == "total_spin_squared") {
    observable = tensor_identity_LHS(0.25 * pauli_squared_total(sites), dimension);
  } else if (observable_name == "single_spin" && model == "hubbard") {
    observable = 0.5 * tensor_identity_LHS
      (nth_subsystem(HubbardOperators::n_up()
		     - HubbardOperators::n_down(), 0, sites), dimension);
    observable = (hubbard_proj * observable * hubbard_proj.adjoint()).eval();
    std::cout << "observable" << std::endl;
    print_matrix_dim(observable);
    std::cout << "light_matter" << std::endl;
    print_matrix_dim(light_matter);
  } else if (observable_name == "single_spin"
	     && (model == "heisenberg" || model == "toy_spin")) {
    observable = tensor_identity_LHS(0.5 * pauli_z_vector(sites)[0],
				     dimension);
  } else if (observable_name == "alternating_spin") {
    std::vector<mat_t> z_vec = pauli_z_vector(sites);
    double sign = 1.0;
    observable = 0.5 * z_vec[0];
    for (int i = 1; i < z_vec.size(); ++i) {
      sign *= -1.0;
      observable += 0.5 * sign * z_vec[i];
    }

    observable = tensor_identity_LHS(observable, dimension);
  } else {
    assert(false);
  }

  print_matrix_dim(observable);
  print_matrix_dim(light_matter);
  
  /*Defining beginning state distributions*/
  // Vacuum
  /*HSpaceDistribution state_distro({1.0}, {static_cast<int>(temperature)},
    dimension);*/
  HSpaceDistribution state_distro = coherent_photon_state(temperature, dimension);
  // Boundary condition for spins
  int boundary_state = parser.parse<int>("boundary_state");
  assert(boundary_state < elec_dim || sites <= 0 || model == "none");

  if (sites > 0 && (model == "heisenberg") || (model == "toy_spin")) {
    HSpaceDistribution electronic_distro({1.0}, {boundary_state}, elec_dim);
    state_distro += electronic_distro;
  } else if (sites > 0 && model ==  "hubbard") {
    HSpaceDistribution electronic_distro = HubbardNeelState(sites,
							    hubbard_projector);
    state_distro += electronic_distro;
  } 



  /*Correlation function playground*/
  std::string method = parser.parse<std::string>("method");
  /*double time1 = parser.parse<double>("time1");
  if (method == "mcwf") {
    Eigen::MatrixXd n_ensemble = two_time_correlation(system,
						      state_distro,
						      time1, time, dt,
						      runs, observable,
						      observable);
    Eigen::VectorXd averaged = n_ensemble.colwise().mean();
    std::ofstream output("results.csv");
    Eigen::IOFormat fmt(Eigen::StreamPrecision,
			Eigen::DontAlignCols, ", ", ", ", "", "", "", "");
    output << averaged.format(fmt) << std::endl;
    output.close();
    return 0;
  } else if (method == "direct") {
    Eigen::VectorXd averaged = two_time_correlation_direct(system,
							   state_distro,
							   time1, time, dt,
							   observable,
							   observable);
      std::ofstream output("results.csv");
    Eigen::IOFormat fmt(Eigen::StreamPrecision,
			Eigen::DontAlignCols, ", ", ", ", "", "", "", "");
    output << averaged.format(fmt) << std::endl;
    output.close();
    return 0;
    }*/


  
  if (method == "compare") {
    std::vector<mat_t> kutta_dmat = density_matrix_kutta(system, state_distro,
							 time, dt, observable);
    std::vector<mat_t> mcwf_dmat = density_matrix_mcwf(system, state_distro,
						       time, dt,
						       runs);
    std::vector<mat_t> direct_dmat = density_matrix_direct(system, state_distro,
							   time, dt, observable);

    assert(mcwf_dmat.size() == kutta_dmat.size());
    assert(mcwf_dmat.size() == direct_dmat.size());

    for (int i = 0; i < mcwf_dmat.size(); ++i) {
      std::cout << "time step " << i << std::endl;
      std::cout << "mcwf-kutta difference: "
		<< ((mcwf_dmat[i] - kutta_dmat[i])
		    * (mcwf_dmat[i] - kutta_dmat[i])).trace() << std::endl;
      std::cout << "mcwf-direct difference: "
		<< ((mcwf_dmat[i] - direct_dmat[i])
		    * (mcwf_dmat[i] - direct_dmat[i])).trace() << std::endl;
      std::cout << "kutta-direct difference: "
		<< ((kutta_dmat[i] - direct_dmat[i])
		    * (kutta_dmat[i] - direct_dmat[i])).trace() << std::endl;
      std::cout << "==========" << std::endl;
    }
  
    return 0;
  }
  Eigen::VectorXd n_averaged;

  if (method == "mcwf") {
    /*Perform Iteration*/
    Eigen::MatrixXd n_ensemble = observable_calc(system, state_distro,
						 time, dt,
						 runs, observable.sparseView());
    n_averaged = n_ensemble.colwise().mean();
  } else if (method == "runge_kutta") {
    n_averaged = observable_kutta(system, state_distro,
				  time, dt, observable);
  } else if (method == "direct") {
    n_averaged = observable_direct(system, state_distro,
				   time, dt, observable);
  } else if (method == "direct_closed") {
    n_averaged = direct_closed_observable(system.m_system_hamiltonian,
					  state_distro.draw(),
					  time, dt, observable.sparseView());
  } else {
    assert(false);
  }
  /*Average data and write to file*/
  std::ofstream output("results.csv");
  Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "", "");
  output << n_averaged.format(fmt) << std::endl;
  output.close();
}
