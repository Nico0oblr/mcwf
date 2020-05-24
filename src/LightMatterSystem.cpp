#include "LightMatterSystem.hpp"

#include "HubbardModel.hpp"
#include "toy_spin_model.hpp"

LightMatterSystem parse_system(YamlOrCMD & parser) {
  int dimension = parser.parse<int>("dimension");
  PrecomputedOperators.precompute(dimension);
  int sites = parser.parse<int>("sites");
  double temperature = parser.parse<double>("temperature");
  double coupling = parser.parse<double>("coupling");
  double hubbardU = parser.parse<double>("hubbardU");
  double hopping = parser.parse<double>("hopping");
  double frequency = parser.parse<double>("frequency");
  double laser_frequency = parser.parse<double>("laser_frequency");
  double laser_amplitude = parser.parse<double>("laser_amplitude");
  double gamma = parser.parse<double>("gamma");
  bool periodic = parser.parse<bool>("periodic");
  std::string model = parser.parse<std::string>("model");

  mat_t light_matter = mat_t::Zero(dimension, dimension);
  int elec_dim = 1;
  mat_t projector = mat_t::Identity(elec_dim, elec_dim);
  if (model == "hubbard") {
    projector = HubbardProjector(sites, sites / 2, sites / 2);
    mat_t hubbard_proj = tensor_identity_LHS(projector, dimension);  
    light_matter = Hubbard_light_matter(dimension, sites, coupling,
					hopping, hubbardU, periodic);
    light_matter = (hubbard_proj * light_matter * hubbard_proj.adjoint()).eval();
    elec_dim = projector.rows();
  } else if (model == "heisenberg") {
    double Jx = parser.parse<double>("Jx");
    double Jy = parser.parse<double>("Jy");
    double Jz = parser.parse<double>("Jz");
    mat_t heisenberg = HeisenbergChain(sites, Jx, Jy, Jz, periodic);
    mat_t exchange = exchange_interaction_full(dimension, hubbardU, hopping,
					       frequency, coupling, dimension);
    light_matter = Eigen::kroneckerProduct(exchange, heisenberg);
    elec_dim = heisenberg.cols();
    projector = mat_t::Identity(elec_dim, elec_dim);
  } else if (model == "toy_spin") {
    double Jx = parser.parse<double>("Jx");
    double Jy = parser.parse<double>("Jy");
    double Jz = parser.parse<double>("Jz");
    mat_t heisenberg = HeisenbergChain(sites, Jx, Jy, Jz, periodic);
    light_matter = toy_modelize(dimension, heisenberg,
				hubbardU, hopping,
				frequency,coupling);
    elec_dim = heisenberg.cols();
    projector = mat_t::Identity(elec_dim, elec_dim);
  }

  Lindbladian system = drivenCavity(temperature,
				    frequency, laser_frequency, gamma,
				    laser_amplitude, dimension, elec_dim);
  
  if (sites > 0) {
    system.m_system_hamiltonian->add(calc_mat_t(light_matter));
  }

  return LightMatterSystem {system, light_matter, projector,
      dimension, sites, temperature, model};
}

mat_t parse_observable(const LightMatterSystem & lms,
		       std::string observable_name) {
  if (observable_name == "photon_number") {
    return tensor_identity(numberOperator(lms.dimension), lms.elec_dim());
  } else if (observable_name == "photon_position") {
    mat_t A = annihilationOperator(lms.dimension);
    mat_t A_t = creationOperator(lms.dimension);
    return tensor_identity<mat_t>(A + A_t, lms.elec_dim());
  } else if (observable_name == "spin_energy") {
    return lms.light_matter;
  } else if (observable_name == "total_spin") {
    return tensor_identity_LHS<mat_t>(0.5 * pauli_z_total(lms.sites), lms.dimension);
  } else if (observable_name == "total_spin_squared") {
    return tensor_identity_LHS<mat_t>(0.25 * pauli_squared_total(lms.sites), lms.dimension);
  } else if ((observable_name == "single_spin") && (lms.model == "hubbard")) {
    return tensor_identity_LHS<mat_t>
      (lms.projector * nth_subsystem(HubbardOperators::n_up()
				     - HubbardOperators::n_down(), 0, lms.sites)
       * lms.projector.adjoint() , lms.dimension);
  } else if (observable_name == "doublet" && lms.model == "hubbard") {
    return tensor_identity_LHS<mat_t>
      (lms.projector * nth_subsystem(HubbardOperators::n_up()
				     * HubbardOperators::n_down(), 0, lms.sites)
       * lms.projector.adjoint(), lms.dimension);
  } else if ((observable_name == "single_spin")
	     && ((lms.model == "heisenberg") || (lms.model == "toy_spin"))) {
    return tensor_identity_LHS<mat_t>(0.5 * pauli_z_vector(lms.sites)[0],
					    lms.dimension);
  } else if (observable_name == "alternating_spin") {
    std::vector<mat_t> z_vec = pauli_z_vector(lms.sites);
    double sign = 1.0;
    mat_t observable = 0.5 * z_vec[0];
    for (size_type i = 1; i < z_vec.size(); ++i) {
      sign *= -1.0;
      observable += 0.5 * sign * z_vec[i];
    }

    return tensor_identity_LHS(observable, lms.dimension);
  } else {
    assert(false && "Observable name not found");
  }
}

HSpaceDistribution parse_initial_distribution(const LightMatterSystem & lms,
					      YamlOrCMD & parser) {
  HSpaceDistribution state_distro = coherent_photon_state(lms.temperature,
							  lms.dimension);
  if ((lms.sites > 0) && ((lms.model == "heisenberg")
			  || (lms.model == "toy_spin"))) {
    int boundary_state = parser.parse<int>("boundary_state");
    assert((boundary_state < lms.elec_dim()) || (lms.sites <= 0)
	   || (lms.model == "none"));
    HSpaceDistribution electronic_distro({1.0}, {boundary_state},
					 lms.elec_dim());
    state_distro += electronic_distro;
  } else if ((lms.sites > 0) && (lms.model ==  "hubbard")) {
    /*double factor = - (hubbardU + std::sqrt(16.0 * hopping * hopping + hubbardU * hubbardU)) / (4.0 * hopping);
      Eigen::Matrix<double, 4, 1> tmp(1, factor, factor, 1);
      tmp /= tmp.norm();
      HSpaceDistribution electronic_distro({1.0}, {tmp});*/
    HSpaceDistribution electronic_distro = HubbardNeelState(lms.sites,
							    lms.projector);
    state_distro += electronic_distro;
  }
  return state_distro;
}

