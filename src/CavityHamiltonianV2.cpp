#include "CavityHamiltonianV2.hpp"
#include "Lindbladian.hpp"
#include "PadeExponential.hpp"
#include "HubbardModel.hpp"


spmat_t Hubbard_site_operator(int photon_dimension,
			    int sites,
			    int site,
			    double coupling,
			    double hopping,
			    double hubbardU,
			    const spmat_t & proj) {
  using namespace HubbardOperators;
  assert(site < sites);
  int dimension = std::pow(4, sites);
  spmat_t hopping_term = spmat_t::Zero(dimension, dimension);
  spmat_t onsite_term = spmat_t::Zero(dimension, dimension);
  int next_site = (site + 1) % sites;
  
  hopping_term += n_th_subsystem_sp(c_up_t(), site, sites)
    * n_th_subsystem_sp(c_up(), next_site, sites);
  hopping_term += n_th_subsystem_sp(c_down_t(), site, sites)
    * n_th_subsystem_sp(c_down(), next_site, sites);
  onsite_term += n_th_subsystem_sp(n_up(), site, sites)
    * n_th_subsystem_sp(n_down(), site, sites);

  mat_t argument = 1.0i * coupling * (creationOperator(photon_dimension)
				      + annihilationOperator(photon_dimension));
  spmat_t e_iA = matrix_exponential(argument);

  onsite_term *= hubbardU;
  hopping_term *= hopping;
  hopping_term = proj * hopping_term * proj.adjoint();
  onsite_term = proj * onsite_term * proj.adjoint();

  hopping_term = Eigen::kroneckerProduct(e_iA, hopping_term).eval();
  onsite_term = tensor_identity_LHS(onsite_term, photon_dimension).eval();
  spmat_t hopping_term_adj = hopping_term.adjoint();
  return hopping_term + hopping_term_adj + onsite_term;
}


spmat_t ST_decomp_exp(int photon_dimension,
		      int sites,
		      double coupling,
		      double hopping,
		      double hubbardU,
		      const spmat_t & proj,
		      double dt) {
  int system_dim = photon_dimension * proj.rows();
  spmat_t evens = spmat_t::Identity(system_dim, system_dim);
  spmat_t odds = spmat_t::Identity(system_dim, system_dim);

  for (int i = 0; i < sites; ++i) {
    if (i % 2 == 0) {
    evens = evens * expm(-1.0i * dt
			 * Hubbard_site_operator(photon_dimension, sites, i,
						 coupling, hopping,
						 hubbardU, proj));
    } else {
    odds = odds * expm(-0.5i * dt
		       * Hubbard_site_operator(photon_dimension, sites, i,
					       coupling, hopping,
					       hubbardU, proj));
    }
  }
  LOG_VAR(sparsity(odds));
  LOG_VAR(sparsity(evens));
  LOG_VAR(sparsity(spmat_t(odds * evens * odds)));
  return odds * evens * odds;
}

vec_t ST_decomp_exp_apply(int photon_dimension,
			    int sites,
			    double coupling,
			    double hopping,
			    double hubbardU,
			    const spmat_t & proj,
			    double dt, const vec_t & state) {
  int system_dim = photon_dimension * proj.rows();
  std::vector<spmat_t> evens;
  std::vector<spmat_t> odds;

  for (int i = 0; i < sites; ++i) {
    if (i % 2 == 0) {
      evens.push_back(expm(-1.0i * dt
			   * Hubbard_site_operator(photon_dimension, sites, i,
						   coupling, hopping,
						   hubbardU, proj)));
    } else {
      odds.push_back(expm(-0.5i * dt
			  * Hubbard_site_operator(photon_dimension, sites, i,
						  coupling, hopping,
						  hubbardU, proj)));
    }
  }
  vec_t vec = state;
  for (const auto & mat : odds) vec = mat * vec;
  for (const auto & mat : evens) vec = mat * vec;
  for (const auto & mat : odds) vec = mat * vec;
  return vec;
}

double CavityHamiltonianV2::driving_term(double t) const {
  return -std::cos(m_laser_frequency * t) / m_laser_frequency;
}

void DiagonalizedMatrix::set_from(const calc_mat_t & matrix) {
  self = matrix;
  Eigen::ComplexEigenSolver<mat_t> solver(matrix, true);
  D = solver.eigenvalues();
  V = solver.eigenvectors();
  Eigen::PartialPivLU<mat_t> inverter(V);
  V_inv = inverter.inverse();
}

DiagonalizedMatrix & DiagonalizedMatrix::operator=(const calc_mat_t & other) {
  set_from(other);
  return *this;
}

mat_t DiagonalizedMatrix::exp(std::complex<double> factor) const {
  return V * (D * factor).array().exp().matrix().asDiagonal() * V_inv;
}

vec_t DiagonalizedMatrix::exp_apply(std::complex<double> factor, const vec_t & vec) {
  return apply_tensor_id(exp(factor), vec);
}

const calc_mat_t & DiagonalizedMatrix::operator()() const {
  return self;
}

CavityHamiltonianV2::CavityHamiltonianV2(double frequency,
					 double laser_frequency,
					 double laser_amplitude, int elec_dim,
					 int dimension,
					 const calc_mat_t & light_matter,
					 double dt, double gamma, double n_b)
  :Base(DrivenCavityHamiltonian(frequency, laser_frequency,
				laser_amplitude, dimension, elec_dim),
	dimension * elec_dim) ,
   m_frequency(frequency),
   m_laser_frequency(laser_frequency),
   m_laser_amplitude(laser_amplitude),
   m_elec_dim(elec_dim),
   m_dimension(dimension),
   m_light_matter(light_matter),
   m_dt(dt),
   m_gamma(gamma),
   m_n_b(n_b),
   m_order(6) {
  calc_mat_t photon_energy = frequency * numberOperator(dimension);
  calc_mat_t  driving_term = laser_amplitude
    * (annihilationOperator(dimension) + creationOperator(dimension));

  if (gamma > 0.0) {
    calc_mat_t dissipation = lindblad_term({annihilationOperator(dimension),
					    creationOperator(dimension)},
      {gamma * (1.0 + n_b), gamma * n_b});
    photon_energy -= 0.5i * dissipation;
  }
  
  // photon_energy = tensor_identity(photon_energy, elec_dim).eval();
  // driving_term = tensor_identity(driving_term, elec_dim).eval();

  m_X = -1.0i * (tensor_identity(photon_energy, elec_dim)
		 + light_matter);
  calc_mat_t X_eff = -1.0i * photon_energy;
  m_e_X = expm(m_X * dt);
  
  /*m_e_X = expm(-0.5i * tensor_identity(photon_energy, elec_dim) * dt)
    * expm(-1.0i * light_matter * dt)
    * expm(-0.5i * tensor_identity(photon_energy, elec_dim) * dt);*/
  m_Y = -1.0i * driving_term;
  m_first_comm = - 0.5 * commutator(X_eff, m_Y());
  m_second_comm = 1.0 / 3.0 * commutator(m_Y(), commutator(X_eff, m_Y()));
  m_third_comm = 1.0 / 6.0 * commutator(X_eff, commutator(X_eff, m_Y()));
  // Only this commutator remains in this order
  m_fourth_comm = -1.0 / 24.0 * commutator(commutator(commutator(X_eff, m_Y()), X_eff), X_eff);
  m_fifth_comm = -1.0 / 120.0 * commutator(commutator(commutator
						      (commutator(X_eff, m_Y()),
						       X_eff), X_eff), X_eff);

  m_Y_norm = m_Y().norm();
  m_first_norm = m_first_comm().norm();
  m_second_norm = m_second_comm().norm();
  m_third_norm = m_third_comm().norm();
  m_fourth_norm = m_fourth_comm().norm();
  m_fifth_norm = m_fifth_comm().norm();	    
}

calc_mat_t CavityHamiltonianV2::propagator(double t, double dt) {
  double dct = std::sin(m_laser_frequency * t) * dt;
  mat_t temp = m_Y.exp(dct)
    * m_first_comm.exp(dct * dt)
    * m_second_comm.exp(dct * dct * dt)
    * m_third_comm.exp(dct * dt * dt)
    * m_fourth_comm.exp(dct * dt * dt * dt)
    * m_fifth_comm.exp(dct * dt * dt * dt * dt);
  return m_e_X * tensor_identity(temp, m_elec_dim);
}
    
vec_t CavityHamiltonianV2::BCH_propagate(double t, double dt,
					 const vec_t & state) {
  double dct = std::sin(m_laser_frequency * t) * dt;

  // The commutator [m_second_comm, m_third_comm]
  // is assumed to be approximately zero here
  // Better since it keeps everything unitary instead of power series
  // calc_mat_t fac1 = matrix_exponential_taylor(m_second_comm() * dct * dct * dt + m_third_comm() * dct * dt * dt, 3);

  /*vec_t tmp = m_fifth_comm.exp_apply(dct * dt * dt * dt * dt, state);
  tmp = m_fourth_comm.exp_apply(dct * dt * dt * dt, tmp);

  tmp = m_third_comm.exp_apply(dct * dt * dt, tmp);
  tmp = m_second_comm.exp_apply(dct * dct * dt, tmp);
  // tmp = apply_tensor_id(fac1, tmp);
  tmp = m_first_comm.exp_apply(dct * dt, tmp);
  tmp = m_Y.exp_apply(dct, tmp);
  tmp = m_e_X * tmp;*/

  mat_t temp = mat_t::Identity(m_Y().rows(), m_Y().cols());

  if (m_order > 0) temp *= m_Y.exp(dct);
  if (m_order > 1) temp *= m_first_comm.exp(dct * dt);
  if (m_order > 2) temp *= m_second_comm.exp(dct * dct * dt);
  if (m_order > 3) temp *= m_third_comm.exp(dct * dt * dt);
  if (m_order > 4) temp *= m_fourth_comm.exp(dct * dt * dt * dt);
  if (m_order > 5) temp *= m_fifth_comm.exp(dct * dt * dt * dt * dt);
  return m_e_X * apply_tensor_id(temp, state);
}

vec_t CavityHamiltonianV2::ST_propagate(double t, double dt,
					const vec_t & state) {
  double dct = std::sin(m_laser_frequency * t) * dt;
  spmat_t Y_half = m_Y.exp(0.5 * dct);
  vec_t out = apply_tensor_id(Y_half, state);
  out = m_e_X * out;
  return apply_tensor_id(Y_half, out);
}

vec_t CavityHamiltonianV2::propagate(double t, double dt, const vec_t & state) {
  return ST_propagate(t, dt, state);
}

void CavityHamiltonianV2::set_order(int order) {
  m_order = order;
}

CavityHamiltonianV2* CavityHamiltonianV2::clone_impl() const
{return new CavityHamiltonianV2(*this);}

CavityLindbladian::CavityLindbladian(double frequency, double laser_frequency,
				     double laser_amplitude, int elec_dim, int dimension,
				     const calc_mat_t & light_matter,
				     double dt, double gamma, double n_b)
  :Base(CavityHamiltonianV2(frequency, laser_frequency,
			    laser_amplitude, elec_dim, dimension,
			    light_matter, dt, 0.0, 0.0)),
   mcwf_hamiltonian(CavityHamiltonianV2(frequency, laser_frequency,
					laser_amplitude, elec_dim, dimension,
					light_matter, dt, gamma, n_b)) {
  assert(elec_dim * dimension == light_matter.rows());
  calc_mat_t A = annihilationOperator_sp(dimension);
  calc_mat_t A_t = creationOperator_sp(dimension);
  A = kroneckerProduct(A, calc_mat_t::Identity(elec_dim, elec_dim)).eval();
  A_t = kroneckerProduct(A_t, calc_mat_t::Identity(elec_dim, elec_dim)).eval();
  Base::m_lindblad_operators = {A, A_t};
  Base::m_lindblad_amplitudes = {gamma * (1.0 + n_b), gamma * n_b};
}
     
std::unique_ptr<Hamiltonian<calc_mat_t>> CavityLindbladian::hamiltonian() const {
  return mcwf_hamiltonian.clone();
}

const CavityHamiltonianV2 & CavityLindbladian::hamiltonian_expl() const {
  return mcwf_hamiltonian;
}

CavityHamiltonianV2 & CavityLindbladian::hamiltonian_expl() {
  return mcwf_hamiltonian;
}
