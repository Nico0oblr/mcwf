#include "CavityHamiltonianV2.hpp"
#include "Lindbladian.hpp"

double CavityHamiltonianV2::driving_term(double t) const {
  return -std::cos(m_laser_frequency * t) / m_laser_frequency;
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
   m_laser_frequency(laser_frequency),
   m_elec_dim(elec_dim), m_order(6) {
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

  calc_mat_t X = -1.0i * (tensor_identity(photon_energy, elec_dim)
			  + light_matter);
  calc_mat_t X_eff = -1.0i * photon_energy;
  m_e_X = matrix_exponential(X * dt);
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
  double dct = driving_term(t + dt) - driving_term(t);
  mat_t temp = m_Y.exp(dct)
    * m_first_comm.exp(dct * dt)
    * m_second_comm.exp(dct * dct * dt)
    * m_third_comm.exp(dct * dt * dt)
    * m_fourth_comm.exp(dct * dt * dt * dt)
    * m_fifth_comm.exp(dct * dt * dt * dt * dt);
  return m_e_X * tensor_identity(temp, m_elec_dim);
}
    
vec_t CavityHamiltonianV2::propagate(double t, double dt,
				   const vec_t & state) {
  double dct = driving_term(t + dt) - driving_term(t);

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

void CavityHamiltonianV2::set_order(int order) {
  m_order = order;
}

CavityHamiltonianV2* CavityHamiltonianV2::clone_impl() const
{return new CavityHamiltonianV2(*this);};
