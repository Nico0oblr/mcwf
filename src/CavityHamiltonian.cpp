#include "CavityHamiltonian.hpp"
#include "Lindbladian.hpp"

double CavityHamiltonian::driving_term(double t) const {
  return -std::cos(m_laser_frequency * t) / m_laser_frequency;
}

CavityHamiltonian::CavityHamiltonian(double frequency, double laser_frequency,
				     double laser_amplitude, int elec_dim,
				     int dimension,
				     const calc_mat_t & light_matter,
				     double dt)
  :Base(DrivenCavityHamiltonian(frequency, laser_frequency,
				laser_amplitude, dimension, elec_dim),
	dimension * elec_dim) ,
   m_laser_frequency(laser_frequency) {
  calc_mat_t photon_energy = frequency * numberOperator(dimension);
  calc_mat_t  driving_term = laser_amplitude
    * (annihilationOperator(dimension) + creationOperator(dimension));
    
  photon_energy = tensor_identity(photon_energy, elec_dim).eval();
  driving_term = tensor_identity(driving_term, elec_dim).eval();

  calc_mat_t X_eff = -1.0i * photon_energy;
  calc_mat_t X = -1.0i * (photon_energy + light_matter);
  m_e_X = matrix_exponential(X * dt);
  m_Y = -1.0i * driving_term;
  m_first_comm = - 0.5 * commutator(X_eff, m_Y);
  m_second_comm = 1.0 / 3.0 * commutator(m_Y, commutator(X_eff, m_Y));
  m_third_comm = 1.0 / 6.0 * commutator(X_eff, commutator(X_eff, m_Y));
  // Only this commutator remains in this order
  m_fourth_comm = -1.0 / 24.0 * commutator(commutator(commutator(X_eff, m_Y),
						      X_eff), X_eff);

  m_Y_norm = m_Y.norm();
  m_first_norm = m_first_comm.norm();
  m_second_norm = m_first_comm.norm();
  m_third_norm = m_first_comm.norm();
  m_fourth_norm = m_first_comm.norm();
}

calc_mat_t CavityHamiltonian::propagator(double t, double dt) {
  double dct = driving_term(t + dt) - driving_term(t);
  return m_e_X * matrix_exponential_taylor(m_Y * dct, 1) * 
    matrix_exponential_taylor(m_first_comm * dct * dt, 1) * 
    matrix_exponential_taylor(m_second_comm * dct * dct * dt + m_third_comm * dct * dt * dt, 1);
}
    
vec_t CavityHamiltonian::propagate(double t, double dt,
				   const vec_t & state) {
  // double dct = driving_term(t + dt) - driving_term(t);
  double dct = std::sin(m_laser_frequency * t) * dt;
  vec_t tmp = apply_matrix_exponential_taylor(m_second_comm * dct * dct * dt + m_third_comm * dct * dt * dt, state, 1);
  tmp = apply_matrix_exponential_taylor(m_first_comm * dct * dt, tmp, 1);
  tmp = apply_matrix_exponential_taylor(m_Y * dct, tmp, 3);
  return m_e_X * tmp;
}
