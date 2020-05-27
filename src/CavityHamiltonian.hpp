#ifndef CAVITYHAMILONIAN_HPP
#define CAVITYHAMILONIAN_HPP

#include "Hamiltonian.hpp"

class CavityHamiltonian : public TimeDependentHamiltonian<calc_mat_t> {
  using Base = TimeDependentHamiltonian<calc_mat_t>;
  
  double m_laser_frequency;
  calc_mat_t m_e_X;
  calc_mat_t m_Y;
  calc_mat_t m_first_comm;
  calc_mat_t m_second_comm;
  calc_mat_t m_third_comm;
  calc_mat_t m_fourth_comm;
  double m_Y_norm;
  double m_first_norm;
  double m_second_norm;
  double m_third_norm;
  double m_fourth_norm;

public:
  double driving_term(double t) const;

  CavityHamiltonian(double frequency, double laser_frequency,
		    double laser_amplitude, int elec_dim, int dimension,
		    const calc_mat_t & light_matter,
		    double dt);
  
  calc_mat_t propagator(double t, double dt) override;
  
  vec_t propagate(double t, double dt, const vec_t & state) override;
  
};

#endif /* CAVITYHAMILONIAN_HPP */
