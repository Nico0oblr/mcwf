#include "Common.hpp"

#ifndef LINDBLADIAN_HPP
#define LINDBLADIAN_HPP

struct Lindbladian {

  bool commutes();

  mat_t hamiltonian() const;

  void add_subsystem(const mat_t sub_hamiltonian);

  void calculate_nh_term();

  mat_t operator()(const mat_t & density_matrix) const;

  spmat_t superoperator() const;
  
  mat_t m_system_hamiltonian;
  std::vector<mat_t> m_lindblad_operators;
  std::vector<scalar_t> m_lindblad_amplitudes;
  mat_t m_nh_term;

  Lindbladian(const mat_t & system_hamiltonian,
	      const std::vector<mat_t> & lindblad_operators,
	      const std::vector<scalar_t> & lindblad_amplitudes);

  /*
    Constructor for nondiagonal Lindblad equation, that must first
    be diagonalized.
   */
  Lindbladian(const mat_t & system_hamiltonian,
	      const std::vector<mat_t> & lindblad_operators,
	      const Eigen::MatrixXd & lindblad_matrix);
};

scalar_t bose_distribution(double temperature,
			   double frequency);

Lindbladian thermalCavity(double n_b,
			  double frequency,
			  scalar_t gamma,
			  int dimension);

Lindbladian drivenCavity(double n_b,
			 double frequency,
			 scalar_t gamma,
			 std::complex<double> amplitude,
			 int dimension);

#endif /* LINDBLADIAN_HPP */
