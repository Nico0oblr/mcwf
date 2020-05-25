#ifndef LINDBLADIAN_HPP
#define LINDBLADIAN_HPP

#include "Common.hpp"
#include "Hamiltonian.hpp"

class DrivenCavityHamiltonian {
public:
  calc_mat_t operator()(double time) const {
    return time_indep + std::sin(laser_frequency * time) * time_dep;
  }

  DrivenCavityHamiltonian(double cavity_frequency,
			  double claser_frequency,
			  std::complex<double> laser_amplitude,
			  int dimension,
			  int elec_dim = 1)
    : laser_frequency(claser_frequency) {
    calc_mat_t A = annihilationOperator_sp(dimension);
    calc_mat_t A_t = creationOperator_sp(dimension);
    calc_mat_t n = numberOperator_sp(dimension);
    time_indep = cavity_frequency * n;
    time_dep = (A + A_t) * laser_amplitude;
    time_indep = kroneckerProduct(time_indep,
				  calc_mat_t::Identity(elec_dim, elec_dim)).eval();
    time_dep = kroneckerProduct(time_dep,
				calc_mat_t::Identity(elec_dim, elec_dim)).eval();
    std::cout << "cavity dim" << std::endl;
    print_matrix_dim(time_indep);
    print_matrix_dim(time_dep);
    std::cout << "dimension: " << dimension << std::endl;
    std::cout << "elec_dim: " << elec_dim << std::endl;
  }

private:
  double laser_frequency;
  calc_mat_t time_indep;
  calc_mat_t time_dep;
};

struct Lindbladian {

  std::unique_ptr<Hamiltonian<calc_mat_t>> hamiltonian() const;

  Hamiltonian<calc_mat_t> & system_hamiltonian();

  void add_subsystem(const calc_mat_t sub_hamiltonian);

  void calculate_nh_term();

  calc_mat_t operator()(double time, const calc_mat_t & density_matrix) const;

  std::unique_ptr<Hamiltonian<calc_mat_t>> superoperator() const;
  
  std::unique_ptr<Hamiltonian<calc_mat_t>> m_system_hamiltonian;
  std::vector<calc_mat_t> m_lindblad_operators;
  std::vector<scalar_t> m_lindblad_amplitudes;
  calc_mat_t m_nh_term;

  Lindbladian(const Hamiltonian<calc_mat_t> & system_hamiltonian,
	      const std::vector<calc_mat_t> & lindblad_operators,
	      const std::vector<scalar_t> & lindblad_amplitudes);

  /*
    Constructor for nondiagonal Lindblad equation, that must first
    be diagonalized.
   */
  Lindbladian(const Hamiltonian<calc_mat_t> & system_hamiltonian,
	      const std::vector<calc_mat_t> & lindblad_operators,
	      const Eigen::MatrixXd & lindblad_matrix);

  Lindbladian(const Lindbladian & other);
};

scalar_t bose_distribution(double temperature,
			   double frequency);

Lindbladian thermalCavity(double n_b,
			  double frequency,
			  scalar_t gamma,
			  int dimension);

Lindbladian drivenCavity(double n_b,
			 double frequency,
			 double laser_frequency,
			 scalar_t gamma,
			 std::complex<double> amplitude,
			 int dimension,
			 int elec_dim = 1);

#endif /* LINDBLADIAN_HPP */
