#ifndef LINDBLADIAN_HPP
#define LINDBLADIAN_HPP

#include "Common.hpp"
#include "Hamiltonian.hpp"
#include "LinearOperator.hpp"

calc_mat_t lindblad_term(const std::vector<calc_mat_t> & lindblad_operators,
			 const std::vector<scalar_t> & lindblad_amplitudes);

lo_ptr lindblad_term(const std::vector<lo_ptr> & lindblad_operators,
		     const std::vector<scalar_t> & lindblad_amplitudes);

class DrivenCavityHamiltonian {
public:
  lo_ptr operator()(double time) const {
    return scale_rhs_and_add(*time_indep, *time_dep,
			     std::sin(laser_frequency * time));
    // return time_indep + std::sin(laser_frequency * time) * time_dep;
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
    calc_mat_t _time_indep = cavity_frequency * n;
    calc_mat_t _time_dep = (A + A_t) * laser_amplitude;
    time_indep = kroneckerOperator_IDRHS(_time_indep, elec_dim);
    time_dep = kroneckerOperator_IDRHS(_time_dep, elec_dim);
  }

  DrivenCavityHamiltonian(const DrivenCavityHamiltonian & other)
    :laser_frequency(other.laser_frequency),
     time_indep(other.time_indep->clone()),
     time_dep(other.time_dep->clone()) {}

private:
  double laser_frequency;
  lo_ptr time_indep;
  lo_ptr time_dep;
};

struct Lindbladian {

  virtual std::unique_ptr<Hamiltonian<calc_mat_t>> hamiltonian() const;

  Hamiltonian<calc_mat_t> & system_hamiltonian();

  void add_subsystem(const calc_mat_t sub_hamiltonian);


  calc_mat_t operator()(double time, const calc_mat_t & density_matrix) const;

  std::unique_ptr<Hamiltonian<calc_mat_t>> superoperator() const;
  
  std::unique_ptr<Hamiltonian<calc_mat_t>> m_system_hamiltonian;
  std::vector<lo_ptr> m_lindblad_operators;
  std::vector<scalar_t> m_lindblad_amplitudes;

  Lindbladian(const Hamiltonian<calc_mat_t> & system_hamiltonian,
	      const std::vector<lo_ptr> & lindblad_operators,
	      const std::vector<scalar_t> & lindblad_amplitudes);
  
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

  virtual ~Lindbladian();

protected:
  Lindbladian(const Hamiltonian<calc_mat_t> & system_hamiltonian);
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
