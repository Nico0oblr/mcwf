#include "direct_solver.hpp"

#include "HSpaceDistribution.hpp"
#include "Lindbladian.hpp"
#include "Operators.hpp"

void observable_direct(const Lindbladian & system,
		       const HSpaceDistribution & state_distro,
		       double time, double dt,
		       RecorderHost<calc_mat_t> & recorder) {
  LOG(logINFO) << "performing direct solution" << std::endl;
  calc_mat_t density_matrix = state_distro.density_matrix();
  int time_steps = static_cast<int>(time / dt);
  int dimension = density_matrix.cols();
  vec_t density_vector = unstack_matrix(density_matrix);
  auto super_hamiltonian = system.superoperator();

  double t = 0.0;
  for (int j = 0; j < time_steps; ++j, t += dt) {
    density_vector = super_hamiltonian->propagate(t, dt, density_vector);
    density_matrix = calc_mat_t(restack_vector(density_vector, dimension));
    recorder.record(density_matrix);
  }
}

Eigen::VectorXd
two_time_correlation_direct(const Lindbladian & system,
			    const HSpaceDistribution & state_distro,
			    double t0, double t1,
			    double dt,
			    const calc_mat_t & A,
			    const calc_mat_t & B) {
  calc_mat_t density_matrix = state_distro.density_matrix();
  int time_steps0 = static_cast<int>(t0 / dt);
  int time_steps1 = static_cast<int>((t1 - t0) / dt);
  int dimension = density_matrix.cols();
  vec_t density_vector = unstack_matrix(density_matrix);
  auto super_hamiltonian = system.superoperator();

  Eigen::VectorXd n_ensemble = Eigen::VectorXd::Zero(time_steps1);
  double t = 0.0;
  for (int j = 0; j < time_steps0; ++j, t += dt) {
    density_vector = super_hamiltonian->propagate(t, dt, density_vector);
  }
  density_vector = unstack_matrix(B * restack_vector(density_vector, dimension));
  
  for (int j = 0; j < time_steps1; ++j, t += dt) {
    density_vector = super_hamiltonian->propagate(t, dt, density_vector);
    density_matrix = calc_mat_t(restack_vector(density_vector, dimension));
    n_ensemble(j) = calc_mat_t(A * density_matrix).trace().real();
  }
  return n_ensemble;
}
