#include "direct_solver.hpp"

#include "HSpaceDistribution.hpp"
#include "Lindbladian.hpp"
#include "Operators.hpp"

std::vector<mat_t> density_matrix_direct(const Lindbladian & system,
					 const HSpaceDistribution & state_distro,
					 double time, double dt,
					 const mat_t & observable) {
  std::vector<mat_t> density_matrices;
  mat_t density_matrix = state_distro.density_matrix();
  int time_steps = static_cast<int>(time / dt);
  vec_t density_vector = unstack_matrix(density_matrix);
  int dimension = density_matrix.cols();
  auto super_hamiltonian = system.superoperator();

  double t = 0.0;
  for (int j = 0; j < time_steps; ++j, t += dt) {
    std::cout << j << std::endl;
    density_vector = super_hamiltonian->propagate(t, dt, density_vector);
    mat_t density_matrix = restack_vector(density_vector, dimension);
    density_matrices.push_back(density_matrix);
  }
  return density_matrices;
}

Eigen::VectorXd observable_direct(const Lindbladian & system,
				  const HSpaceDistribution & state_distro,
				  double time, double dt,
				  const mat_t & observable) {
  mat_t density_matrix = state_distro.density_matrix();
  int time_steps = static_cast<int>(time / dt);
  int dimension = density_matrix.cols();
  vec_t density_vector = unstack_matrix(density_matrix);
  auto super_hamiltonian = system.superoperator();

  Eigen::VectorXd n_ensemble = Eigen::VectorXd::Zero(time_steps);
  double t = 0.0;
  for (int j = 0; j < time_steps; ++j, t += dt) {
    std::cout << j << std::endl;
    density_vector = super_hamiltonian->propagate(t, dt, density_vector);
    mat_t density_matrix = restack_vector(density_vector, dimension);
    n_ensemble(j) = (observable * density_matrix).trace().real();
  }
  return n_ensemble;
}

Eigen::VectorXd
two_time_correlation_direct(const Lindbladian & system,
			    const HSpaceDistribution & state_distro,
			    double t0, double t1,
			    double dt,
			    const mat_t & A,
			    const mat_t & B) {
  mat_t density_matrix = state_distro.density_matrix();
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
    mat_t density_matrix = restack_vector(density_vector, dimension);
    n_ensemble(j) = (A * density_matrix).trace().real();
  }
  return n_ensemble;
}
