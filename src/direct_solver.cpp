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
  spmat_t super_liouvillian = system.superoperator() * dt;
  spmat_t propagator = matrix_exponential_taylor(super_liouvillian);
  
  for (int j = 0; j < time_steps; ++j) {
    std::cout << j << std::endl;
    density_vector = propagator * density_vector;
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
  spmat_t super_liouvillian = system.superoperator() * dt;
  spmat_t propagator = matrix_exponential_taylor(super_liouvillian);
  vec_t density_vector = unstack_matrix(density_matrix);
  std::cout << dmat_sparsity(propagator) << std::endl;

  Eigen::VectorXd n_ensemble = Eigen::VectorXd::Zero(time_steps);
  for (int j = 0; j < time_steps; ++j) {
    std::cout << j << std::endl;
    density_vector = propagator * density_vector;
    mat_t density_matrix = restack_vector(density_vector, dimension);
    n_ensemble(j) = (observable * density_matrix).trace().real();
  }
  return n_ensemble;
}
