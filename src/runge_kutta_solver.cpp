#include "runge_kutta_solver.hpp"

#include "Operators.hpp"
#include "HSpaceDistribution.hpp"
#include "Lindbladian.hpp"
#include "runge_kutta.hpp"

Eigen::VectorXd observable_kutta(const Lindbladian & system,
				 const HSpaceDistribution & state_distro,
				 double time, double dt,
				 const mat_t & observable) {
  RungeKuttaSolver solver = build_runge_kutta_4();
  std::cout << "performing runge kutta" << std::endl;
  // RungeKuttaSolver solver = dormand_price();
  // RungeKuttaSolver solver = cash_karp();
  
  calc_mat_t density_matrix = state_distro.density_matrix().sparseView();
  int time_steps = static_cast<int>(time / dt);
  Eigen::VectorXd n_ensemble = Eigen::VectorXd::Zero(time_steps);
  double t = 0.0;
  for (int j = 0; j < time_steps; ++j, t += dt) {
    std::cout << j << std::endl;
    density_matrix = solver.perform_step<calc_mat_t>(t, dt, density_matrix, system);
    n_ensemble(j) = (observable * density_matrix).trace().real();
  }
  return n_ensemble;
}

std::vector<mat_t>
density_matrix_kutta(const Lindbladian & system,
		     const HSpaceDistribution & state_distro,
		     double time, double dt,
		     const mat_t & observable) {
  // RungeKuttaSolver solver = build_runge_kutta_4();
  std::cout << "performing runge kutta" << std::endl;
  // RungeKuttaSolver solver = dormand_price();
  RungeKuttaSolver solver = cash_karp();

  std::vector<mat_t> density_matrices;
  calc_mat_t density_matrix = state_distro.density_matrix().sparseView();
  int time_steps = static_cast<int>(time / dt);
  double t = 0.0;
  for (int j = 0; j < time_steps; ++j, t += dt) {
    std::cout << j << std::endl;
    density_matrix = solver.perform_step<calc_mat_t>(t, dt, density_matrix, system);
    density_matrices.push_back(density_matrix);
  }
  return density_matrices;
}
