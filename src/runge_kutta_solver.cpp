#include "runge_kutta_solver.hpp"

#include "Operators.hpp"
#include "HSpaceDistribution.hpp"
#include "Lindbladian.hpp"
#include "runge_kutta.hpp"

void observable_kutta(const Lindbladian & system,
		      const HSpaceDistribution & state_distro,
		      double time, double dt,
		      RecorderHost<calc_mat_t> & recorder) {
  std::cout << "performing runge kutta" << std::endl;
  RungeKuttaSolver solver = build_runge_kutta_4();
  calc_mat_t density_matrix = state_distro.density_matrix();
  int time_steps = static_cast<int>(time / dt);
  double t = 0.0;
  for (int j = 0; j < time_steps; ++j, t += dt) {
    density_matrix = solver.perform_step<calc_mat_t>(t, dt,
						     density_matrix, system);
    recorder.record(density_matrix);
  }
}
