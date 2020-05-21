#include "direct_closed_solver.hpp"

#include "HSpaceDistribution.hpp"
#include "Lindbladian.hpp"
#include "Operators.hpp"

Eigen::VectorXd direct_closed_observable(const mat_t & system,
					 const vec_t & cstate,
					 double time, double dt,
					 const spmat_t & observable) {
  vec_t state = cstate;
  spmat_t propagator = matrix_exponential(-1.0i * system * dt).sparseView();
  std::cout << "hermitian? " << (system - system.adjoint()).norm() << std::endl;
  int time_steps = static_cast<int>(time / dt);
  Eigen::VectorXd n_ensemble = Eigen::VectorXd::Zero(time_steps);

  double t = 0;
  for (int j = 0; j < time_steps; ++j, t += dt) {
    state = propagator * state;
    state /= state.norm();
    n_ensemble(j) = ((state.adjoint() * observable * state).real())(0);
    
    std::cout << "n_ensemble(j): " << n_ensemble(j) << std::endl;
  }

  return n_ensemble;
}
