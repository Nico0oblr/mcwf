#include "direct_closed_solver.hpp"

#include "HSpaceDistribution.hpp"
#include "Lindbladian.hpp"
#include "Operators.hpp"

void direct_closed_observable(Hamiltonian<calc_mat_t> & system,
			      const vec_t & cstate,
			      double time, double dt,
			      RecorderHost<vec_t> & recorder) {
  vec_t state = cstate;
  int time_steps = static_cast<int>(time / dt);
  // Eigen::VectorXd n_ensemble = Eigen::VectorXd::Zero(time_steps);
  
  double t = 0;
  for (int j = 0; j < time_steps; ++j, t += dt) {
    std::cout << "j: " << j << std::endl;
    // mat_t propagator = system(t, dt);
    state = system.propagate(t, dt, state);
    // Just in case. Numerical errors increase norm when using many time steps.
    state /= state.norm();
    // n_ensemble(j) = expval(observable, state);
    recorder.record(state);
  }
  
  // return n_ensemble;
}

Eigen::VectorXd
direct_closed_two_time_correlation(Hamiltonian<calc_mat_t> & system,
				   const vec_t & cstate,
				   double t0, double t1,
				   double dt,
				   const calc_mat_t & A,
				   const calc_mat_t & B) {
  vec_t state = cstate;
  int time_steps0 = static_cast<int>(t0 / dt);
  int time_steps1 = static_cast<int>((t1 - t0) / dt);

  double t = 0;
  Eigen::VectorXd n_ensemble = Eigen::VectorXd::Zero(time_steps1);
  for (int j = 0; j < time_steps0; ++j, t += dt) {
    state = system.propagate(t, dt, state);
    state /= state.norm();
  }
  vec_t Bstate = B * state;
  double current_norm = Bstate.norm();
  
  for (int j = 0; j < time_steps1; ++j, t += dt) {
    Bstate = system.propagate(t, dt, Bstate);
    state = system.propagate(t, dt, state);
    state /= state.norm();
    Bstate *= (current_norm / Bstate.norm());
    n_ensemble(j) = state.dot(A * Bstate).real();
  }
  return n_ensemble;
}
