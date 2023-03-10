#include "direct_closed_solver.hpp"

#include "HSpaceDistribution.hpp"
#include "Lindbladian.hpp"
#include "Operators.hpp"

void direct_closed_observable(Hamiltonian<calc_mat_t> & system,
			      const vec_t & cstate,
			      double time, double dt,
			      RecorderHost<vec_t> & recorder) {
  LOG(logINFO) << "Running direct closed solver" << std::endl;
  vec_t state = cstate;
  int time_steps = static_cast<int>(time / dt);
  // Eigen::VectorXd n_ensemble = Eigen::VectorXd::Zero(time_steps);
  
  double t = 0;
  for (int j = 0; j < time_steps; ++j, t += dt) {
    // mat_t propagator = system(t, dt);
    state = system.propagate(t, dt, state);
    // Just in case. Numerical errors increase norm when using many time steps.
    // state /= state.norm();
    // n_ensemble(j) = expval(observable, state);
    recorder.record(state);
  }
  
  // return n_ensemble;
}

void direct_closed_two_time_correlation(Hamiltonian<calc_mat_t> & system,
					const vec_t & cstate,
					double t0, double t1,
					double dt,
					const calc_mat_t & A,
					CorrelationRecorderMixin & recorder) {
  vec_t state = cstate;
  int time_steps0 = static_cast<int>(t0 / dt);
  int time_steps1 = static_cast<int>((t1 - t0) / dt);

  double t = 0;
  // Eigen::VectorXd n_ensemble = Eigen::VectorXd::Zero(time_steps1);
  for (int j = 0; j < time_steps0; ++j, t += dt) {
    state = system.propagate(t, dt, state);
    state /= state.norm();
  }
  vec_t Astate = A * state;
  double current_norm = Astate.norm();
  
  for (int j = 0; j < time_steps1; ++j, t += dt) {
    Astate = system.propagate(t, dt, Astate);
    state = system.propagate(t, dt, state);
    state /= state.norm();
    Astate *= (current_norm / Astate.norm());
    recorder.record(state, Astate);
    // n_ensemble(j) = state.dot(A * Astate).real();
  }
}
