#include "mcwf_functions.hpp"

#include "Operators.hpp"
#include "Lindbladian.hpp"
#include "HSpaceDistribution.hpp"

vec_t jump_process(const vec_t & state,
		   const Lindbladian & system) {
  // LOG(logINFO) << "JUMP" << std::endl;
  Eigen::VectorXd jump_probabilities(system.m_lindblad_operators.size());

  for (size_type i = 0; i < system.m_lindblad_operators.size(); ++i) {
    jump_probabilities(i) = (*system.m_lindblad_operators[i] * state).squaredNorm();
    jump_probabilities(i) *= std::abs(system.m_lindblad_amplitudes[i]);
  }
  if (jump_probabilities.sum() < tol) return state / state.norm();
  
  jump_probabilities /= jump_probabilities.sum();
  int jump_index = linear_search(jump_probabilities);
  calc_mat_t out = *system.m_lindblad_operators.at(jump_index) * state;
  return out / out.norm();
}

void perform_time_step(const Lindbladian & system,
		       Hamiltonian<calc_mat_t> & hamiltonian,
		       double t, double dt,
		       vec_t & state) {
  state = hamiltonian.propagate(t, dt, state);
  double norm = state.squaredNorm();

  if (norm > 1.0) {
    LOG(logWARNING) << "Warning: Norm should be decreasing with time: "
		    << std::endl;
  }
  
  if ((std::abs(1 - norm)) > 0.05) {
    LOG(logWARNING) << "Warning: Norm deviation very large: "
		    << std::abs(1 - norm) << std::endl;
  }
      
  double eta = dis(mt_rand);
  if (norm < eta) {
    state = jump_process(state, system);
  } else {
    state /= std::sqrt(norm);
  }
}

void mcwf_singlerun(const Lindbladian & system,
		    const HSpaceDistribution & state_distro,
		    double time, double dt,
		    MCWFRecorder & recorder) {
  auto hamiltonian = system.hamiltonian();
  int run_id = recorder.register_run();
  vec_t state = state_distro.draw();
  int time_steps = static_cast<int>(time / dt);
  double t = 0.0;
  for (int j = 0; j < time_steps; ++j, t += dt) {
    perform_time_step(system, *hamiltonian, t, dt, state);
    recorder.record(state, run_id, j);
  }    
}

void observable_calc(const Lindbladian & system,
		     const HSpaceDistribution & state_distro,
		     double time, double dt, int runs,
		     MCWFRecorder & recorder) {
  LOG(logINFO) << "starting mcwf observable run" << std::endl;
#pragma omp parallel for
  for (int i = 0; i < runs; ++i) {
    mcwf_singlerun(system, state_distro, time, dt, recorder);
  }
}

void two_time_correlation_singlerun(const Lindbladian & system,
				    const HSpaceDistribution & state_distro,
				    double t1, double t2, double dt,
				    const calc_mat_t & A0,
				    MCWFCorrelationRecorderMixin & recorder) {
  int sub_dim = system.m_system_hamiltonian->dimension();
  auto doubled_system_ham = system.m_system_hamiltonian->clone();
  doubled_system_ham->doubleMe();
  Lindbladian doubled_system(*doubled_system_ham,
			     doubleOperatorVector(system.m_lindblad_operators),
			     system.m_lindblad_amplitudes);
  auto hamiltonian = system.hamiltonian();
  auto doubled_hamiltonian = doubled_system.hamiltonian();
  int time_steps = static_cast<int>((t2 - t1) / dt);
  int run_id = recorder.register_run();
  vec_t state = state_distro.draw();
  double t = 0.0;
  for (t = 0.0; t <= t1; t += dt) {
    perform_time_step(system, *hamiltonian, t, dt, state);
  }

  vec_t doubled_state = add_vectors(state, A0 * state);
  double norm = doubled_state.norm() / state.norm();
  doubled_state /= norm;
  for (int j = 0; j < time_steps; ++j) {
    perform_time_step(doubled_system, *doubled_hamiltonian,
		      t, dt, doubled_state);
    double factor = norm * norm / doubled_state.squaredNorm();
    recorder.record(doubled_state.head(sub_dim),
		    doubled_state.tail(sub_dim) * factor,
		    run_id, j);
    t += dt;
  }
}

void two_time_correlation(const Lindbladian & system,
			  const HSpaceDistribution & state_distro,
			  double t1, double t2, double dt,
			  int runs,
			  const calc_mat_t & A0,
			  MCWFCorrelationRecorderMixin & recorder) {
#pragma omp parallel for
  for (int i = 0; i < runs; ++i) {
    two_time_correlation_singlerun(system, state_distro, t1, t2,
				   dt, A0, recorder);
  }
}
