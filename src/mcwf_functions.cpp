#include "mcwf_functions.hpp"

#include "Operators.hpp"
#include "Lindbladian.hpp"
#include "HSpaceDistribution.hpp"

vec_t jump_process(const vec_t & state,
		   const Lindbladian & system) {
  Eigen::VectorXd jump_probabilities(system.m_lindblad_operators.size());

  for (int i = 0; i < system.m_lindblad_operators.size(); ++i) {
    jump_probabilities(i) = (system.m_lindblad_operators[i] * state).squaredNorm();
    jump_probabilities(i) *= std::abs(system.m_lindblad_amplitudes[i]);
  }
  if (jump_probabilities.sum() < tol) return state / state.norm();
  
  jump_probabilities /= jump_probabilities.sum();
  int jump_index = linear_search(jump_probabilities);
  mat_t out = system.m_lindblad_operators.at(jump_index) * state;
  return out / out.norm();
}

void perform_time_step(const Lindbladian & system,
		       const spmat_t & propagator,
		       vec_t & state) {
  state = propagator * state;
  double norm = state.squaredNorm();

  if (!(norm <= 1.0)) {
    std::cout << "Warning: Norm should be decreasing with time: "
	      << norm << std::endl;
  }
  
  if ((std::abs(1 - norm)) > 0.05) {
    std::cout << "Warning: Norm deviation very large: "
	      << std::abs(1 - norm) << std::endl;
  }
      
  double eta = dis(mt_rand);
  if (norm < eta) {
    state = jump_process(state, system);
  } else {
    state /= std::sqrt(norm);
  }
}

Eigen::MatrixXd observable_calc(const Lindbladian & system,
				const HSpaceDistribution & state_distro,
				double time, double dt,
				int runs,
				const spmat_t & observable) {
  std::cout << "starting mcwf observable run" << std::endl;
  spmat_t propagator = matrix_exponential_taylor(-1.0i * system.hamiltonian() * dt).sparseView();
  std::cout << "percentage zeroes propagator: "
	    << propagator.nonZeros()
    / static_cast<double>(propagator.size()) << std::endl;
  int time_steps = static_cast<int>(time / dt);
  Eigen::MatrixXd n_ensemble = Eigen::MatrixXd::Zero(runs, time_steps);
#pragma omp parallel for
  for (int i = 0; i < runs; ++i) {
#pragma omp critical
    std::cout << i << std::endl;
    vec_t state = state_distro.draw();
    double t = 0.0;
    for (int j = 0; j < time_steps; ++j, t += dt) {
      perform_time_step(system, propagator, state);
      n_ensemble(i, j) = ((state.adjoint() * observable * state).real())(0);
    }    
  }
  return n_ensemble;
}

Eigen::MatrixXd two_time_correlation(const Lindbladian & system,
				     const HSpaceDistribution & state_distro,
				     double t1, double t2, double dt,
				     int runs,
				     const mat_t A0,
				     const mat_t A1) {
  spmat_t propagator = matrix_exponential(-1.0i * system.hamiltonian() * dt).sparseView();
  int sub_dim = system.m_system_hamiltonian.cols();
  Lindbladian doubled_system(double_matrix(system.m_system_hamiltonian),
			     double_matrix(system.m_lindblad_operators),
			     system.m_lindblad_amplitudes);
  spmat_t doubled_propagator = double_matrix(propagator).sparseView();
  int time_steps = static_cast<int>((t2 - t1) / dt);
  print_matrix_dim(doubled_propagator);

  Eigen::MatrixXd n_ensemble = Eigen::MatrixXd::Zero(runs, time_steps);
  for (int i = 0; i < runs; ++i) {
    vec_t state = state_distro.draw();
    double t = 0.0;
    for (double t = 0; t <= t1; t += dt) {
      perform_time_step(system, propagator, state);
    }

    vec_t doubled_state = add_vectors(state, A0 * state);
    double norm = doubled_state.norm() / state.norm();
    doubled_state /= norm;
    for (int j = 0; j < time_steps; ++j, t += dt) {
      perform_time_step(doubled_system, doubled_propagator, doubled_state);
      double correlation = ((doubled_state.head(sub_dim).adjoint() * A1
			     * doubled_state.tail(sub_dim)
			     / doubled_state.squaredNorm()
			     * norm * norm).real())(0);
      n_ensemble(i, j) = correlation;
    }
  }
  return n_ensemble;
}

std::vector<mat_t>
density_matrix_mcwf(const Lindbladian & system,
		    const HSpaceDistribution & state_distro,
		    double time, double dt,
		    int runs) {
  spmat_t propagator = matrix_exponential(-1.0i * system.hamiltonian() * dt).sparseView();
  int time_steps = static_cast<int>(time / dt);
  std::vector<mat_t> density_matrix_ensemble(time_steps, mat_t(0,0));
  for (int i = 0; i < runs; ++i) {
    std::cout << i << std::endl;
    vec_t state = state_distro.draw();
    double t = 0.0;
    for (int j = 0; j < time_steps; ++j, t += dt) {
      perform_time_step(system, propagator, state);

      if (density_matrix_ensemble[j].size() == 0) {
	density_matrix_ensemble[j] = state * state.adjoint();
      } else {
	density_matrix_ensemble[j] += state * state.adjoint();
      }
    }
  }

  for (int j = 0; j < time_steps; ++j) {
    density_matrix_ensemble[j] /= static_cast<double>(runs);
  }
  
  return density_matrix_ensemble;
}
