#ifndef DIRECT_CLOSED_SOLVER_HPP
#define DIRECT_CLOSED_SOLVER_HPP

#include "Common.hpp"
#include "Operators.hpp"
#include "Hamiltonian.hpp"
#include "Recorders.hpp"

class HSpaceDistribution;
class Lindbladian;

void direct_closed_observable(Hamiltonian<calc_mat_t> & system,
			      const vec_t & cstate,
			      double time, double dt,
			      RecorderHost<vec_t> & recorder);

Eigen::VectorXd
direct_closed_two_time_correlation(Hamiltonian<calc_mat_t> & system,
				   const vec_t & cstate,
				   double t0, double t1, double dt,
				   const calc_mat_t & A,
				   const calc_mat_t & B);


#endif /* DIRECT_CLOSED_SOLVER_HPP */
