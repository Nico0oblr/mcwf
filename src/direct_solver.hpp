#ifndef DIRECT_SOLVER_HPP
#define DIRECT_SOLVER_HPP

#include "Common.hpp"
#include "Recorders.hpp"
struct Lindbladian;
class HSpaceDistribution;

void observable_direct(const Lindbladian & system,
		       const HSpaceDistribution & state_distro,
		       double time, double dt,
		       RecorderHost<calc_mat_t> & recorder);


void two_time_correlation_direct(const Lindbladian & system,
				 const HSpaceDistribution & state_distro,
				 double t0, double t1,
				 double dt,
				 const calc_mat_t & A,
				 RecorderHost<calc_mat_t> & recorder);

#endif /* DIRECT_SOLVER_HPP */
