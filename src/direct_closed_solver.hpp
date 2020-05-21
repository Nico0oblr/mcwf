#ifndef DIRECT_CLOSED_SOLVER_HPP
#define DIRECT_CLOSED_SOLVER_HPP

#include "Common.hpp"

class HSpaceDistribution;
class Lindbladian;

Eigen::VectorXd direct_closed_observable(const mat_t & system,
					 const vec_t & cstate,
					 double time, double dt,
					 const spmat_t & observable);

#endif /* DIRECT_CLOSED_SOLVER_HPP */
