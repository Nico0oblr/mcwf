#ifndef DIRECT_SOLVER_HPP
#define DIRECT_SOLVER_HPP

#include "Common.hpp"
class Lindbladian;
class HSpaceDistribution;

std::vector<mat_t> density_matrix_direct(const Lindbladian & system,
					 const HSpaceDistribution & state_distro,
					 double time, double dt,
					 const mat_t & observable);

Eigen::VectorXd observable_direct(const Lindbladian & system,
				  const HSpaceDistribution & state_distro,
				  double time, double dt,
				  const mat_t & observable);

#endif /* DIRECT_SOLVER_HPP */
