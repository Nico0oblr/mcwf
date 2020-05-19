#ifndef RUNGE_KUTTA_SOLVER_HPP
#define RUNGE_KUTTA_SOLVER_HPP

#include "Common.hpp"
class Lindbladian;
class HSpaceDistribution;

Eigen::VectorXd observable_kutta(const Lindbladian & system,
				 const HSpaceDistribution & state_distro,
				 double time, double dt,
				 const mat_t & observable);

std::vector<mat_t>
density_matrix_kutta(const Lindbladian & system,
		     const HSpaceDistribution & state_distro,
		     double time, double dt,
		     const mat_t & observable);

#endif /* RUNGE_KUTTA_SOLVER_HPP */
