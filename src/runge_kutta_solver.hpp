#ifndef RUNGE_KUTTA_SOLVER_HPP
#define RUNGE_KUTTA_SOLVER_HPP

#include "Common.hpp"
#include "Recorders.hpp"
struct Lindbladian;
class HSpaceDistribution;

void observable_kutta(const Lindbladian & system,
		      const HSpaceDistribution & state_distro,
		      double time, double dt,
		      RecorderHost<calc_mat_t> & recorder);

#endif /* RUNGE_KUTTA_SOLVER_HPP */
