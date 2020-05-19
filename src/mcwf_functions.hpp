#ifndef MCWF_FUNCTIONS_HPP
#define MCWF_FUNCTIONS_HPP

#include "Common.hpp"
class Lindbladian;
class HSpaceDistribution;

vec_t jump_process(const vec_t & state,
		   const Lindbladian & system);

void perform_time_step(const Lindbladian & system,
		       const spmat_t & propagator,
		       vec_t & state);

/*
  Do mcwf run and evaluate the observable given. 
*/
Eigen::MatrixXd observable_calc(const Lindbladian & system,
				const HSpaceDistribution & state_distro,
				double time, double dt,
				int runs,
				const spmat_t & observable);

/*
  Two-time correlation function for fixed times t1 and t0.
*/
/*Eigen::MatrixXd two_time_correlation(const Lindbladian & system,
				     const HSpaceDistribution & state_distro,
				     const mat_t & propagator,
				     double t1, double t2, double dt,
				     int runs,
				     const mat_t A0,
				     const mat_t A1);*/

std::vector<mat_t>
density_matrix_mcwf(const Lindbladian & system,
		    const HSpaceDistribution & state_distro,
		    double time, double dt,
		    int runs);

/*
  For the calculation of a two-time correlation function <A(t_2)B(t_1)>. one has to split the trajectory at time t_1
  and then double the trajectory dimension. This doubled trajectory has to then be evolved until t_2.
  To evaluate this quantity for arbitrary
  have to perform joint evolution and joint jumps
*/

#endif /* MCWF_FUNCTIONS_HPP */
