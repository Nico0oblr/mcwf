#ifndef OPERATORS_HPP
#define OPERATORS_HPP

#include "Common.hpp"

/*
  Object, that precomputes powers of optical operators
*/
struct PrecomputedOperators_str {
  std::vector<spmat_t> A_powers;
  std::vector<spmat_t> A_t_powers;
  std::vector<spmat_t> n_powers;

  void precompute(int dimension);

  spmat_t A_t(int power) const;
  
  spmat_t A(int power) const;
  
  spmat_t n(int power) const;
  
  int m_dimension;
};

/*Global object, that has to be initialized with the dimension before usage*/
extern PrecomputedOperators_str PrecomputedOperators;

/*
  Defines creation operator in the number basis of a
  purely photonic hilbert space
*/
mat_t creationOperator(int dimension);

/*
  Defines adnnihilation operator in the number basis of a
  purely photonic hilbert space
*/
mat_t annihilationOperator(int dimension);

/*
  Defines number operator in the number basis of a
  purely photonic hilbert space
*/
mat_t numberOperator(int dimension);

/*
  Sparse photonic creation operator
*/
spmat_t creationOperator_sp(int dimension);

/*
  Sparse photonic annihilation operator
*/
spmat_t annihilationOperator_sp(int dimension);

/*
  Sparse photonic number operator
*/
spmat_t numberOperator_sp(int dimension);

/*
  Auxilary function to define exchange interaction. 
  See https://arxiv.org/pdf/2002.12912.pdf
  Is globally visible, such that it can be tested
*/
double L_p(double omega_bar, double coupling, int p);

/*
  Auxilary function to define exchange interaction. 
  See https://arxiv.org/pdf/2002.12912.pdf
  Is globally visible, such that it can be tested
*/
double L_c_m(double omega_bar, double coupling, int c, int m);

/*
  Auxilary function to define exchange interaction. 
  See https://arxiv.org/pdf/2002.12912.pdf
  mth term of the photon-diagonal exchange interaction
*/
spmat_t exchange_interaction_term(int m,
				  double coupling,
				  double omega_bar,
				  int dimension);

/*
  Full exchange interaction up to term order
  in nondiagonal processes. Contains some reasonable numerical cutoffs.
*/
spmat_t exchange_interaction_full(int dimension,
				  double hubbardU,
				  double hopping,
				  double frequency,
				  double coupling,
				  int order);

/*
  Exchange interaction up to g^2.
*/
mat_t exchange_interaction(int dimension,
			   double hubbardU,
			   double hopping,
			   double frequency,
			   double coupling);

/*
  Exchange interaction J0
*/
mat_t J0(int dimension,
	 double hubbardU,
	 double hopping,
	 double frequency,
	 double coupling);

/*
  Generates an operator
  Id \tensor Id ... \tensor op \tensor id ...
  where the position of op is n_subsystem and
  the total number of operators is n_subsystems
*/
mat_t nth_subsystem(const mat_t & op,
		    int n_subsystem,
		    int n_subsystems);

/*
  Creates a vector, where the ith element only acts on the ith site
 */
std::vector<mat_t> operator_vector(const Eigen::Ref<const mat_t> & op,
				   int sites);

/*
  Sums the action of the operator op on every site
*/
mat_t sum_operator(const mat_t & op,
		   int sites);

spmat_t n_th_subsystem_sp(const spmat_t & op,
			  int n_subsystem,
			  int n_subsystems);

spmat_t sum_operator_sp(const spmat_t & op, int sites);

#endif /* OPERATORS_HPP */
