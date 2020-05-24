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
				  int order,
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
std::vector<mat_t> operator_vector(const mat_t & op,
				   int sites);

/*
  Sums the action of the operator op on every site
*/
mat_t sum_operator(const mat_t & op,
		   int sites);

/*
  Pauli x matrix for a single spin for z-axis quantized.
*/
mat_t pauli_x();

/*
  Pauli y matrix for a single spin for z-axis quantized.
*/
mat_t pauli_y();

/*
  Pauli z matrix for a single spin for z-axis quantized.
*/
mat_t pauli_z();

/*
  Creates vector of pauli x matrices acting on the ith particle for 
  some number of total sites.
*/
std::vector<mat_t> pauli_x_vector(int sites);

/*
  Creates vector of pauli y matrices acting on the ith particle for 
  some number of total sites.
*/
std::vector<mat_t> pauli_y_vector(int sites);

/*
  Creates vector of pauli z matrices acting on the ith particle for 
  some number of total sites.
*/
std::vector<mat_t> pauli_z_vector(int sites);

/*
  The total pauli z matrix of a system with a given number of sites
*/
mat_t pauli_z_total(int sites);

/*
  Total squared pauli matrix for a sites-spin system
*/
mat_t pauli_squared_total(int sites);

/*
  Defines the Hamiltonian for the heisenberg chain
  with a given number of sites, an anisotropy Jx, Jy, Jz
  and an option for periodic or open system.
 */
mat_t HeisenbergChain(int sites,
		      double Jx, double Jy, double Jz,
		      bool periodic);
#endif /* OPERATORS_HPP */
