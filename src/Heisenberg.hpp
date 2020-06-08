#ifndef HEISENBERG_HPP
#define HEISENBERG_HPP

#include "Common.hpp"
#include "Operators.hpp"

// spmat_t sector_projector() {}
spmat_t Heisenberg_spin_projector(int sites, double spin);

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

spmat_t HeisenbergChain_sp(int sites,
			   double Jx, double Jy, double Jz,
			   bool periodic);

#endif /* HEISENBERG_HPP */
