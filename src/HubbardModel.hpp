#include "Common.hpp"

#ifndef HUBBARDMODEL_HPP
#define HUBBARDMODEL_HPP

class HSpaceDistribution;

/*
  Operators in basis |0>, |down>, |up>, |up,down>.
*/
namespace HubbardOperators {
  mat_t c_up_t();

  mat_t c_down_t();

  mat_t c_down();

  mat_t c_up();

  mat_t n_down();

  mat_t n_up();  
}

mat_t Hubbard_hamiltonian(int sites,
			  double hopping,
			  double hubbardU,
			  bool periodic);

/*
  Define Peierls substitution hubbard model hamiltonian.
*/
mat_t Hubbard_light_matter(int photon_dimension,
			   int sites,
			   double coupling,
			   double hopping,
			   double hubbardU,
			   bool periodic);

/*
  Get the spin sector of a Hubbard state
*/
std::pair<int, int> get_spin_sector(const vec_t & state);

HSpaceDistribution HubbardNeelState(int sites, const mat_t & projection);

HSpaceDistribution HubbardNeelState_sp(int sites, const spmat_t & projection);

HSpaceDistribution DimerGroundState(double hopping, double hubbardU);

mat_t HubbardProjector(int sites, int total_spins_up, int total_spins_down);

spmat_t HubbardProjector_sp(int sites, int total_spins_up, int total_spins_down);

spmat_t Hubbard_light_matter_sp(int photon_dimension,
				int sites,
				double coupling,
				double hopping,
				double hubbardU,
				bool periodic,
				const spmat_t & proj);

spmat_t Hubbard_hamiltonian_sp(int sites,
			       double hopping,
			       double hubbardU,
			       bool periodic,
			       const spmat_t & proj);

HSpaceDistribution HubbardGroundState(int sites,
				      double hopping,
				      double hubbardU,
				      bool periodic,
				      const spmat_t & proj);
#endif /* HUBBARDMODEL_HPP */
