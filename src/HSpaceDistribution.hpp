#ifndef HSPACEDISTRIBUTION_HPP
#define HSPACEDISTRIBUTION_HPP

#include "Common.hpp"

/*
  This class models a discrete distribution of wave functions on a Hilbert space
*/
class HSpaceDistribution {
public:
  /*
    Draws a wave function from distribution
   */
  vec_t draw() const;

  /*
    Explicitly define discrete probability distribution
   */
  HSpaceDistribution(const std::vector<double> & probabilities,
		     const std::vector<vec_t> & states);

  /*
    Defines distribution on basis vectors e_i with i listed in vector states
    and their respective probabilities in the vector probabilities
   */
  HSpaceDistribution(const std::vector<double> & probabilities,
		     const std::vector<int> & states,
		     int dimension);

  /*
    Define equiprobable distribution on all basis states.
   */
  HSpaceDistribution(int dimension);

  /*
    Combine two distributions via the tensor product in the obvious way.
    No entangled states will be generated.
   */
  HSpaceDistribution & operator+=(const HSpaceDistribution & other);

  /*
    Construct corresponding density matrix
  */
  mat_t density_matrix() const;
  
private:
  std::vector<double> m_probabilities;
  std::vector<vec_t> m_states;
};

HSpaceDistribution coherent_photon_state(double mean_photons, int dimension);

#endif /* HSPACEDISTRIBUTION_HPP */
