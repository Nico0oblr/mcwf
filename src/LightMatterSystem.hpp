#ifndef LIGHTMATTERSYSTEM_HPP
#define LIGHTMATTERSYSTEM_HPP

#include "Common.hpp"
#include "Lindbladian.hpp"
#include "argparse.hpp"
#include "HSpaceDistribution.hpp"
#include "Heisenberg.hpp"

struct LightMatterSystem {
  Lindbladian system;
  mat_t light_matter;
  mat_t projector;
  int dimension;
  int sites;
  double temperature;
  std::string model;

  int elec_dim() const {
    return projector.rows();
  }

  const Hamiltonian<calc_mat_t> & hamiltonian() const {
    return *(system.m_system_hamiltonian);
  }

  Hamiltonian<calc_mat_t> & hamiltonian() {
    return *(system.m_system_hamiltonian);
  }
};

LightMatterSystem parse_system(YamlOrCMD & parser);

mat_t parse_observable(const LightMatterSystem & lms,
		       std::string observable_name);

HSpaceDistribution parse_initial_distribution(const LightMatterSystem & lms,
					      YamlOrCMD & parser);

#endif /* LIGHTMATTERSYSTEM_HPP */
