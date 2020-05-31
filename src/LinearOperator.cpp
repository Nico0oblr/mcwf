#include "LinearOperator.hpp"

std::unique_ptr<LinearOperator<spmat_t>>
Hubbard_light_matter_Operator(int photon_dimension,
			      int sites,
			      double coupling,
			      double hopping,
			      double hubbardU,
			      bool periodic,
			      const spmat_t & proj) {
  int dimension = std::pow(4, sites);
  spmat_t hopping_terms = spmat_t::Zero(dimension, dimension);
  spmat_t onsite_terms = spmat_t::Zero(dimension, dimension);

  using namespace HubbardOperators;
  for (int i = 0; i + 1 < sites; ++i) {
    hopping_terms += n_th_subsystem_sp(c_up_t(), i, sites)
      * n_th_subsystem_sp(c_up(), i + 1, sites);
    hopping_terms += n_th_subsystem_sp(c_down_t(), i, sites)
      * n_th_subsystem_sp(c_down(), i + 1, sites);
  }

  if (periodic && sites > 2) {
    hopping_terms += n_th_subsystem_sp(c_up_t(), sites - 1, sites)
      * n_th_subsystem_sp(c_up(), 0, sites);
    hopping_terms += n_th_subsystem_sp(c_down_t(), sites - 1, sites)
      * n_th_subsystem_sp(c_down(), 0, sites);
  }

  for (int i = 0; i < sites; ++i) {
    onsite_terms += n_th_subsystem_sp(n_up(), i, sites)
      * n_th_subsystem_sp(n_down(), i, sites);
  }

  onsite_terms *= hubbardU;
  hopping_terms *= hopping;
  mat_t argument = 1.0i * coupling * (creationOperator(photon_dimension)
				      + annihilationOperator(photon_dimension));
  spmat_t e_iA = matrix_exponential(argument);
  spmat_t e_iA_adj = e_iA.adjoint();
  hopping_terms = proj * hopping_terms * proj.adjoint();
  onsite_terms = proj * onsite_terms * proj.adjoint();
  spmat_t hopping_terms_adj = hopping_terms.adjoint();

  return kroneckerOperator(e_iA, hopping_terms)
    + kroneckerOperator(e_iA_adj, hopping_terms_adj)
    + kroneckerOperator_IDLHS(onsite_terms, photon_dimension);
}
