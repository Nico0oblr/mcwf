#include "HubbardModel.hpp"
#include "Operators.hpp"
#include "HSpaceDistribution.hpp"

mat_t HubbardOperators::c_up_t() {
  mat_t out(4, 4);
  out <<
    0, 0, 0, 0,
    0, 0, 0, 0,
    1, 0, 0, 0,
    0, 1, 0, 0;
  return out;
}

mat_t HubbardOperators::c_down_t() {
  mat_t out(4, 4);
  out <<
    0, 0, 0, 0,
    1, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 1, 0;
  return out;
}

mat_t HubbardOperators::c_down() {
  return c_down_t().adjoint();
}

mat_t HubbardOperators::c_up() {
  return c_up_t().adjoint();
}

mat_t HubbardOperators::n_down() {
  mat_t out(4, 4);
  out <<
    0, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 1;
  return out;
}

mat_t HubbardOperators::n_up() {
  mat_t out(4, 4);
  out <<
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1;
  return out;
}

mat_t Hubbard_hamiltonian(int sites,
			  double hopping,
			  double hubbardU,
			  bool periodic) {
  std::vector<mat_t> c_up = operator_vector(HubbardOperators::c_up(), sites);
  std::vector<mat_t> c_up_t = operator_vector(HubbardOperators::c_up_t(), sites);
  std::vector<mat_t> n_up = operator_vector(HubbardOperators::n_up(), sites);;
  std::vector<mat_t> c_down = operator_vector(HubbardOperators::c_down(), sites);
  std::vector<mat_t> c_down_t = operator_vector(HubbardOperators::c_down_t(), sites);
  std::vector<mat_t> n_down = operator_vector(HubbardOperators::n_down(), sites);
  // Onsite dimension of 4, when |up, down> and |down,up> are identified
  int dimension = std::pow(4, sites);
  mat_t hopping_terms = mat_t::Zero(dimension, dimension);
  mat_t onsite_terms = mat_t::Zero(dimension, dimension);

  for (int i = 0; i + 1 < sites; ++i) {
    hopping_terms += c_up_t[i] * c_up[i + 1];
    hopping_terms += c_down_t[i] * c_down[i + 1];
  }

  if (periodic && sites > 2) {
    hopping_terms += c_up_t[sites - 1] * c_up[0];
    hopping_terms += c_down_t[sites - 1] * c_down[0];
  }

  for (int i = 0; i < sites; ++i) {
    onsite_terms += n_up[i] * n_down[i];
  }
  
  mat_t n_up_total = mat_t::Zero(dimension, dimension);
  mat_t n_down_total = mat_t::Zero(dimension, dimension);
  for (int i = 0; i < sites; ++i) {
    n_down_total += n_down[i];
    n_up_total += n_up[i];
  }

  return hopping * hopping_terms
    + hopping * hopping_terms.adjoint()
    + hubbardU * onsite_terms;
}

std::pair<int, int> get_spin_sector(const vec_t & state) {
  int sites = state.size() / 4;
  assert(state.size() == sites * 4);
  mat_t n_up_total = sum_operator(HubbardOperators::n_up(), sites);
  mat_t n_down_total = sum_operator(HubbardOperators::n_down(), sites);
  return std::make_pair(static_cast<int>(expval(n_up_total, state) + 0.5),
			static_cast<int>(expval(n_down_total, state) + 0.5));
}

mat_t Hubbard_light_matter(int photon_dimension,
			   int sites,
			   double coupling,
			   double hopping,
			   double hubbardU,
			   bool periodic) {
  std::vector<mat_t> c_up = operator_vector(HubbardOperators::c_up(), sites);
  std::vector<mat_t> c_up_t = operator_vector(HubbardOperators::c_up_t(), sites);
  std::vector<mat_t> n_up = operator_vector(HubbardOperators::n_up(), sites);;
  std::vector<mat_t> c_down = operator_vector(HubbardOperators::c_down(), sites);
  std::vector<mat_t> c_down_t = operator_vector(HubbardOperators::c_down_t(), sites);
  std::vector<mat_t> n_down = operator_vector(HubbardOperators::n_down(), sites);
  // Onsite dimension of 4, when |up, down> and |down,up> are identified
  int dimension = std::pow(4, sites);
  mat_t hopping_terms = mat_t::Zero(dimension, dimension);
  mat_t onsite_terms = mat_t::Zero(dimension, dimension);

  for (int i = 0; i + 1 < sites; ++i) {
    hopping_terms += c_up_t[i] * c_up[i + 1];
    hopping_terms += c_down_t[i] * c_down[i + 1];
  }

  if (periodic && sites > 2) {
    hopping_terms += c_up_t[sites - 1] * c_up[0];
    hopping_terms += c_down_t[sites - 1] * c_down[0];
  }

  for (int i = 0; i < sites; ++i) {
    onsite_terms += n_up[i] * n_down[i];
  }

  onsite_terms *= hubbardU;
  hopping_terms *= hopping;
  mat_t argument = 1.0i * coupling * (creationOperator(photon_dimension)
				      + annihilationOperator(photon_dimension));
  mat_t e_iA = matrix_exponential(argument);
  hopping_terms = Eigen::kroneckerProduct(e_iA, hopping_terms).eval();
  onsite_terms = tensor_identity_LHS(onsite_terms, photon_dimension).eval();
  return hopping_terms + hopping_terms.adjoint() + onsite_terms;
}

HSpaceDistribution HubbardNeelState(int sites, const mat_t & projection) {
  vec_t state = vec_t::Unit(std::pow(4, sites), 0);
  vec_t state1 = vec_t::Unit(std::pow(4, sites), 0);
  for (int i = 0; i < sites; ++i) {
    if (i % 2 == 0) {
      state = nth_subsystem(HubbardOperators::c_up_t(), i, sites) * state;
      state1 = nth_subsystem(HubbardOperators::c_down_t(), i, sites) * state;
    } else {
      state = nth_subsystem(HubbardOperators::c_down_t(), i, sites) * state;
      state1 = nth_subsystem(HubbardOperators::c_up_t(), i, sites) * state;
    }
  }
  state /= state.norm();

  return HSpaceDistribution({1.0}, {projection * state});
}

HSpaceDistribution HubbardNeelState_sp(int sites, const spmat_t & projection) {
  vec_t state = vec_t::Unit(std::pow(4, sites), 0);
  for (int i = 0; i < sites; ++i) {
    if (i % 2 == 0) {
      state = n_th_subsystem_sp(HubbardOperators::c_up_t(), i, sites) * state;
    } else {
      state = n_th_subsystem_sp(HubbardOperators::c_down_t(), i, sites) * state;
    }
  }
  state /= state.norm();

  return HSpaceDistribution({1.0}, {projection * state});
}

HSpaceDistribution DimerGroundState(double hopping, double hubbardU) {
  double factor = - (hubbardU + std::sqrt(16.0 * hopping * hopping + hubbardU * hubbardU)) / (4.0 * hopping);
  Eigen::Matrix<double, 4, 1> tmp(1, factor, factor, 1);
  tmp /= tmp.norm();
  return HSpaceDistribution({1.0}, {tmp});
}

mat_t HubbardProjector(int sites, int total_spins_up, int total_spins_down) {
  int dimension = std::pow(4, sites);
  mat_t n_up_operator = sum_operator(HubbardOperators::n_up(), sites);
  mat_t n_down_operator = sum_operator(HubbardOperators::n_down(), sites);

  LOG_VAR(sparsity(n_up_operator));
  LOG_VAR(sparsity(n_down_operator));

  std::vector<int> projection_basis;
  for (int i = 0; i < dimension; ++i) {
    if (std::abs(total_spins_up - n_up_operator(i, i).real()) < tol
	&& std::abs(total_spins_down - n_down_operator(i, i).real()) < tol) {
      projection_basis.push_back(i);
    }
  }

  mat_t projection = mat_t::Zero(projection_basis.size(), dimension);
  for (size_type i = 0; i < projection_basis.size(); ++i) {
    projection.row(i) = vec_t::Unit(dimension, projection_basis[i]);
  }

  return projection;
}

spmat_t HubbardProjector_sp(int sites, int total_spins_up, int total_spins_down) {
  int dimension = std::pow(4, sites);
  spmat_t n_up_operator = sum_operator_sp(HubbardOperators::n_up(), sites);
  spmat_t n_down_operator = sum_operator_sp(HubbardOperators::n_down(), sites);
  vec_t n_up_operator_d = n_up_operator.diagonal();
  vec_t n_down_operator_d = n_down_operator.diagonal();
  
  std::vector<int> projection_basis;
  for (int i = 0; i < dimension; ++i) {
    if (std::abs(total_spins_up - n_up_operator_d(i).real()) < tol
	&& std::abs(total_spins_down - n_down_operator_d(i).real()) < tol) {
      projection_basis.push_back(i);
    }
  }


  using T = Eigen::Triplet<scalar_t>;
  std::vector<T> tripletList;
  spmat_t projection = mat_t::Zero(projection_basis.size(), dimension);
  for (size_type i = 0; i < projection_basis.size(); ++i) {
    tripletList.push_back(T(i, projection_basis[i], 1.0));
    // projection.row(i) = vec_t::Unit(dimension, projection_basis[i]);
  }

  projection.setFromTriplets(tripletList.begin(), tripletList.end());
  return projection;
}

spmat_t Hubbard_light_matter_sp(int photon_dimension,
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
  hopping_terms = proj * hopping_terms * proj.adjoint();
  onsite_terms = proj * onsite_terms * proj.adjoint();

  hopping_terms = Eigen::kroneckerProduct(e_iA, hopping_terms).eval();
  onsite_terms = tensor_identity_LHS(onsite_terms, photon_dimension).eval();
  spmat_t hopping_terms_adj = hopping_terms.adjoint();
  return hopping_terms + hopping_terms_adj + onsite_terms;
}

spmat_t Hubbard_hamiltonian_sp(int sites,
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
  hopping_terms = proj * hopping_terms * proj.adjoint();
  onsite_terms = proj * onsite_terms * proj.adjoint();
  return hopping_terms + spmat_t(hopping_terms.adjoint()) + onsite_terms;
}

#include "Lanczos.hpp"

HSpaceDistribution HubbardGroundState(int sites,
				      double hopping,
				      double hubbardU,
				      bool periodic,
				      const spmat_t & proj) {
  spmat_t hubbard = Hubbard_hamiltonian_sp(sites, hopping, hubbardU, periodic,
					   proj);
  int dimension = hubbard.rows();
  int niter = std::min(75, dimension / 2);
  return HSpaceDistribution({1.0}, {find_groundstate(hubbard, niter).second});
}


std::vector<int> HubbardProjector_basis(int sites, int total_spins_up,
					int total_spins_down) {
  int dimension = std::pow(4, sites);
  spmat_t n_up_operator = sum_operator_sp(HubbardOperators::n_up(), sites);
  spmat_t n_down_operator = sum_operator_sp(HubbardOperators::n_down(), sites);
  vec_t n_up_operator_d = n_up_operator.diagonal();
  vec_t n_down_operator_d = n_down_operator.diagonal();
  
  std::vector<int> projection_basis;
  for (int i = 0; i < dimension; ++i) {
    if (std::abs(total_spins_up - n_up_operator_d(i).real()) < tol
	&& std::abs(total_spins_down - n_down_operator_d(i).real()) < tol) {
      projection_basis.push_back(i);
    }
  }
  return projection_basis;
}
