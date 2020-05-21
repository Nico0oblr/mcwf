#include "Operators.hpp"

#include "HSpaceDistribution.hpp"

namespace HubbardOperators {
  mat_t c_up_t() {
    mat_t out(4, 4);
    out <<
      0, 0, 0, 0,
      0, 0, 0, 0,
      1, 0, 0, 0,
      0, 1, 0, 0;
    return out;
  }

  mat_t c_down_t() {
    mat_t out(4, 4);
    out <<
      0, 0, 0, 0,
      1, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 1, 0;
    return out;
  }

  mat_t c_down() {
    return c_down_t().adjoint();
  }

  mat_t c_up() {
    return c_up_t().adjoint();
  }

  mat_t n_down() {
    mat_t out(4, 4);
    out <<
      0, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 1;
    return out;
  }

  mat_t n_up() {
    mat_t out(4, 4);
    out <<
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1;
    return out;
  }
  
}

mat_t Hubbard_light_matter(int photon_dimension,
			   int sites,
			   double coupling,
			   double hopping,
			   double hubbardU,
			   bool periodic) {
  mat_t argument = 1.0i * coupling * (creationOperator(photon_dimension)
				      + annihilationOperator(photon_dimension));
  mat_t e_iA = matrix_exponential(argument);
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

  hopping_terms = Eigen::kroneckerProduct(e_iA, hopping * hopping_terms).eval();
  onsite_terms = tensor_identity_LHS(hubbardU * onsite_terms,
				     photon_dimension).eval();
  return hopping_terms + hopping_terms.adjoint() + onsite_terms;
}

mat_t Hubbard_light_matter(int photon_dimension,
			   int sites,
			   double coupling,
			   double hopping,
			   double hubbardU,
			   bool periodic);
HSpaceDistribution HubbardNeelState(int sites, const mat_t & projection) {
  int dimension = std::pow(4, sites);
  vec_t state = vec_t::Zero(dimension);
  state(0) = 1.0;

  for (int i = 0; i < sites; ++i) {
    if (i % 2 == 0) {
      mat_t op = nth_subsystem(HubbardOperators::c_up_t(), i, sites);
      state = op * state;
    } else {
      mat_t op = nth_subsystem(HubbardOperators::c_down_t(), i, sites);
      state = op * state;
    }
  }
  state /= state.norm();

  std::cout << "n0up: " << state.dot(nth_subsystem(HubbardOperators::n_up(), 0, sites) * state).real() << std::endl;
  std::cout << "n1up: " << state.dot(nth_subsystem(HubbardOperators::n_up(), 1, sites) * state).real() << std::endl;
  std::cout << "n0down: " << state.dot(nth_subsystem(HubbardOperators::n_down(), 0, sites) * state).real() << std::endl;
  std::cout << "n1down: " << state.dot(nth_subsystem(HubbardOperators::n_down(), 1, sites) * state).real() << std::endl;
  
  return HSpaceDistribution({1.0}, {projection * state});
}

mat_t HubbardProjector(int sites, int total_spins_up, int total_spins_down) {
  int dimension = std::pow(4, sites);
  mat_t n_up_operator = mat_t::Zero(dimension, dimension);
  mat_t n_down_operator = mat_t::Zero(dimension, dimension);
  for (int i = 0; i < sites; ++i) {
    n_up_operator += nth_subsystem(HubbardOperators::n_up(), i, sites);
    n_down_operator += nth_subsystem(HubbardOperators::n_down(), i, sites);
  }

  std::vector<vec_t> projection_basis;
  for (int i = 0; i < dimension; ++i) {
    vec_t basis_i = vec_t::Zero(dimension);
    basis_i(i) = 1.0;
    double num_up = basis_i.dot(n_up_operator * basis_i).real();
    double num_down = basis_i.dot(n_down_operator * basis_i).real();

    if (std::abs(total_spins_up - num_up) < tol
	&& std::abs(total_spins_down - num_down) < tol) {
      projection_basis.push_back(basis_i);
    }
  }

  mat_t projection = mat_t::Zero(projection_basis.size(), dimension);
  for (int i = 0; i < projection_basis.size(); ++i) {
    projection.row(i) = projection_basis[i];
  }

  print_matrix_dim(projection);
  return projection;
}
