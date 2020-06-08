#include "Operators.hpp"

void PrecomputedOperators_str::precompute(int dimension) {
  spmat_t A_m = spmat_t::Identity(dimension, dimension);
  spmat_t A_t_m = spmat_t::Identity(dimension, dimension);
  spmat_t n_m = spmat_t::Identity(dimension, dimension);
  
  spmat_t A = annihilationOperator_sp(dimension);
  spmat_t A_t = creationOperator_sp(dimension);
  spmat_t n = numberOperator_sp(dimension);
  
  for (int i = 0; i <= dimension; ++i) {
    A_powers.push_back(A_m);
    A_t_powers.push_back(A_t_m);
    n_powers.push_back(n_m);
    A_m = A_m * A;
    A_t_m = A_t_m * A_t;
    n_m = n_m * n;
  }

  m_dimension = dimension;
}


spmat_t PrecomputedOperators_str::A_t(int power) const {
  if (power <= m_dimension) {
    return A_t_powers.at(power);
  } else {
    return spmat_t(m_dimension, m_dimension);
  }
}
  
spmat_t PrecomputedOperators_str::A(int power) const {
  if (power <= m_dimension) {
    return A_powers.at(power);
  } else {
    return spmat_t(m_dimension, m_dimension);
  }
}
  
spmat_t PrecomputedOperators_str::n(int power) const {
  if (power <= m_dimension) {
    return n_powers.at(power);
  } else {
    return spmat_t(m_dimension, m_dimension);
  }
}

PrecomputedOperators_str PrecomputedOperators;

mat_t creationOperator(int dimension) {
  mat_t op = mat_t::Zero(dimension, dimension);
  for (Eigen::Index i = 1; i < dimension; ++i) {
    op(i, i - 1) = std::sqrt(i);
  }
  return op;
}

mat_t annihilationOperator(int dimension) {
  mat_t op = mat_t::Zero(dimension, dimension);
  for (Eigen::Index i = 1; i < dimension; ++i) {
    op(i - 1, i) = std::sqrt(i);
  }
  return op;
}

mat_t numberOperator(int dimension) {
 mat_t op = mat_t::Zero(dimension, dimension);
  for (Eigen::Index i = 0; i < dimension; ++i) {
    op(i, i) = i;
  }
  return op;
}

spmat_t creationOperator_sp(int dimension) {
  using T = Eigen::Triplet<scalar_t>;
  std::vector<T> tripletList;
  tripletList.reserve(dimension);
  for (Eigen::Index i = 1; i < dimension; ++i) {
    tripletList.push_back(T(i, i - 1, std::sqrt(i)));
  }
  Eigen::SparseMatrix<scalar_t> out(dimension, dimension);
  out.setFromTriplets(tripletList.begin(), tripletList.end());
  return out;
}

spmat_t annihilationOperator_sp(int dimension) {
  using T = Eigen::Triplet<scalar_t>;
  std::vector<T> tripletList;
  tripletList.reserve(dimension);
  for (Eigen::Index i = 1; i < dimension; ++i) {
    tripletList.push_back(T(i - 1, i, std::sqrt(i)));
  }
  Eigen::SparseMatrix<scalar_t> out(dimension, dimension);
  out.setFromTriplets(tripletList.begin(), tripletList.end());
  return out;
}

spmat_t numberOperator_sp(int dimension) {
  using T = Eigen::Triplet<scalar_t>;
  std::vector<T> tripletList;
  tripletList.reserve(dimension);
  for (Eigen::Index i = 0; i < dimension; ++i) {
    tripletList.push_back(T(i, i, i));
  }
  Eigen::SparseMatrix<scalar_t> out(dimension, dimension);
  out.setFromTriplets(tripletList.begin(), tripletList.end());
  return out;
}

mat_t exchange_interaction(int dimension,
			   double hubbardU,
			   double hopping,
			   double frequency,
			   double coupling) {
  double g_sq = coupling * coupling;
  double wbar = frequency / hubbardU;
  double wbar_sq = wbar * wbar;
  double wbar_quad = wbar_sq * wbar_sq;
  double J_ex = 4.0 * hopping * hopping / hubbardU;
  mat_t A = annihilationOperator(dimension);
  mat_t A_t = creationOperator(dimension);
  mat_t id = mat_t::Identity(dimension, dimension);
  mat_t J_0 = (1 - g_sq * wbar / (1.0 + wbar)) * id
    + g_sq * A_t * A * 2.0 * wbar_sq / (1.0 - wbar_sq);
  mat_t J_2 = id * g_sq * (wbar_sq + 2.0 * wbar_quad)
    / ((1 - 4.0 * wbar_sq) * (1 - wbar_sq));
  mat_t out = J_ex * (J_0 + A_t * A_t * J_2 + A * A * J_2);
  return out;
}

double L_p(double omega_bar, double coupling, int p) {
  double g_sq = coupling * coupling;
  double out = 0;
  // Essentially sums to infinity
  for (int r = 0; r < 7; ++r) {
    if (std::abs(1.0 + static_cast<double>(r + p) * omega_bar) < tol) {
      continue;
    }
    out += std::pow(g_sq, 1.0 * r) / (static_cast<double>(factorial(r)))
      * 1.0 / (1.0 + static_cast<double>(r + p) * omega_bar);
  }

  return std::exp(-g_sq) * out;
}

double L_c_m(double omega_bar, double coupling, int c, int m) {
  double out = 0.0;
  for (int p = 0; p <= 2 * (c + m); ++p) {
    out += minus_one_power(p) * binomial(2 * c + 2 * m, p)
      * (L_p(omega_bar, coupling, p - c) + L_p(omega_bar, coupling, p - c - 2 * m));
  }

  return 1.0 / (2.0 * factorial(c + 2 * m) * factorial(c)) * out;
}

spmat_t exchange_interaction_term(int m,
				  double coupling,
				  double omega_bar,
				  int dimension) {
  spmat_t out(dimension, dimension);
  // c + m < 10 is a numerical cutoff, otherwise you get integer overflow etc.
  // c + m >= 10 essentially vanishes anyways
  for (int c = 0; c + m <= dimension && c + m < 10; ++c) {
    spmat_t term = std::pow(coupling, 2.0 * c + 2.0 * m)
      * PrecomputedOperators.A_t(c)
      * PrecomputedOperators.A(c)
      * L_c_m(omega_bar, coupling, c, m);
    out += term;
  }
  return out;
}

/*
  g^{2c+2m}L_{c,m}
  L_p -> \sum_r g^{2r}
  2c+2m+2r < order
*/
spmat_t exchange_interaction_full(int dimension,
				  double hubbardU,
				  double hopping,
				  double frequency,
				  double coupling,
				  int order) {
  PrecomputedOperators.precompute(dimension);
  double wbar = frequency / hubbardU;
  double J_ex = 4.0 * hopping * hopping / hubbardU;
  spmat_t J0 = exchange_interaction_term(0, coupling, wbar, dimension);
  spmat_t JI(J0.rows(), J0.cols());

  for (int m = 1; m < order; ++m) {
    spmat_t term = PrecomputedOperators.A_t(2 * m)
      * exchange_interaction_term(m, coupling, wbar, dimension);
    JI += term;
  }

  JI += spmat_t(JI.adjoint());
  return J_ex * (J0 + JI);
}


mat_t J0(int dimension,
	 double hubbardU,
	 double hopping,
	 double frequency,
	 double coupling) {
  double wbar = frequency / hubbardU;
  double J_ex = 4.0 * hopping * hopping / hubbardU;
  spmat_t J_0 = exchange_interaction_term(0, coupling, wbar, dimension);
  return J_ex * J_0;
}


mat_t nth_subsystem(const mat_t & op,
		    int n_subsystem,
		    int n_subsystems) {
  assert(n_subsystem < n_subsystems);
  mat_t out;
  mat_t id = mat_t::Identity(op.cols(), op.rows());

  if (n_subsystem == 0) {
    out = op;
  } else {
    out = id;
  }
  
  for (int i = 1; i < n_subsystems; ++i) {
    if (i == n_subsystem) {
      out = Eigen::kroneckerProduct(out, op).eval();
    } else {
      out = Eigen::kroneckerProduct(out, id).eval();
    }
  }

  return out;
}

std::vector<mat_t> operator_vector(const Eigen::Ref<const mat_t> & op,
				   int sites) {
  std::vector<mat_t> matrices;
  matrices.reserve(sites);
  for (int i = 0; i < sites; ++i) {
    matrices.push_back(nth_subsystem(op, i, sites));
  }
  return matrices;
}

mat_t sum_operator(const mat_t & op,
		   int sites) {
  assert(op.cols() == op.rows());
  int dimension = std::pow(op.cols(), sites);
  mat_t total = mat_t::Zero(dimension, dimension);

  for (int i = 0; i < sites; ++i) {
    total += nth_subsystem(op, i, sites);
  }
  return total;
}

spmat_t n_th_subsystem_sp(const spmat_t & op,
			  int n_subsystem,
			  int n_subsystems) {
  int left_dimension = n_subsystem;
  int right_dimension = n_subsystems - n_subsystem - 1;
  int system_dim = op.rows();
  assert(left_dimension >= 0);
  assert(right_dimension >= 0);
  assert(op.rows() == op.cols());
  int ldim = std::pow(system_dim, left_dimension);
  int rdim = std::pow(system_dim, right_dimension);
  
  if ((right_dimension == 0) && (left_dimension == 0)) {
    return op;
  }
  if (right_dimension == 0) {
    return tensor_identity_LHS(op, ldim);
  } else if (left_dimension == 0) {
    return tensor_identity(op, rdim);
  } else {
    spmat_t tmp = tensor_identity(op, rdim);
    return tensor_identity_LHS(tmp, ldim);
  }
}

spmat_t sum_operator_sp(const spmat_t & op, int sites) {
  assert(sites > 0);
  spmat_t out = n_th_subsystem_sp(op, 0, sites);
  for (int i = 1; i < sites; ++i) {
    out += n_th_subsystem_sp(op, i, sites);
  }
  return out;
}
