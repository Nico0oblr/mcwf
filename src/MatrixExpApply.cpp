#include "MatrixExpApply.hpp"


long _compute_cost_div_m(int m, int p,
			 LazyOperatorNormInfo<spmat_t> & norm_info) {
  return static_cast<long>(std::ceil(norm_info.alpha(p) / _theta.at(m)) + 0.5);
}

int compute_p_max(int m_max) {
  double sqrt_m_max = std::sqrt(m_max);
  int p_low = static_cast<int>(std::floor(sqrt_m_max) + 0.5);
  int p_high = static_cast<int>(std::ceil(sqrt_m_max + 1) + 0.5);

  int pmax = p_low;
  for (int p = p_low; p <= p_high; ++p) {
    if (p * (p - 1) <= m_max + 1) pmax = p;
  }

  return pmax;
}

std::pair<int, long> _fragment_3_1(LazyOperatorNormInfo<spmat_t> & norm_info,
				   int n0, double tolerance, int m_max,
				   int ell) {
  assert(tolerance == std::pow(2.0, -53.0));
  assert(ell > 0 && "expected ell to be a positive integer");
  int best_m = -1;
  long best_s = -1;
  if (_condition_3_13(norm_info.onenorm(), n0, m_max, ell)) {
    for (auto it = _theta.begin(); it != _theta.end(); ++it) {
      int m = it->first;
      double theta = it->second;
      long s = static_cast<long>(std::ceil(norm_info.onenorm() / theta) + 0.5);
      assert(m > 0);
      assert(s > 0);
      if ((best_m < 0) || (m * s < best_m * best_s)) {
	best_m = m;
	best_s = s;
      }
    }
  } else {
    //  Equation (3.11).
    for (int p = 2; p < compute_p_max(m_max) + 1; ++p) {
      for (int m = p * (p - 1) - 1; m < m_max + 1; ++m) {
	if (_theta.find(m) != _theta.end()) {
	  long s = _compute_cost_div_m(m, p, norm_info);
	  assert(m > 0);
	  assert(s > 0);
	  if ((best_m < 0) || (m * s < best_m * best_s)) {
	    best_m = m;
	    best_s = s;		  
	  }
	}
      }
    }
    best_s = std::max(best_s, 1l);
  }

  return std::pair<int, long>(best_m, best_s);
}

bool _condition_3_13(double A_1_norm, int n0, int m_max, int ell) {
  // This is the rhs of equation (3.12).
  int p_max = compute_p_max(m_max);
  int a = 2 * ell * p_max * (p_max + 3);

  // Evaluate the condition (3.13).
  double b = _theta.at(m_max) / static_cast<double>(n0 * m_max);
  return A_1_norm <= a * b;
}

vec_t expm_multiply_simple(const LinearOperator<spmat_t> & _A,
			   const vec_t & B,
			   double t) {
  lo_ptr A = _A.clone();
  assert(A->cols() == B.rows()
	 && "the matrices A and B have incompatible shapes");
  spmat_t ident = spmat_t::Identity(A->rows(), A->cols());
  int n = A->rows();
  int n0 = B.cols();
  double u_d = std::pow(2.0, -53.0);
  double tolerance = u_d;
  std::complex<double> mu = 0.0;//A->eval().trace() / static_cast<double>(n);
  // A = A - operatorize(spmat_t(mu * ident));
  double A_1_norm = onenormest(*A);
  if (std::abs(t * A_1_norm) < tol) {
    int m_star = 0;
    long s = 1;
    return _expm_multiply_simple_core(*A, B, t, mu, m_star, s, tolerance);
  } else {
    int ell = 2;
    LazyOperatorNormInfo<spmat_t> norm_info(*(t * A), t * A_1_norm, ell);
    auto tmp = _fragment_3_1(norm_info, n0, tolerance, 55, ell);
    int m_star = tmp.first;
    long s = tmp.second;
    return _expm_multiply_simple_core(*A, B, t, mu, m_star, s, tolerance);
  }
}

vec_t _expm_multiply_simple_core(const LinearOperator<spmat_t> & A,
				 const vec_t & _B,
				 double t, std::complex<double> mu,
				 int m_star, long s,
				 double tolerance) {
  int matmuls = 0;
  assert(tolerance == std::pow(2.0, -53.0));
  vec_t B = _B;
  vec_t F = B;
  std::complex<double> eta = std::exp(t * mu / static_cast<double>(s));
  for (long i = 0; i < s; ++i) {
    double c1 = B.lpNorm<Eigen::Infinity>();
    for (int j = 0; j < m_star; ++j) {
      double coeff = t / static_cast<double>(s * (j + 1));
      B = coeff * (A * B);
      ++matmuls;
      double c2 = B.lpNorm<Eigen::Infinity>();
      F = F + B;
      if (c1 + c2 <= tolerance * F.lpNorm<Eigen::Infinity>()) break;
      c1 = c2;
    }

    F = eta * F;
    B = F;
  }

  // LOG_VAR(matmuls);
  return F;
}

vec_t expm_multiply_simple(const spmat_t & A, const vec_t & B,
			   double t) {
  return expm_multiply_simple(*operatorize(A), B, t);
}
