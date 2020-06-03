#include "PadeExponential.hpp"

Eigen::SparseMatrix<double> spmat_abs(const spmat_t & A) {
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;

  tripletList.reserve(A.nonZeros());
  for (int k=0; k < A.outerSize(); ++k) {
    for (typename spmat_t::InnerIterator it(A, k); it; ++it) {
      tripletList.push_back(T(it.row(), it.col(), std::abs(it.value())));
    }
  }
  Eigen::SparseMatrix<double> out(A.rows(), A.cols());
  out.setFromTriplets(tripletList.begin(), tripletList.end());
  return out;
}


double onenorm_power(const spmat_t & A, size_type power) {
  assert(A.rows() == A.cols());
  Eigen::VectorXd v = Eigen::VectorXd::Constant(A.rows(), 1.0);

  Eigen::SparseMatrix<double> A_abs = spmat_abs(A);
  for (size_type i = 0; i < power; ++i) {
    v = A_abs * v;
  }
  return v.maxCoeff();
}

int _ell(const spmat_t & A, int m) {
  assert(A.cols() == A.rows());
  int p = 2*m + 1;

  int choose_2p_p = binomial(2 * p, p);
  double abs_c_recip = static_cast<double>(choose_2p_p * factorial(2 * p + 1));

  // unit roundoff
  double u = std::pow(2, -53);
  double A_abs_onenorm = onenorm_power(A, p);

  if (A_abs_onenorm == 0.0) return 0;

  double alpha = A_abs_onenorm / (A.oneNorm() * abs_c_recip);
  double log2_alpha_div_u = std::log2(alpha / u);
  int value = static_cast<int>(std::ceil(log2_alpha_div_u / (2 * m)));
  return std::max(value, 0);
}

spmat_t expm(const spmat_t & A) {
  ExpmPadeHelper<spmat_t> h(A);
  double eta_1 = std::max(h.d4(), h.d6());
  if ((eta_1 < 1.495585217958292e-002) && (_ell(h.A(), 3) == 0)) {
    return solve_P_Q(h.pade3());
  }

  double eta_2 = std::max(h.d4(), h.d6());
  if ((eta_2 < 2.539398330063230e-001) && (_ell(h.A(), 5) == 0)) {
    return solve_P_Q(h.pade5());
  }
    
  double eta_3 = std::max(h.d6(), h.d8());
  if ((eta_3 < 9.504178996162932e-001) && (_ell(h.A(), 7) == 0)) {
    return solve_P_Q(h.pade7());
  }
        
  if ((eta_3 < 2.097847961257068e+000) && (_ell(h.A(), 9) == 0)) {
    return solve_P_Q(h.pade9());
  }

  double eta_4 = std::max(h.d8(), h.d10());
  double eta_5 = std::min(eta_3, eta_4);
  double theta_13 = 4.25;
  int s = std::max(static_cast<int>(std::ceil(std::log2(eta_5 / theta_13))), 0);
  s = s + _ell(std::pow(2, -s) * h.A(), 13);
  spmat_t X = solve_P_Q(h.pade13_scaled(s));

  for (int i = 0; i < s; ++i) {
    X = X * X;
  }
  return X;
}

spmat_t solve_P_Q(const PadePair<spmat_t> & p) {
  spmat_t P = p.U + p.V;
  spmat_t Q = -p.U + p.V;

  if (sparsity(Q) < 0.1) {
    Eigen::SparseLU<spmat_t> solver(Q);
    spmat_t result = solver.solve(P);
    LOG_VAR(sparsity(result));
    return result;
  } else {
    Eigen::PartialPivLU<mat_t> solver(Q);
    return solver.solve(mat_t(P));
  }

}
