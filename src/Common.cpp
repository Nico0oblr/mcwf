#include "Common.hpp"

mat_t matrix_exponential(const mat_t & matrix) {
  Eigen::ComplexEigenSolver<mat_t> solver(matrix, true);
  mat_t V = solver.eigenvectors();
  mat_t expD = solver.eigenvalues().array().exp().matrix().asDiagonal();
  Eigen::PartialPivLU<mat_t> inverter(V);
  mat_t V_inv = inverter.inverse();
  return V * expD * V_inv;
}

vec_t add_vectors(const vec_t & vec1, const vec_t & vec2) {
  vec_t out = vec_t::Zero(vec1.size() + vec2.size());
  out.head(vec1.size()) = vec1;
  out.tail(vec2.size()) = vec2;
  return out;
}

mat_t superoperator_left(const mat_t & op, int dimension) {
  return tensor_identity(op, dimension);
}

mat_t superoperator_right(const mat_t & op, int dimension) {
  return tensor_identity_LHS<mat_t>(op.transpose(), dimension);
}

vec_t unstack_matrix(const mat_t & mat) {
  vec_t out(mat.rows() * mat.cols());

  /*for (int i = 0; i < mat.cols(); ++i) {
    auto vec_seq = Eigen::seq(mat.rows() * i, mat.rows() * (i + 1) - 1);
    out(vec_seq) = mat.col(i);
    }*/

  for (int i = 0; i < mat.rows(); ++i) {
    auto vec_seq = Eigen::seq(mat.cols() * i, mat.cols() * (i + 1) - 1);
    out(vec_seq) = mat.row(i);
  }
  return out;
}

vec_t unstack_matrix_alt(const mat_t & mat) {
  vec_t out = vec_t::Zero(mat.rows() * mat.cols());
  for (int i = 0; i < mat.rows(); ++i) {
    vec_t unit_i = vec_t::Zero(mat.rows());
    unit_i(i) = 1.0;
    
    for (int j = 0; j < mat.cols(); ++j) {
      vec_t unit_j = vec_t::Zero(mat.rows());
      unit_j(j) = 1.0;
      out += mat(i, j) * Eigen::kroneckerProduct(unit_i, unit_j);
    }
  }
  return out;
}

mat_t restack_vector(const vec_t & vec, int dimension) {
  assert(vec.size() == dimension * dimension);
  mat_t out(dimension, dimension);

  for (int i = 0; i < out.rows(); ++i) {
    auto vec_seq = Eigen::seq(out.cols() * i, out.cols() * (i + 1) - 1);
    out.row(i) = vec(vec_seq);
  }
  return out;
}

double dmat_sparsity(const mat_t &mat) {
  double zeros = (mat.array().abs() < tol).count();
  double entries = mat.size();
  return zeros / entries;
}

int factorial(int n) {
  return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

int binomial(int n, int k) {
  double res = 1;
  for (int i = 1; i <= k; ++i) {
    res = res * (n - k + i) / i;
  }
  return static_cast<int>(res + 0.01);
}

int minus_one_power(int n) {
  if (n < 0) n = -n;
  if (n % 2 == 1) return -1;
  return 1;
}

double poisson(double N, int n) {
  double out = -N + n * std::log(N);
  for (int m = 1; m <= n; ++m) out -= std::log(m);
  return std::exp(0.5 * out);
}

spmat_t cI(Eigen::Index n, std::complex<double> value) {
  spmat_t Ident(n, n);
  Ident.setIdentity();
  Ident *= value;
  return Ident;
}

spmat_t matrix_exponential_taylor(const spmat_t & matrix) {
  return cI(matrix.cols(), 1.0) + matrix + 0.5 * matrix * matrix;
}

mat_t matrix_exponential_taylor(const mat_t & matrix) {
  return mat_t::Identity(matrix.cols(), matrix.rows()) + matrix + 0.5 * matrix * matrix;
}


double expval(const mat_t & observable, const vec_t & state) {
  return state.dot(observable * state).real();
}
