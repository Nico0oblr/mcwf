#include "Common.hpp"

vec_t add_vectors(const vec_t & vec1, const vec_t & vec2) {
  vec_t out = vec_t::Zero(vec1.size() + vec2.size());
  out.head(vec1.size()) = vec1;
  out.tail(vec2.size()) = vec2;
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
