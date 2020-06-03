#include "OneNormEst.hpp"

mat_t sign_round_up(const mat_t X) {
  return X.unaryExpr([](auto v)
		     { return v == 0.0 ? 1.0 : v / std::abs(v);});
}

// _max_abs_axis1(X)
Eigen::VectorXd sum_abs_colwise(const mat_t & X) {
  return X.cwiseAbs().colwise().sum();
}

// _sum_abs_axis0(X):
Eigen::VectorXd max_abs_rowwise(const mat_t & X) {
  return X.cwiseAbs().rowwise().maxCoeff();
}

// LOL at the name
bool every_col_of_X_is_parallel_to_a_col_of_Y(const mat_t & X,
					      const mat_t & Y) {
  for (Eigen::Index i = 0; i < X.cols(); ++i) {
    bool any_parallel = false;
    for (Eigen::Index j = 0; j < Y.cols(); ++j) {
      if (vectors_parallel(X.col(i), Y.col(j))) {
	  any_parallel = true;
	  break;
	}
    }
    if (!any_parallel) return false;
  }
  return true;
}


bool column_needs_resampling(int i, const mat_t & X) {
  for (Eigen::Index j = 0; j < i; ++j) {
    if (vectors_parallel(X.col(i), X.col(j))) return true;
  }
  return false;
}
/*
  column i of X needs resampling if either
  it is parallel to a previous column of X or
  it is parallel to a column of Y
*/
bool column_needs_resampling(int i, const mat_t & X, const mat_t & Y) {
  if (column_needs_resampling(i, X)) return true;
  for (Eigen::Index j = 0; j < Y.cols(); ++j) {
    if (vectors_parallel(X.col(i), Y.col(j))) return true;
  }
  return false;
}


Eigen::VectorXi randint(int low, int high, Eigen::Index size) {
  assert(high > low);
  std::uniform_int_distribution<int> distribution(low, high - 1);
  Eigen::VectorXi out(size);
  for (Eigen::Index i = 0; i < size; ++i) out(i) = distribution(mt_rand);
  return out;
}

bool close(double a, double b, double rtol, double atol) {
  return std::abs(a-b) <= std::max(rtol * std::max(std::abs(a), std::abs(b)),
				   atol);
}

bool less_than_or_close(double a, double b) {
  return close(a, b) || (a < b);
}

std::vector<int> invert_indexer(std::vector<int> indexer, int size) {
  std::sort(indexer.begin(), indexer.end());
  std::vector<int> asc(size);
  std::iota (std::begin(asc), std::end(asc), 0);

  std::vector<int> out;
  std::set_difference(asc.begin(), asc.end(),
		      indexer.begin(), indexer.end(), std::back_inserter(out));
  return out;
}
