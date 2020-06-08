#include "SD.hpp"

#include "RedSvd.hpp"

std::pair<int, int>
tensor_basis_index(int index, int left_dim, int right_dim) {
  int lhs = index / right_dim;
  int rhs = index - lhs * right_dim;
  // assert(rhs * left_dim + lhs == index);
  return std::make_pair(lhs, rhs);
}


spmat_t vector_to_matrix(const vec_t & vec, int left_dim) {
  int right_dim = vec.size() / left_dim;

  typedef Eigen::Triplet<scalar_t> T;
  std::vector<T> tripletList;
  for (int i = 0; i < vec.size(); ++i) {
    if (std::abs(vec(i)) < tol) continue;
    auto ind_pair = tensor_basis_index(i, left_dim, right_dim);
    tripletList.push_back(T(ind_pair.first, ind_pair.second, vec(i)));
  }

  spmat_t out(left_dim, right_dim);
  out.setFromTriplets(tripletList.begin(), tripletList.end());
  return out;
}

spmat_t vector_to_matrix(const vec_t & vec, int left_dim,
			 const std::vector<int> & projection,
			 int total_dimension) {
  assert(projection.size() == vec.size());
  int right_dim = total_dimension / left_dim;

  typedef Eigen::Triplet<scalar_t> T;
  std::vector<T> tripletList;
  for (int i = 0; i < vec.size(); ++i) {
    if (std::abs(vec(i)) < tol) continue;
    int index = projection[i];
    auto ind_pair = tensor_basis_index(index, left_dim, right_dim);

    assert(ind_pair.first < left_dim);
    assert(ind_pair.second < right_dim);
    tripletList.push_back(T(ind_pair.first, ind_pair.second, vec(i)));
  }

  spmat_t out(left_dim, right_dim);
  out.setFromTriplets(tripletList.begin(), tripletList.end());
  return out;
}

std::tuple<mat_t, vec_t, mat_t> compute_svd(const spmat_t & mat) {
  RedSVD::RedSVD<spmat_t> svd(mat);
  return std::make_tuple(svd.matrixU(), svd.singularValues(), svd.matrixV());
}
