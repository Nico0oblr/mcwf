#ifndef SD_HPP
#define SD_HPP

#include "Common.hpp"

std::pair<int, int>
tensor_basis_index(int index, int left_dim, int right_dim);

spmat_t vector_to_matrix(const vec_t & vec, int left_dim);

spmat_t vector_to_matrix(const vec_t & vec, int left_dim,
			 const std::vector<int> & projection,
			 int total_dimension);

std::tuple<mat_t, vec_t, mat_t> compute_svd(const spmat_t & mat);

#endif /* SD_HPP */
