#include "MatrixWrappers.hpp"

vec_t kroneckerApply(const spmat_t & A,
		     const spmat_t & B,
		     const vec_t & vec) {
  assert(vec.size() == A.cols() * B.cols()
	 && "Vector must have dimA * dimB in tensor space");
  mat_t ABvec = B * Eigen::Map<const mat_t>(vec.data(), B.cols(), A.cols()) * A;
  return Eigen::Map<vec_t>(ABvec.data(), vec.size());
}

vec_t kroneckerApply_id(const spmat_t & A,
			int codimension,
			const vec_t & vec) {
  assert(vec.size() == A.cols() * codimension
	 && "Vector must have dimA * codimension in tensor space");
  mat_t ABvec = Eigen::Map<const mat_t>(vec.data(), codimension, A.cols()) * A;
  return Eigen::Map<vec_t>(ABvec.data(), vec.size());
}

vec_t kroneckerApply_LHS(const spmat_t & B,
			 int codimension,
			 const vec_t & vec) {
  assert(vec.size() == B.cols() * codimension
	 && "Vector must have codimension * dimB in tensor space");
  mat_t ABvec = B * Eigen::Map<const mat_t>(vec.data(), B.cols(), codimension);
  return Eigen::Map<vec_t>(ABvec.data(), vec.size());
}
