#include "Kron.hpp"

vec_t kroneckerApply(const spmat_t & A,
		     const spmat_t & B,
		     const vec_t & vec) {
  assert(vec.size() == A.cols() * B.cols()
	 && "Vector must have dimA * dimB in tensor space");
  mat_t Bvec = B * Eigen::Map<const mat_t>(vec.data(), B.cols(), A.cols());
  vec_t out = vec_t::Zero(vec.size());
  for (int k = 0; k < A.outerSize(); ++k) {
    for (spmat_t::InnerIterator it(A, k); it; ++it) {
      int i = it.row();
      int j = it.col();
      out(Eigen::seq(i * B.cols(), (i + 1) *  B.cols() - 1))
	+= it.value() * Bvec.col(j);
    }
  }

  return out;
}

vec_t kroneckerApply_id(const spmat_t & A,
			int codimension,
			const vec_t & vec) {
  assert(vec.size() == A.cols() * codimension);
  vec_t out = vec_t::Zero(vec.size());
  for (int k = 0; k < A.outerSize(); ++k) {
    for (spmat_t::InnerIterator it(A, k); it; ++it) {
      int i = it.row();
      int j = it.col();
      out(Eigen::seq(i * codimension, (i + 1) * codimension - 1))
	+= it.value()
	* vec(Eigen::seq(j * codimension, (j + 1) * codimension - 1));
    }
  }
  return out;
}

/*vec_t kroneckerApply_id(const spmat_t & A,
			int codimension,
			const vec_t & vec) {
  assert(vec.size() == A.cols() * codimension
	 && "Vector must have dimA * codimension in tensor space");
  mat_t ABvec = Eigen::Map<const mat_t>(vec.data(), codimension, A.cols()) * A;
  return Eigen::Map<vec_t>(ABvec.data(), vec.size());
  }*/

vec_t kroneckerApply_LHS(const spmat_t & B,
			 int codimension,
			 const vec_t & vec) {
  assert(vec.size() == B.cols() * codimension
	 && "Vector must have codimension * dimB in tensor space");
  mat_t ABvec = B * Eigen::Map<const mat_t>(vec.data(), B.cols(), codimension);
  return Eigen::Map<vec_t>(ABvec.data(), vec.size());
}
