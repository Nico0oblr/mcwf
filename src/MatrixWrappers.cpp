#include "MatrixWrappers.hpp"

vec_t kroneckerApply(const spmat_t & A,
		     const spmat_t & B,
		     const vec_t & vec) {
  assert(vec.size() == A.cols() * B.cols());
  Eigen::Map<const mat_t> vec_view(vec.data(), B.cols(), A.cols());
  mat_t Bvec = B * vec_view;
  // std::vector<vec_t> Bout;
  // Bout.reserve(A.rows());

  // for (int i = 0; i < A.rows(); ++i) {
  //Bout.emplace_back(B * vec(Eigen::seq(i * B.cols(), (i + 1) *  B.cols() - 1)));
  // }

  /*vec_t out = vec_t::Zero(vec.size());
  for (int k = 0; k < A.outerSize(); ++k) {
    for (spmat_t::InnerIterator it(A, k); it; ++it) {
      int i = it.row();
      int j = it.col();
      out(Eigen::seq(i * B.cols(), (i + 1) *  B.cols() - 1))
	+= it.value() * Bout[j];
    }
    }

    return out;
  */

  print_matrix_dim(Bvec);
  print_matrix_dim(A);
  mat_t ABvec = Bvec * A;
  return Eigen::Map<vec_t>(ABvec.data(), vec.size());
}

vec_t kroneckerApplyLazy(const spmat_t & A,
			 const spmat_t & B,
			 const vec_t & vec) {
  assert(vec.size() == A.cols() * B.cols());


  struct LazyEval {

    const vec_t & operator()(int i) {
      if (m_bout[i].size() == 0) {
	m_bout[i] = (*m_B) * (*m_vec)(Eigen::seq(i * m_B->cols(),
						 (i + 1) *  m_B->cols() - 1));
      }
      return m_bout[i];
    }
    
    LazyEval(const spmat_t & c_B,
	     const vec_t & c_vec,
	     int rows)
      :m_B(&c_B),
       m_vec(&c_vec) {
      m_bout.resize(rows);
    };
    
    std::vector<vec_t> m_bout;
    const spmat_t * m_B;
    const vec_t * m_vec;
  };

  LazyEval ev(B, vec, A.rows());
  vec_t out = vec_t::Zero(vec.size());
  for (int k = 0; k < A.outerSize(); ++k) {
    for (spmat_t::InnerIterator it(A, k); it; ++it) {
      int i = it.row();
      int j = it.col();
      out(Eigen::seq(i * B.cols(), (i + 1) *  B.cols() - 1))
	+= it.value() * ev(j);
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
	+= it.value() * vec;
    }
  }
  return out;
}

vec_t kroneckerApply_LHS(const spmat_t & B,
			 int codimension,
			 const vec_t & vec) {
  assert(vec.size() == B.cols() * codimension);

  vec_t out = vec_t::Zero(vec.size());
  for (int i = 0; i < codimension; ++i) {
      out(Eigen::seq(i * B.cols(), (i + 1) *  B.cols() - 1))
	+= B * vec(Eigen::seq(i * B.cols(), (i + 1) *  B.cols() - 1));
  }
  return out;
}
