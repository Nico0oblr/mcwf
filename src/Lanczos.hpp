#ifndef LANCZOS_HPP
#define LANCZOS_HPP

#include "Common.hpp"

template<typename MatrixType>
std::pair<Eigen::Matrix<typename MatrixType::Scalar,
			Eigen::Dynamic, Eigen::Dynamic>,
	  Eigen::Matrix<typename MatrixType::Scalar,
			Eigen::Dynamic, Eigen::Dynamic>>
lanczos_iteration(const MatrixType & A, int niter, const vec_t & vec) {
  using Scalar = typename MatrixType::Scalar;
  using DenseType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  // assert((A - MatrixType(A.adjoint())).norm() < 1e-12);
  
  int n = A.rows();
  DenseType V(n, niter);
  vec_t alpha(niter);
  vec_t beta(niter - 1);

  if (vec.size() != 0) {
    V.col(0) = vec;
  } else {
    V.col(0).setRandom();
  }

  V.col(0) = V.col(0) / V.col(0).norm();
  vec_t v = A * V.col(0);
  alpha[0] = v.dot(V.col(0));
  v -= alpha[0] * V.col(0);
  int i = 1;
  
  for (; i < niter; ++i) {
    beta[i - 1] = v.norm();
    
    if (std::abs(beta[i - 1]) < 1e-12) {
      LOG(logERROR) << "Krylov space closed at index "
		    << i << std::endl;
      break;
    }

    V.col(i) = v / beta[i - 1];
    v = A * V.col(i);
    alpha[i] = v.dot(V.col(i));
    v -= alpha[i] * V.col(i) + beta[i - 1] * V.col(i - 1);
  }

  DenseType H = DenseType::Zero(i, i);
  H.diagonal() = alpha.head(i);
  H.template diagonal<1>() = beta.head(i - 1);
  H.template diagonal<-1>() = beta.head(i - 1);
  return std::make_pair(H, V.topLeftCorner(n, i));
}

template<typename MatrixType>
std::pair<Eigen::Matrix<double, Eigen::Dynamic, 1>,
	  Eigen::Matrix<typename MatrixType::Scalar,
			Eigen::Dynamic, Eigen::Dynamic> >
diagonalize_iteration(const MatrixType & H, const MatrixType & V) {
  Eigen::ComplexEigenSolver<MatrixType> solver(H);
  MatrixType eigenvectors = V * solver.eigenvectors();
  return std::make_pair(solver.eigenvalues().real(), eigenvectors);
}

template<typename MatrixType>
auto find_groundstate(const MatrixType & A, int niter) {

			  // if (A.rows() > 200) {
    auto [H, V] = lanczos_iteration(A, niter, vec_t());
    auto [eival, eivec] = diagonalize_iteration(H.eval(), V.eval());
    auto me = std::min_element(eival.data(), eival.data() + eival.size());
    int min_index = me - eival.data();
    double minval = eival(min_index);
    vec_t minvec = eivec.col(min_index);
    return std::make_pair(minval, minvec);
/*} else {
    using Scalar = typename MatrixType::Scalar;
    using DenseType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  
    Eigen::ComplexEigenSolver<DenseType> solver(A);
    auto eival = solver.eigenvalues().real().eval();
    auto eivec = solver.eigenvectors().eval();
    auto me = std::min_element(eival.data(), eival.data() + eival.size());
    int min_index = me - eival.data();
    double minval = eival(min_index);
    vec_t minvec = eivec.col(min_index);
    return std::make_pair(minval, minvec);
    }*/
}

#endif /* LANCZOS_HPP */
