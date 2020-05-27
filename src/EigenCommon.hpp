#ifndef EIGENCOMMON_HPP
#define EIGENCOMMON_HPP

#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_SPARSEMATRIX_PLUGIN "SparseAddons.h"
#define EIGEN_SPARSEMATRIXBASE_PLUGIN "SparseBaseAddons.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "KroneckerProduct.hpp"
#pragma GCC diagnostic pop
#include <iostream>

const double tol = 1e-10;

/*
  n!
*/
int factorial(int n);

using scalar_t = std::complex<double>;
using vec_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
using mat_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
using diag_mat_t = Eigen::DiagonalMatrix<scalar_t, Eigen::Dynamic,
					 Eigen::Dynamic>;
using spmat_t = Eigen::SparseMatrix<scalar_t>;
using calc_mat_t = mat_t;


/*
  Prints the matrix cols / rows to stdout
*/
template<typename Derived>
void print_matrix_dim(const Eigen::EigenBase<Derived> & mat) {
  std::cout << mat.rows() << "/" << mat.cols() << std::endl;
}

/*
  Calculate op\tensor\identity
  with dimension sub_dim of identity
 */
template<typename Derived>
auto tensor_identity(const Eigen::EigenBase<Derived> & op,
		     int sub_dim) {
  using SparseType = Eigen::SparseMatrix<typename Derived::Scalar>;
  SparseType id = SparseType::Identity(sub_dim, sub_dim);
  return Eigen::kroneckerProduct(op.derived(), id).eval();
}

/*
  Calculate \identity\tensor op
  with dimension sub_dim of identity
*/
template<typename Derived>
auto tensor_identity_LHS(const Eigen::EigenBase<Derived> & op,
			 int sub_dim) {
  using SparseType = Eigen::SparseMatrix<typename Derived::Scalar>;
  SparseType id = SparseType::Identity(sub_dim, sub_dim);
  return Eigen::kroneckerProduct(id, op.derived()).eval();
}

/*
  Double the matrix dimension with diagonal blocks
  being mat.
*/
template<typename Derived>
auto double_matrix(const Eigen::EigenBase<Derived> & mat) {
  return Eigen::kroneckerProduct(Derived::PlainObject::Identity(2, 2),
				 mat.derived());
}

/*
  Double matrix dimension of vector of matrices
*/
template<typename matrix_type>
std::vector<matrix_type> double_matrix(const std::vector<matrix_type> & mats) {
  std::vector<matrix_type> out;
  for (const matrix_type & mat : mats) {
    out.push_back(double_matrix(mat));
  }
  return out;
}

template<typename Derived>
auto matrix_exponential_taylor(const Eigen::EigenBase<Derived> & matrix) {
  return Derived::PlainObject::Identity(matrix.cols(), matrix.rows())
    + matrix.derived() + 0.5 * matrix.derived() * matrix.derived();
}

template<typename Derived>
auto matrix_exponential_taylor(const Eigen::EigenBase<Derived> & matrix,
			       int order) {
  typename Derived::PlainObject mat_n = Derived::PlainObject::Identity(matrix.rows(), matrix.cols());
  typename Derived::PlainObject result = Derived::PlainObject::Zero(matrix.rows(), matrix.cols());

  for (int i = 0; i <= order; ++i) {
    result += mat_n / factorial(i);
    mat_n = mat_n * matrix.derived();
  }
  return result;
}

template<typename Derived>
vec_t apply_matrix_exponential_taylor(const Eigen::EigenBase<Derived> & matrix,
				      const vec_t & vec,
				      int order) {
  vec_t result = vec;
  vec_t vec_c = vec;
  
  for (int i = 1; i <= order; ++i) {
    vec_c = matrix.derived() * vec_c;
    result += vec_c / factorial(i);
  }
  return result;
}

template<typename Derived>
vec_t apply_matrix_exponential_taylorV2(const Eigen::EigenBase<Derived> & matrix,
					const vec_t & vec,
					int order) {
  typename Derived::PlainObject exp_n = matrix.derived() / static_cast<double>(order);
  vec_t vec_c = vec;
  
  for (int i = 0; i < order; ++i) {
    vec_c = vec_c + exp_n * vec_c;
  }
  return vec_c;
}

template<typename Derived>
auto superoperator_left(const Eigen::EigenBase<Derived> & op, int dimension) {
  return tensor_identity(op.derived(), dimension);
}

template<typename Derived>
auto superoperator_right(const Eigen::EigenBase<Derived> & op, int dimension) {
  return tensor_identity_LHS(op.derived().transpose(), dimension);
}

template<typename Derived>
double expval(const Eigen::EigenBase<Derived> & observable,
	      const vec_t & state) {
  return state.dot(observable.derived() * state).real();
}

template<typename Derived>
vec_t unstack_matrix(const Eigen::EigenBase<Derived> & mat) {
  vec_t out(mat.rows() * mat.cols());

  for (int i = 0; i < mat.rows(); ++i) {
    auto vec_seq = Eigen::seq(mat.cols() * i, mat.cols() * (i + 1) - 1);
    out(vec_seq) = mat.derived().row(i);
  }
  return out;
}

/*
  Returns the superoperator vector due to a density matrix mat.
  Alternative implementation, that explicitly uses tensor products.
*/
template<typename Derived>
vec_t unstack_matrix_alt(const Eigen::EigenBase<Derived> & mat) {
  vec_t out = vec_t::Zero(mat.rows() * mat.cols());
  for (int i = 0; i < mat.rows(); ++i) {
    vec_t unit_i = vec_t::Zero(mat.rows());
    unit_i(i) = 1.0;
    
    for (int j = 0; j < mat.cols(); ++j) {
      vec_t unit_j = vec_t::Zero(mat.rows());
      unit_j(j) = 1.0;
      out += mat.derived().coeff(i, j) * Eigen::kroneckerProduct(unit_i, unit_j);
    }
  }
  return out;
}

/*
  Prints the percentage of matrix entries below a certain tolerance.
 */
template<typename Derived>
double sparsity(const Eigen::MatrixBase<Derived> & mat) {
  double zeros = (mat.derived().array().abs() < tol).count();
  double entries = mat.size();
  return zeros / entries;
}

template<typename Derived>
double sparsity(const Eigen::SparseMatrixBase<Derived> & mat) {
  return mat.nonZeros() / static_cast<double>(mat.size());
}

/*
  Performs matrix exponential via diagonalization of matrix
*/
inline mat_t matrix_exponential(const mat_t & mat) {
  Eigen::ComplexEigenSolver<mat_t> solver(mat, true);
  mat_t V = solver.eigenvectors();
  mat_t expD = solver.eigenvalues().array().exp().matrix().asDiagonal();
  Eigen::PartialPivLU<mat_t> inverter(V);
  mat_t V_inv = inverter.inverse();
  return V * expD * V_inv;
}

template<typename Derived>
auto matrix_exponential_sparse(const Eigen::SparseMatrixBase<Derived> & matrix) {
  return matrix_exponential_taylor(matrix, 4);
}


template<typename A, typename B>
auto commutator(const Eigen::EigenBase<A> & lhs,
		const Eigen::EigenBase<B> & rhs) {
  return lhs.derived() * rhs.derived() - rhs.derived() * lhs.derived();
}


#endif /* EIGENCOMMON_HPP */
