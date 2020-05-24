#ifndef COMMON_HPP
#define COMMON_HPP

#define EIGEN_DONT_PARALLELIZE
#define EIGEN_SPARSEMATRIX_PLUGIN "SparseAddons.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <random>
#include <iostream>

/*Some type and constants, that will be used throughout the program*/
using namespace std::complex_literals;
using scalar_t = std::complex<double>;
using vec_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
using mat_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
using spmat_t = Eigen::SparseMatrix<scalar_t>;
using calc_mat_t = mat_t;
/*Mersenne twister random engine*/
static std::mt19937 mt_rand(110794);
static std::uniform_real_distribution<double> dis(0.0, 1.0);
/*Numerical tolerance for double comp*/
static double tol = 1e-10;

spmat_t cI(Eigen::Index n, std::complex<double> value);

/*
  Calculate op\tensor\identity
  with dimension sub_dim of identity
 */
template<typename matrix_type>
matrix_type tensor_identity(const matrix_type & op, int sub_dim) {
  matrix_type id = matrix_type::Identity(sub_dim, sub_dim);
  return Eigen::kroneckerProduct(op, id);
}

/*
  Calculate \identity\tensor op
  with dimension sub_dim of identity
*/
template<typename matrix_type>
matrix_type tensor_identity_LHS(const matrix_type & op, int sub_dim) {
  matrix_type id = matrix_type::Identity(sub_dim, sub_dim);
  return Eigen::kroneckerProduct(id, op);
}

/*
  Double the matrix dimension with diagonal blocks
  being mat.
*/
template<typename matrix_type>
matrix_type double_matrix(const matrix_type & mat) {
  return Eigen::kroneckerProduct(matrix_type::Identity(2, 2), mat);
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

/*
  Add two vectors vec1 and vec2 by stacking them on top of each other
*/
vec_t add_vectors(const vec_t & vec1, const vec_t & vec2);

/*
  Perform a linear interval search in order to sample 
  discrete probability distribution. Returns drawn index.
  If distribution does not add up to 1, may exit program.
  TODO: throw proper exception
*/
template<typename vector_type>
int linear_search(const vector_type & probabilities) {
  std::uniform_real_distribution<double> dis(0.0, 1.0);
  double eta = dis(mt_rand);
  double cumulative = 0.0;
  for (int i = 0; i < probabilities.size(); ++i) {
    cumulative += probabilities[i];
    if (eta <= cumulative) {
      return i;
    }
  }
  assert(false);
}

/*
  Performs matrix exponential via diagonalization of matrix
*/
mat_t matrix_exponential(const mat_t & matrix);

/*
  Obtains the matrix exponential via the taylor expansion up to second order
*/
spmat_t matrix_exponential_taylor(const spmat_t & matrix);

/*
  Same thing for dense matrices
*/
mat_t matrix_exponential_taylor(const mat_t & matrix);

/*
  Prints the matrix cols / rows to stdout
*/
template<typename matrix_type>
void print_matrix_dim(const matrix_type & mat) {
  std::cout << mat.rows() << "/" << mat.cols() << std::endl;
}

/*
  Calculates superoperator due to application from left
*/
mat_t superoperator_left(const mat_t & op, int dimension);

/*
  Calculates superoperator due to application from right
*/
mat_t superoperator_right(const mat_t & op, int dimension);

/*
  Returns the superoperator vector due to a density matrix mat.
*/
vec_t unstack_matrix(const mat_t & mat);

/*
  Returns the superoperator vector due to a density matrix mat.
  Alternative implementation, that explicitly uses tensor products.
*/
vec_t unstack_matrix_alt(const mat_t & mat);

/*
  Reconstructs a density matrix from a vector in superoperator notation.
*/
mat_t restack_vector(const vec_t & vec, int dimension);

/*
  Prints the percentage of matrix entries below a certain tolerance.
 */
double dmat_sparsity(const mat_t & mat);

/*
  n!
*/
int factorial(int n);

/*
  n over k
*/
int binomial(int n, int k);

/*
  (-1)^n for integers n.
*/
int minus_one_power(int n);

/*
  Defines poission with mean N at n.
*/
double poisson(double N, int n);

double expval(const mat_t & observable, const vec_t & state);

template<typename matrix_type>
matrix_type matrix_exponential_taylor(const matrix_type & matrix, int order) {
  matrix_type mat_n = cI(matrix.rows(), 1.0);
  matrix_type result = cI(matrix.rows(), 0.0);

  for (int i = 0; i <= order; ++i) {
    result += mat_n / factorial(i);
    mat_n = mat_n * matrix;
  }
  return result;
}

template<typename matrix_type>
vec_t apply_matrix_exponential_taylor(const matrix_type & matrix,
				      const vec_t & vec,
				      int order) {
  vec_t result = vec;
  vec_t vec_c = vec;
  
  for (int i = 1; i <= order; ++i) {
    vec_c = matrix * vec_c;
    result += vec_c / factorial(i);
  }
  return result;
}

#endif /* COMMON_HPP */
