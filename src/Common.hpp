#ifndef COMMON_HPP
#define COMMON_HPP

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
mat_t tensor_identity(const mat_t & op, int sub_dim);

/*
  Calculate \identity\tensor op
  with dimension sub_dim of identity
*/
mat_t tensor_identity_LHS(const mat_t & op, int sub_dim);

/*
  Double the matrix dimension with diagonal blocks
  being mat.
*/
mat_t double_matrix(const mat_t & mat);

/*
  Double matrix dimension of vector of matrices
*/
std::vector<mat_t> double_matrix(const std::vector<mat_t> & mats);

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

#endif /* COMMON_HPP */
