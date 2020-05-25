#ifndef COMMON_HPP
#define COMMON_HPP

#include <random>
#include <iostream>
#include "EigenCommon.hpp"

/*Some type and constants, that will be used throughout the program*/
using namespace std::complex_literals;
/*Mersenne twister random engine*/
static std::mt19937 mt_rand(110794);
static std::uniform_real_distribution<double> dis(0.0, 1.0);
/*Numerical tolerance for double comp*/
using size_type = std::size_t;

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
  Reconstructs a density matrix from a vector in superoperator notation.
*/
mat_t restack_vector(const vec_t & vec, int dimension);

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
