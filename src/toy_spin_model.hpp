#ifndef TOY_SPIN_MODEL_HPP
#define TOY_SPIN_MODEL_HPP

#include "Common.hpp"

mat_t J0_n(int dimension,
	   double hubbardU,
	   double hopping,
	   double frequency,
	   double coupling);

mat_t toy_modelize(int dimension, const mat_t & electronic_ham,
		   double hubbardU,
		   double hopping,
		   double frequency,
		   double coupling);

#endif /* TOY_SPIN_MODEL_HPP */
