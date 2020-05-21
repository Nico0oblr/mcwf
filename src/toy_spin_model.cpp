#include "toy_spin_model.hpp"

#include "Operators.hpp"

mat_t J0_n(int dimension,
	   double hubbardU,
	   double hopping,
	   double frequency,
	   double coupling) {
  return J0(dimension, hubbardU, hopping, frequency, coupling).diagonal().asDiagonal();
}

mat_t toy_modelize(int dimension, const mat_t & electronic_ham,
		   double hubbardU,
		   double hopping,
		   double frequency,
		   double coupling) {
  return Eigen::kroneckerProduct(J0_n(dimension, hubbardU,
				      hopping, frequency, coupling),
				 electronic_ham);
}
