#include "Recorders.hpp"

double evaluate_impl(const vec_t & state,
		     const LinearOperator<calc_mat_t> & observable) {
  return state.dot(observable * state).real();
}

double evaluate_impl(const calc_mat_t & density_matrix,
		     const LinearOperator<calc_mat_t> & observable) {
  return calc_mat_t(observable.eval() * density_matrix).trace().real();
}

calc_mat_t density_impl(const calc_mat_t & density_matrix)
{return density_matrix;}
calc_mat_t density_impl(const vec_t & state)
{return state * state.adjoint();}
