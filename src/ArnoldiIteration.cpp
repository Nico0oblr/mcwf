#include "ArnoldiIteration.hpp"

vec_t exp_krylov(const spmat_t & A, const vec_t & vec, int nruns) {
  ArnoldiIteration<spmat_t> iteration(A, nruns, nruns, vec);
  return iteration.apply_exp(vec, nruns);
}

vec_t exp_krylov_alt(const spmat_t & A, const vec_t & vec,
		     int nruns) {
  ArnoldiIteration<spmat_t> iteration(A, nruns, nruns, vec);
  return iteration.eigenvectors()
    * (iteration.eigenvalues().array().exp().matrix().asDiagonal()
       * (iteration.eigenvectors().adjoint() * vec));
}

