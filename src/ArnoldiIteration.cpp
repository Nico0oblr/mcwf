#include "ArnoldiIteration.hpp"

vec_t exp_krylov(const spmat_t & A, const vec_t & vec, int nruns) {
  LOG(logINFO) << "running exp_krylov" << std::endl;
  ArnoldiIteration<spmat_t> iteration(A, nruns, nruns, vec);
  return iteration.apply_exp(vec, iteration.nit());
}

vec_t exp_krylov_alt(const spmat_t & A, const vec_t & vec,
		     int nruns) {
  std::complex<double> trace_mean = A.trace() / static_cast<double>(A.rows());
  spmat_t id = spmat_t::Identity(A.rows(), A.cols());
  LOG(logINFO) << "running exp_krylov_alt" << std::endl;
  ArnoldiIteration<spmat_t> iteration((A - trace_mean * id).eval(),
				      nruns, nruns, vec);
  return iteration.eigenvectors()
    * (iteration.eigenvalues().array().exp().matrix().asDiagonal()
       * (iteration.eigenvectors().adjoint() * vec))
    + std::exp(trace_mean) * vec;
}

