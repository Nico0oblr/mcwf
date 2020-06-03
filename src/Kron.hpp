#ifndef KRON_HPP
#define KRON_HPP

#include "Common.hpp"

using spvec = Eigen::SparseVector<scalar_t>;

vec_t kroneckerApply(const spmat_t & A,
		     const spmat_t & B,
		     const vec_t & vec);

vec_t kroneckerApplyLazy(const spmat_t & A,
			 const spmat_t & B,
			 const vec_t & vec);

vec_t kroneckerApply_id(const spmat_t & A,
			int codimension,
			const vec_t & vec);

vec_t kroneckerApply_LHS(const spmat_t & B,
			 int codimension,
			 const vec_t & vec);

vec_t kroneckerApply_sp(const spmat_t & A,
			const spmat_t & B,
			const spvec & vec);

#endif /* KRON_HPP */
