#ifndef LINEAROPERATOREIGENPROD_HPP
#define LINEAROPERATOREIGENPROD_HPP

namespace Eigen {
namespace internal {

  template<typename MatrixType, typename Rhs>
  struct generic_product_impl<LinearOperator<MatrixType>, Rhs, SparseShape, DenseShape, GemvProduct>
  // GEMV stands for matrix-vector
  : generic_product_impl_base<LinearOperator<MatrixType>,Rhs, generic_product_impl<LinearOperator<MatrixType>, Rhs> > {
    typedef typename Product<LinearOperator<MatrixType>, Rhs>::Scalar Scalar;
 
    template<typename Dest>
    static void scaleAndAddTo(Dest& dst, const LinearOperator<MatrixType> & lhs,
			      const Rhs& rhs, const Scalar& alpha) {
      assert(alpha==Scalar(1) && "scaling is not implemented");
      EIGEN_ONLY_USED_FOR_DEBUG(alpha);
      dst.noalias() += lhs.apply_to(rhs);
    }
  };
}
}

namespace Eigen {
  namespace internal {
    template<typename Lhs, typename MatrixType>
    struct generic_product_impl<Lhs, LinearOperator<MatrixType>,
				DenseShape, SparseShape, GemvProduct>
      : generic_product_impl_base<Lhs, LinearOperator<MatrixType>, generic_product_impl<Lhs, LinearOperator<MatrixType>, DenseShape, SparseShape, GemvProduct> > {
      typedef typename Product<Lhs, LinearOperator<MatrixType>>::Scalar Scalar;
 
      template<typename Dest>
      static void scaleAndAddTo(Dest& dst, const Lhs & lhs,
				const LinearOperator<MatrixType> & rhs,
				const Scalar& alpha) {
	assert(alpha==Scalar(1) && "scaling is not implemented");
	EIGEN_ONLY_USED_FOR_DEBUG(alpha);
	dst.noalias() += rhs.applied_to(lhs);
      }
    };
  }
}

namespace Eigen {
namespace internal {

  template<typename MatrixType, typename Rhs>
  struct generic_product_impl<LinearOperator<MatrixType>, Rhs, SparseShape,
			      DenseShape, GemmProduct>
  // GEMM stands for matrix-matrix
  : generic_product_impl_base<LinearOperator<MatrixType>,Rhs,
			      generic_product_impl<LinearOperator<MatrixType>,
						   Rhs> > {
    typedef typename Product<LinearOperator<MatrixType>,Rhs>::Scalar Scalar;

    template<typename Dest>
    static void scaleAndAddTo(Dest& dst, const LinearOperator<MatrixType> & lhs,
			      const Rhs& rhs, const Scalar& alpha) {
      assert(alpha==Scalar(1) && "scaling is not implemented");
      EIGEN_ONLY_USED_FOR_DEBUG(alpha);

      for (int i = 0; i < rhs.cols(); ++i) {
	dst.col(i).noalias() += lhs.apply_to(rhs.col(i));
      }
    }
  };
}}

namespace Eigen {
  namespace internal {
    template<typename Lhs, typename MatrixType>
    struct generic_product_impl<Lhs, LinearOperator<MatrixType>, DenseShape, SparseShape, GemmProduct>
      : generic_product_impl_base<Lhs, LinearOperator<MatrixType>, generic_product_impl<Lhs, LinearOperator<MatrixType>, DenseShape, SparseShape, GemmProduct> > {
      typedef typename Product<Lhs, LinearOperator<MatrixType>>::Scalar Scalar;
      
      template<typename Dest>
      static void scaleAndAddTo(Dest& dst, const Lhs & lhs,
				const LinearOperator<MatrixType> & rhs,
				const Scalar& alpha) {
	assert(alpha==Scalar(1) && "scaling is not implemented");
	EIGEN_ONLY_USED_FOR_DEBUG(alpha);
	for (int i = 0; i < lhs.cols(); ++i) {
	  dst.row(i).noalias() += rhs.applied_to(lhs.row(i));
	}
      }
    };   
  }
}


#endif /* LINEAROPERATOREIGENPROD_HPP */
