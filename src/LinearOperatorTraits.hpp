#ifndef LINEAROPERATORTRAITS_HPP
#define LINEAROPERATORTRAITS_HPP

template<typename MatrixType>
struct LinearOperator;

namespace Eigen {
  namespace internal {
    template<typename MatrixType>
    struct traits<LinearOperator<MatrixType>> :
      public Eigen::internal::traits<MatrixType>
    {};
  }

  template<typename MatrixType>
  struct NumTraits<LinearOperator<MatrixType>>
 {
   using type = LinearOperator<MatrixType>;
   using Scalar = typename LinearOperator<MatrixType>::Scalar;
   using RealScalar = typename NumTraits<Scalar>::Real;
  
   enum {
     IsComplex = NumTraits<Scalar>::IsComplex,
     IsInteger = NumTraits<Scalar>::IsInteger,
     IsSigned  = NumTraits<Scalar>::IsSigned,
     RequireInitialization = 1,
     ReadCost = HugeCost,
     AddCost  = HugeCost,
     MulCost  = HugeCost,
   };
  
   EIGEN_DEVICE_FUNC
   static inline RealScalar epsilon()
   { return NumTraits<RealScalar>::epsilon(); }
   EIGEN_DEVICE_FUNC
   static inline RealScalar dummy_precision()
   { return NumTraits<RealScalar>::dummy_precision(); }
  
   static inline int digits10() { return NumTraits<Scalar>::digits10(); }
 };
}

#endif /* LINEAROPERATORTRAITS_HPP */
