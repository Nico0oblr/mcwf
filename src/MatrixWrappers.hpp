#ifndef MATRIXWRAPPERS_HPP
#define MATRIXWRAPPERS_HPP

#include "Common.hpp"

/*Less complicated direct wrappers without inheritance*/

template<typename _MatrixType>
struct PowerWrapper {
  using MatrixType = typename _MatrixType::PlainObject;

  enum {
	RowsAtCompileTime = MatrixType::RowsAtCompileTime,
	ColsAtCompileTime = MatrixType::ColsAtCompileTime,
	Options = MatrixType::Options,
	MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
	MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
  };
  
  typedef typename MatrixType::Scalar Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef Eigen::Index Index;
  typedef std::complex<RealScalar> ComplexScalar;  
  
  template<typename RHS_t>
  auto apply_to(const RHS_t & RHS) const {
    LOG_VAR(power);
    using ResultType = typename decltype(value * RHS)::PlainObject;
    if (power == 0) return ResultType(RHS);
    ResultType tmp = value * RHS;
    for (int i = 1; i < power; ++i) tmp = value * tmp;
    return tmp;
  }

  PowerWrapper<MatrixType> adjoint() const {
    return PowerWrapper<MatrixType>{value.adjoint(), power};
  }

  Eigen::Index rows() const {
    return value.rows();
  }

  Eigen::Index cols() const {
    return value.cols();
  }
  
  _MatrixType value;
  int power;
};

template<typename _MatrixType>
struct KroneckerWrapper {
  using MatrixType = _MatrixType;

  enum {
	RowsAtCompileTime = MatrixType::RowsAtCompileTime,
	ColsAtCompileTime = MatrixType::ColsAtCompileTime,
	Options = MatrixType::Options,
	MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
	MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
  };
  
  typedef typename MatrixType::Scalar Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef Eigen::Index Index;
  typedef std::complex<RealScalar> ComplexScalar;

  Eigen::Index rows() const {
    return LHS.rows() * RHS.rows();
  }

  Eigen::Index cols() const {
    return LHS.cols() * RHS.cols();
  }

  template<typename vec_type>
  auto apply_to(const vec_type & vec) const {
    return kroneckerApplyLazy(LHS, RHS, vec);
  }

  KroneckerWrapper(const MatrixType & A, const MatrixType & B)
    :LHS(A), RHS(B) {}
  
  MatrixType LHS, RHS;
};

template<typename _MatrixType>
struct AdditionWrapper {
  using MatrixType = _MatrixType;

  enum {
	RowsAtCompileTime = MatrixType::RowsAtCompileTime,
	ColsAtCompileTime = MatrixType::ColsAtCompileTime,
	Options = MatrixType::Options,
	MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
	MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
  };
  
  typedef typename MatrixType::Scalar Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef Eigen::Index Index;
  typedef std::complex<RealScalar> ComplexScalar;

  Eigen::Index rows() const {
    return LHS.rows();
  }

  Eigen::Index cols() const {
    return RHS.cols();
  }

  template<typename vec_type>
  auto apply_to(const vec_type & vec) const {
    return LHS * vec + RHS * vec;
  }

  AdditionWrapper(const MatrixType & A, const MatrixType & B)
    :LHS(A), RHS(B) {}
  
  MatrixType LHS, RHS;
};

template<typename MatrixType, typename T>
auto operator*(const PowerWrapper<MatrixType> & lhs, const T & rhs) {
  return lhs.apply_to(rhs);
}

template<typename MatrixType, typename T>
auto operator*(const AdditionWrapper<MatrixType> & lhs, const T & rhs) {
  return lhs.apply_to(rhs);
}

template<typename MatrixType, typename T>
auto operator*(const KroneckerWrapper<MatrixType> & lhs, const T & rhs) {
  return lhs.apply_to(rhs);
}

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

#endif /* MATRIXWRAPPERS_HPP */
