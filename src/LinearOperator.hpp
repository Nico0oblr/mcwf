#ifndef LINEAROPERATOR_HPP
#define LINEAROPERATOR_HPP

#include "MatrixWrappers.hpp"
#include "HubbardModel.hpp"
#include "Operators.hpp"
#include <memory>

template<typename MatrixType>
struct LinearOperator;

namespace Eigen {
  namespace internal {
    template<typename MatrixType>
    struct traits<LinearOperator<MatrixType>> :
      public Eigen::internal::traits<MatrixType>
    {};
  }
}

template<typename _MatrixType>
struct LinearOperator
// : public Eigen::EigenBase<LinearOperator<_MatrixType>>
{
  using MatrixType = _MatrixType;
  using this_t = LinearOperator<MatrixType>;
  
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
  using StorageIndex = int;
  typedef std::complex<RealScalar> ComplexScalar;
  const static bool IsRowMajor = false;

  using VectorType =  Eigen::Matrix<Scalar, ColsAtCompileTime, 1>;
  using AdjointVectorType = Eigen::Matrix<Scalar, 1, RowsAtCompileTime>;

  virtual VectorType apply_to(const VectorType &) const {assert(false);};
  virtual AdjointVectorType applied_to(const AdjointVectorType &) const {assert(false);}
  virtual MatrixType eval() const {assert(false);}
  virtual Eigen::Index rows() const {assert(false);};
  virtual Eigen::Index cols() const {assert(false);};
  auto clone() const { return std::unique_ptr<this_t>(clone_impl()); }
  
  /*template<typename Rhs>
  VectorType operator*(const Eigen::MatrixBase<Rhs>& x) const {
    return this->apply_to(x.derived());
    }*/

  template<typename Rhs>
  Eigen::Product<this_t, Rhs, Eigen::AliasFreeProduct>
  operator*(const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<this_t, Rhs, Eigen::AliasFreeProduct>
      (*this, x.derived());
  }

  template<typename OtherDerived> friend
  Eigen::Product<OtherDerived, this_t>
  operator*(const Eigen::MatrixBase<OtherDerived>& lhs,
	    const this_t & rhs) {
    return Eigen::Product<OtherDerived, this_t>(lhs.derived(), rhs);
  }
     
  this_t & operator*=(Scalar s) {
    mult_by_scalar(s);
    return *this;
  }
  
  virtual void mult_by_scalar(Scalar s) {assert(false);}
protected:
  
  virtual this_t* clone_impl() const {assert(false);};
};


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
  struct generic_product_impl<Lhs, LinearOperator<MatrixType>, DenseShape, SparseShape, GemvProduct>
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
	assert(false);
      }
    };   
  }
}

template<typename _MatrixType>
struct BareLinearOperator : public LinearOperator<_MatrixType> {
  using MatrixType = _MatrixType;
  using Base = LinearOperator<MatrixType>;
  using VectorType = typename Base::VectorType;
  using AdjointVectorType = typename Base::AdjointVectorType;

  virtual VectorType apply_to(const VectorType & vector) const override {
    return self * vector;
  }

  virtual AdjointVectorType
  applied_to(const AdjointVectorType & vector) const override {
    return vector * self;
  }
  
  Eigen::Index rows() const override {
    return self.rows();
  }
  
  Eigen::Index cols() const override {
    return self.cols();
  }

  BareLinearOperator(const MatrixType & A)
    : self(A) {}

  using Scalar = typename Base::Scalar;
  void mult_by_scalar(Scalar s) override {self *= s;}

  MatrixType eval() const {return self;}
  
  using this_t = BareLinearOperator<MatrixType>;
  this_t* clone_impl() const override {return new this_t(*this);};
  
  MatrixType self;
};

template<typename _MatrixType>
struct SumLinearOperator : public LinearOperator<_MatrixType> {
  using MatrixType = _MatrixType;
  using Base = LinearOperator<_MatrixType>;
  using BasePointer = std::unique_ptr<Base>;
  using VectorType = typename Base::VectorType;

  VectorType apply_to(const VectorType & vector) const override {
    return LHS->apply_to(vector) + RHS->apply_to(vector);
  }
  
  Eigen::Index rows() const override {
    return LHS->rows();
  }
  
  Eigen::Index cols() const override {
    return LHS->cols();
  }

  SumLinearOperator(const Base & A, const Base & B)
    :LHS(A.clone()), RHS(B.clone()) {
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
  }

  SumLinearOperator(const SumLinearOperator<MatrixType> & other)
    :LHS(other.LHS->clone()), RHS(other.RHS->clone()) {}

  MatrixType eval() const {return LHS->eval() + RHS->eval();}

  using Scalar = typename Base::Scalar;
  void mult_by_scalar(Scalar s) override {
    *LHS *= s;
    *RHS *= s;
  }

  using this_t = SumLinearOperator<MatrixType>;
  this_t* clone_impl() const override {return new this_t(*this);};
  
  BasePointer LHS, RHS;
};

/*
  Requires explicit iteration over entries, until abstracted
  this requires actual matrices as entries 
*/
template<typename _MatrixType>
struct KroneckerLinearOperator : public LinearOperator<_MatrixType>{
  using MatrixType = _MatrixType;
  using Base = LinearOperator<MatrixType>;
  using VectorType = typename Base::VectorType;

  Eigen::Index rows() const override {
    return LHS.rows() * RHS.rows();
  }

  Eigen::Index cols() const override {
    return LHS.cols() * RHS.cols();
  }

  VectorType apply_to(const VectorType & vector) const override {
    return kroneckerApply(LHS, RHS, vector);
  }

  KroneckerLinearOperator(const MatrixType & A, const MatrixType & B)
    :LHS(A), RHS(B) {}

  using Scalar = typename Base::Scalar;
  void mult_by_scalar(Scalar s) override {LHS *= s;}

  MatrixType eval() const {
    return Eigen::kroneckerProduct(LHS, RHS);
  }
  
  using this_t = KroneckerLinearOperator<MatrixType>;
  this_t* clone_impl() const override {return new this_t(*this);};
  
  MatrixType LHS, RHS;
};

template<typename _MatrixType>
struct KroneckerIDRHSLinearOperator
  : public LinearOperator<_MatrixType>{
  using MatrixType = _MatrixType;
  using Base = LinearOperator<MatrixType>;
  using VectorType = typename Base::VectorType;

  Eigen::Index rows() const override {
    return LHS.rows() * codimension;
  }

  Eigen::Index cols() const override {
    return LHS.cols() * codimension;
  }

  VectorType apply_to(const VectorType & vector) const override {
    return kroneckerApply_id(LHS, codimension, vector);
  }

  KroneckerIDRHSLinearOperator(const MatrixType & A, int dim)
    :LHS(A), codimension(dim) {}

  using Scalar = typename Base::Scalar;
  void mult_by_scalar(Scalar s) override {LHS *= s;}

  MatrixType eval() const {
    return tensor_identity(LHS, codimension);
  }

  using this_t = KroneckerIDRHSLinearOperator<MatrixType>;
  this_t* clone_impl() const override {return new this_t(*this);};
  
  MatrixType LHS;
  int codimension;
};

template<typename _MatrixType>
struct KroneckerIDLHSLinearOperator
  : public LinearOperator<_MatrixType>{
  using MatrixType = _MatrixType;
  using Base = LinearOperator<MatrixType>;
  using VectorType = typename Base::VectorType;

  Eigen::Index rows() const override {
    return RHS.rows() * codimension;
  }

  Eigen::Index cols() const override {
    return RHS.cols() * codimension;
  }

  VectorType apply_to(const VectorType & vector) const override {
    return kroneckerApply_LHS(RHS, codimension, vector);
  }

  KroneckerIDLHSLinearOperator(const MatrixType & A, int dim)
    :RHS(A), codimension(dim) {}

  using Scalar = typename Base::Scalar;
  void mult_by_scalar(Scalar s) override {RHS *= s;}

  MatrixType eval() const {
    return tensor_identity_LHS(RHS, codimension);
  }

  using this_t = KroneckerIDLHSLinearOperator<MatrixType>;
  this_t* clone_impl() const override {return new this_t(*this);};
  
  MatrixType RHS;
  int codimension;
};

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
kroneckerOperator(const MatrixType & A, const MatrixType & B) {
  return KroneckerLinearOperator<MatrixType>(A, B).clone();
}


template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
kroneckerOperator_IDRHS(const MatrixType & A, int dim) {
  return KroneckerIDRHSLinearOperator<MatrixType>(A, dim).clone();
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
kroneckerOperator_IDLHS(const MatrixType & A, int dim) {
  return KroneckerIDLHSLinearOperator<MatrixType>(A, dim).clone();
}

/*Operator+*/
template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
sumOperator(const LinearOperator<MatrixType> & A,
	    const LinearOperator<MatrixType> & B) {
  return SumLinearOperator<MatrixType>(A, B).clone();
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
operator+(const LinearOperator<MatrixType> & A,
	  const LinearOperator<MatrixType> & B) {
  return sumOperator(A, B);
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
operator+(const std::unique_ptr<LinearOperator<MatrixType>> & A,
	  const LinearOperator<MatrixType> & B) {
  return sumOperator(*A, B);
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
operator+(const LinearOperator<MatrixType> & A,
	  const std::unique_ptr<LinearOperator<MatrixType>> & B) {
  return sumOperator(A, *B);
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
operator+(const std::unique_ptr<LinearOperator<MatrixType>> & A,
	  const std::unique_ptr<LinearOperator<MatrixType>> & B) {
  return sumOperator(*A, *B);
}

/* Operator* */

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
operator*(const std::unique_ptr<LinearOperator<MatrixType>> & A,
	  typename MatrixType::Scalar sc) {
  auto tmp = A->copy();
  tmp->mult_by_scalar(sc);
  return tmp;
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
operator*(const LinearOperator<MatrixType> & A,
	  typename MatrixType::Scalar sc) {
  auto tmp = A.copy();
  tmp->mult_by_scalar(sc);
  return tmp;
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
operatorize(const MatrixType & mat) {
  return BareLinearOperator<MatrixType>(mat).clone();
}

std::unique_ptr<LinearOperator<spmat_t>>
Hubbard_light_matter_Operator(int photon_dimension,
			      int sites,
			      double coupling,
			      double hopping,
			      double hubbardU,
			      bool periodic,
			      const spmat_t & proj);



#endif /* LINEAROPERATOR_HPP */
