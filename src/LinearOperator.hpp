#ifndef LINEAROPERATOR_HPP
#define LINEAROPERATOR_HPP

#include "MatrixWrappers.hpp"
#include "HubbardModel.hpp"
#include "Operators.hpp"
#include <memory>

template<typename _MatrixType>
struct LinearOperator {
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

  using VectorType =  Eigen::Matrix<Scalar, ColsAtCompileTime, 1,
				    Options&(~Eigen::RowMajor),
				    MaxColsAtCompileTime, 1>;

  virtual VectorType apply_to(const VectorType &) const {assert(false);};
  virtual Eigen::Index rows() const {assert(false);};
  virtual Eigen::Index cols() const {assert(false);};
  auto clone() const { return std::unique_ptr<this_t>(clone_impl()); }
  
  template<typename Rhs>
  VectorType operator*(const Eigen::MatrixBase<Rhs>& x) const {
    return this->apply_to(x.derived());
  }

  this_t & operator*=(Scalar s) {
    mult_by_scalar(s);
    return *this;
  }
  
  virtual void mult_by_scalar(Scalar s) {assert(false);}
protected:
  
  virtual this_t* clone_impl() const {assert(false);};
};

template<typename _MatrixType>
struct BareLinearOperator : public LinearOperator<_MatrixType> {
  using MatrixType = _MatrixType;
  using Base = LinearOperator<MatrixType>;
  using VectorType = typename Base::VectorType;

  virtual VectorType apply_to(const VectorType & vector) const override {
    return self * vector;
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

  using this_t = SumLinearOperator<MatrixType>;
  this_t* clone_impl() const override {return new this_t(*this);};

  using Scalar = typename Base::Scalar;
  void mult_by_scalar(Scalar s) override {
    *LHS *= s;
    *RHS *= s;
  }
  
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

  using this_t = KroneckerLinearOperator<MatrixType>;
  this_t* clone_impl() const override {return new this_t(*this);};

  using Scalar = typename Base::Scalar;
  void mult_by_scalar(Scalar s) override {LHS *= s;}
  
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

  using this_t = KroneckerIDRHSLinearOperator<MatrixType>;
  this_t* clone_impl() const override {return new this_t(*this);};

  using Scalar = typename Base::Scalar;
  void mult_by_scalar(Scalar s) override {LHS *= s;}
  
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

  using this_t = KroneckerIDLHSLinearOperator<MatrixType>;
  this_t* clone_impl() const override {return new this_t(*this);};

  using Scalar = typename Base::Scalar;
  void mult_by_scalar(Scalar s) override {RHS *= s;}
  
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

std::unique_ptr<LinearOperator<spmat_t>>
Hubbard_light_matter_Operator(int photon_dimension,
			      int sites,
			      double coupling,
			      double hopping,
			      double hubbardU,
			      bool periodic,
			      const spmat_t & proj);

#endif /* LINEAROPERATOR_HPP */
