#ifndef LINEAROPERATOR_HPP
#define LINEAROPERATOR_HPP

#include "Kron.hpp"
#include "HubbardModel.hpp"
#include "Operators.hpp"
#include <memory>
#include "LinearOperatorTraits.hpp"

template<typename _MatrixType>
struct LinearOperator {
  using MatrixType = typename _MatrixType::PlainObject;
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

  virtual VectorType apply_to(const VectorType &) const = 0;
  virtual AdjointVectorType applied_to(const AdjointVectorType &) const
  {assert(false && "Called Abstract base class method");}
  virtual MatrixType eval() const = 0;
  virtual Eigen::Index rows() const = 0;
  virtual Eigen::Index cols() const = 0;
  virtual void mult_by_scalar(Scalar s) = 0;
  virtual void adjointInPlace() = 0;
  virtual std::unique_ptr<this_t> clone() const = 0;

  std::unique_ptr<this_t> adjoint() const {
    std::unique_ptr<this_t> copy = this->clone();
    copy->adjointInPlace();
    return copy;
  }

  /*If the rhs is not dense, just evaluate for now*/
  template<typename Rhs>
  auto operator*(const Eigen::EigenBase<Rhs>& x) const {
    return this->eval() * x.derived();
  }

  /*If the lhs is not dense, just evaluate for now*/
  template<typename OtherDerived> friend
  auto operator*(const Eigen::EigenBase<OtherDerived>& lhs,
	    const this_t & rhs) {
    return lhs.derived() * rhs.eval();
  }
  
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

  int size() const {return rows() * cols();}

  virtual ~LinearOperator() = 0;
};

template<typename MatrixType>
LinearOperator<MatrixType>::~LinearOperator() {}

std::unique_ptr<LinearOperator<spmat_t>>
Hubbard_light_matter_Operator(int photon_dimension,
			      int sites,
			      double coupling,
			      double hopping,
			      double hubbardU,
			      bool periodic,
			      const spmat_t & proj);

#include "LinearOperatorEigenProd.hpp"
#include "BareLinearOperator.hpp"
#include "KroneckerLinearOperator.hpp"
#include "SumLinearOperator.hpp"
#include "MultLinearOperator.hpp"
#include "PowerLinearOperator.hpp"
#include "DoubledLinearOperator.hpp"
#include "LinearOperations.hpp"
using lo_ptr = std::unique_ptr<LinearOperator<calc_mat_t>>;
#endif /* LINEAROPERATOR_HPP */
