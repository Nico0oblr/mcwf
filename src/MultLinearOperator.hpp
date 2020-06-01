#ifndef MULTLINEAROPERATOR_HPP
#define MULTLINEAROPERATOR_HPP

template<typename _MatrixType>
struct MultLinearOperator : public LinearOperator<_MatrixType> {
  using Base = LinearOperator<_MatrixType>;
  using MatrixType = typename Base::MatrixType;
  using BasePointer = std::unique_ptr<Base>;
  using VectorType = typename Base::VectorType;
  using AdjointVectorType = typename Base::AdjointVectorType;

  VectorType apply_to(const VectorType & vector) const override {
    return LHS->apply_to(RHS->apply_to(vector));
  }

  AdjointVectorType applied_to(const AdjointVectorType & vector) const override {
    return RHS->applied_to(LHS->applied_to(vector));
  }
  
  Eigen::Index rows() const override {
    return LHS->rows();
  }
  
  Eigen::Index cols() const override {
    return RHS->cols();
  }

  MultLinearOperator(const Base & A, const Base & B)
    :LHS(A.clone()), RHS(B.clone()) {
    assert(A.cols() == B.rows());
  }

  MultLinearOperator(const MultLinearOperator<MatrixType> & other)
    :LHS(other.LHS->clone()), RHS(other.RHS->clone()) {}

  MatrixType eval() const override {return LHS->eval() * RHS->eval();}

  void adjointInPlace() override
  {LHS->adjointInPlace();RHS->adjointInPlace(); std::swap(LHS, RHS);}
  
  using Scalar = typename Base::Scalar;
  void mult_by_scalar(Scalar s) override {
    *LHS *= s;
  }

  using this_t = MultLinearOperator<MatrixType>;
  virtual std::unique_ptr<Base>
  clone() const override {return std::make_unique<this_t>(*this);}
  ~MultLinearOperator() override  {LHS.reset();RHS.reset();}
  
  BasePointer LHS, RHS;
};

#endif /* MULTLINEAROPERATOR_HPP */
