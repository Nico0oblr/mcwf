#ifndef SUMLINEAROPERATOR_HPP
#define SUMLINEAROPERATOR_HPP

template<typename _MatrixType>
struct SumLinearOperator : public LinearOperator<_MatrixType> {
  using Base = LinearOperator<_MatrixType>;
  using MatrixType = typename Base::MatrixType;
  using BasePointer = std::unique_ptr<Base>;
  using VectorType = typename Base::VectorType;
  using AdjointVectorType = typename Base::AdjointVectorType;
  using Scalar = typename Base::Scalar;

  VectorType apply_to(const VectorType & vector) const override {
    return LHS->apply_to(LHS_s * vector) + RHS->apply_to(RHS_s * vector);
  }

  AdjointVectorType applied_to(const AdjointVectorType & vector) const override {
    return LHS->applied_to(LHS_s * vector) + RHS->applied_to(RHS_s * vector);
  }
  
  Eigen::Index rows() const override {
    return LHS->rows();
  }
  
  Eigen::Index cols() const override {
    return LHS->cols();
  }

  SumLinearOperator(const Base & A, const Base & B,
		    Scalar a = Scalar(1.0), Scalar b = Scalar(1.0))
    :LHS(A.clone()), RHS(B.clone()),
     LHS_s(a), RHS_s(b){
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
  }

  SumLinearOperator(const SumLinearOperator<MatrixType> & other)
    :LHS(other.LHS->clone()), RHS(other.RHS->clone()),
     LHS_s(other.LHS_s), RHS_s(other.RHS_s){}

  MatrixType eval() const override {
    return LHS_s * LHS->eval() + RHS_s * RHS->eval();
  }

  void adjointInPlace() override {
    LHS->adjointInPlace();
    RHS->adjointInPlace();
    LHS_s = std::conj(LHS_s);
    RHS_s = std::conj(RHS_s);
  }

  void mult_by_scalar(Scalar s) override {
    LHS_s *= s;
    RHS_s *= s;
  }

  using this_t = SumLinearOperator<MatrixType>;
  virtual std::unique_ptr<Base>
  clone() const override {return std::make_unique<this_t>(*this);}
  ~SumLinearOperator() override {LHS.reset();RHS.reset();}
  
  BasePointer LHS, RHS;
  Scalar LHS_s, RHS_s;
};

#endif /* SUMLINEAROPERATOR_HPP */
