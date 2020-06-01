#ifndef DOUBLEDLINEAROPERATOR_HPP
#define DOUBLEDLINEAROPERATOR_HPP

template<typename _MatrixType>
struct DoubledLinearOperator : public LinearOperator<_MatrixType>{
  using Base = LinearOperator<_MatrixType>;
  using MatrixType = typename Base::MatrixType;
  using VectorType = typename Base::VectorType;
  using AdjointVectorType = typename Base::AdjointVectorType;
  using BasePointer = std::unique_ptr<Base>;

  Eigen::Index rows() const override {
    return self->rows() * 2;
  }

  Eigen::Index cols() const override {
    return self->cols() * 2;
  }

  VectorType apply_to(const VectorType & vector) const override {
    VectorType copy = vector;
    assert(copy.size() == rows());
    copy(Eigen::seq(0, self->rows() - 1))
      = self->apply_to(copy(Eigen::seq(0, self->rows() - 1)));
    copy(Eigen::seq(self->rows(), 2 * self->rows() - 1))
      = self->apply_to(copy(Eigen::seq(self->rows(), 2 * self->rows() - 1)));
    return copy;
  }

  AdjointVectorType applied_to(const AdjointVectorType & vector) const override {
    VectorType copy = vector;
    assert(copy.size() == cols());
    copy(Eigen::seq(0, self->rows() - 1))
      = self->applied_to(copy(Eigen::seq(0, self->cols() - 1)));
    copy(Eigen::seq(self->cols(), 2 * self->cols() - 1))
      = self->applied_to(copy(Eigen::seq(self->cols(), 2 * self->cols() - 1)));
    return copy;
  }

  DoubledLinearOperator(const Base & A)
    :self(A.clone()) {}

  DoubledLinearOperator(const DoubledLinearOperator<MatrixType> & other)
    :self(other.self->clone()) {}

  using Scalar = typename Base::Scalar;
  void mult_by_scalar(Scalar s) override {self->mult_by_scalar(s);}

  void adjointInPlace() override {self->adjointInPlace();}
  
  MatrixType eval() const override {
    return double_matrix(self->eval());
  }
  
  using this_t = DoubledLinearOperator<MatrixType>;
  virtual std::unique_ptr<Base>
  clone() const override {return std::make_unique<this_t>(*this);}

  ~DoubledLinearOperator() override {self.reset();}
  this_t& operator=(const this_t & other)
  {return *this = this_t(other);}
  
  BasePointer self;
};

#endif /* DOUBLEDLINEAROPERATOR_HPP */
