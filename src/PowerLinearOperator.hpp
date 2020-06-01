#ifndef POWERLINEAROPERATOR_HPP
#define POWERLINEAROPERATOR_HPP

template<typename _MatrixType>
struct PowerLinearOperator : public LinearOperator<_MatrixType>{
  using Base = LinearOperator<_MatrixType>;
  using MatrixType = typename Base::MatrixType;
  using VectorType = typename Base::VectorType;
  using AdjointVectorType = typename Base::AdjointVectorType;
  using BasePointer = std::unique_ptr<Base>;

  Eigen::Index rows() const override {
    return self->rows();
  }

  Eigen::Index cols() const override {
    return self->cols();
  }

  VectorType apply_to(const VectorType & vector) const override {
    if (power == 0) return vector;
    VectorType out = self->apply_to(vector);
    for (int j = 1; j < power; ++j) out = self->apply_to(out);
    return out;
  }

  AdjointVectorType applied_to(const AdjointVectorType & vector) const override {
    if (power == 0) return vector;
    AdjointVectorType out = self->applied_to(vector);
    for (int j = 1; j < power; ++j) out = self->applied_to(out);
    return out;
  }

  PowerLinearOperator(const Base & A, int p)
    :self(A.clone()), power(p) {assert(power >= 0);}

  PowerLinearOperator(const PowerLinearOperator<MatrixType> & other)
    :self(other.self->clone()), power(other.power) {}

  using Scalar = typename Base::Scalar;
  void mult_by_scalar(Scalar s) override
  {self->mult_by_scalar(std::pow(s, 1.0 / power));}

  void adjointInPlace() override {self->adjointInPlace();}
  
  MatrixType eval() const override {
    if (power == 0) return MatrixType::Identity(rows(), cols());
    MatrixType selfeval = self->eval(); // Just in case this is expensive
    MatrixType out = selfeval;
    for (int j = 0; j < power; ++j) out = out * selfeval;
    return out;
  }
  
  using this_t = PowerLinearOperator<MatrixType>;
  virtual std::unique_ptr<Base>
  clone() const override {return std::make_unique<this_t>(*this);}
  
  ~PowerLinearOperator() override {self.reset();}
  this_t& operator=(const this_t & other)
  {return *this = this_t(other);}
  
  BasePointer self;
  int power;
};

#endif /* POWERLINEAROPERATOR_HPP */
