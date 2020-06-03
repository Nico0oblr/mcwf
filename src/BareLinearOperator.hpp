#ifndef BARELINEAROPERATOR_HPP
#define BARELINEAROPERATOR_HPP

template<typename _MatrixType>
struct BareLinearOperator : public LinearOperator<_MatrixType> {
  using Base = LinearOperator<_MatrixType>;
  using MatrixType = typename Base::MatrixType;
  using VectorType = typename Base::VectorType;
  using AdjointVectorType = typename Base::AdjointVectorType;

  virtual VectorType apply_to(const VectorType & vector) const override {
    return self * vector;
  }

  AdjointVectorType applied_to(const AdjointVectorType & vector) const override {
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

  void adjointInPlace() override {self.adjointInPlace();}

  MatrixType eval() const override {return self;}
  
  using this_t = BareLinearOperator<MatrixType>;
  virtual std::unique_ptr<Base>
  clone() const override {return std::make_unique<this_t>(*this);}
  ~BareLinearOperator() override {self.resize(0, 0);}
  //BareLinearOperator(const this_t &) = default;
  //this_t& operator=(const this_t &) = default;
  
  MatrixType self;
};

#endif /* BARELINEAROPERATOR_HPP */
