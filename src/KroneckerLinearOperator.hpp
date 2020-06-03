#ifndef KRONECKERLINEAROPERATOR_HPP
#define KRONECKERLINEAROPERATOR_HPP

/*
  Requires explicit iteration over entries, until abstracted
  this requires actual matrices as entries 
*/
template<typename _MatrixType>
struct KroneckerLinearOperator : public LinearOperator<_MatrixType>{
  using Base = LinearOperator<_MatrixType>;
  using MatrixType = typename Base::MatrixType;
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

  void adjointInPlace() override {LHS.adjointInPlace();RHS.adjointInPlace();}
  
  MatrixType eval() const override {
    return Eigen::kroneckerProduct(LHS, RHS);
  }
  
  using this_t = KroneckerLinearOperator<MatrixType>;
  virtual std::unique_ptr<Base>
  clone() const override {return std::make_unique<this_t>(*this);}
  ~KroneckerLinearOperator() override {LHS.resize(0, 0);RHS.resize(0,0);}
  
  MatrixType LHS, RHS;
};

template<typename _MatrixType>
struct KroneckerIDRHSLinearOperator
  : public LinearOperator<_MatrixType>{
  using Base = LinearOperator<_MatrixType>;
  using MatrixType = typename Base::MatrixType;
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

  void adjointInPlace() override {LHS.adjointInPlace();}
  
  using Scalar = typename Base::Scalar;
  void mult_by_scalar(Scalar s) override {LHS *= s;}

  MatrixType eval() const override {
    return tensor_identity(LHS, codimension);
  }

  using this_t = KroneckerIDRHSLinearOperator<MatrixType>;
  virtual std::unique_ptr<Base>
  clone() const override {return std::make_unique<this_t>(*this);}
 
  ~KroneckerIDRHSLinearOperator() override {LHS.resize(0, 0);}
  //KroneckerIDRHSLinearOperator(const this_t &) = default;
  //this_t& operator=(const this_t &) = default;
  
  MatrixType LHS;
  int codimension;
};

template<typename _MatrixType>
struct KroneckerIDLHSLinearOperator
  : public LinearOperator<_MatrixType>{
  using Base = LinearOperator<_MatrixType>;
  using MatrixType = typename Base::MatrixType;
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

  void adjointInPlace() override {RHS.adjointInPlace();}
  
  MatrixType eval() const override {
    return tensor_identity_LHS(RHS, codimension);
  }

  using this_t = KroneckerIDLHSLinearOperator<MatrixType>;
  virtual std::unique_ptr<Base>
  clone() const override {return std::make_unique<this_t>(*this);}

  ~KroneckerIDLHSLinearOperator() override {RHS.resize(0, 0);}
  //KroneckerIDLHSLinearOperator(const this_t &) = default;
  //this_t& operator=(const this_t &) = default;
  
  MatrixType RHS;
  int codimension;
};

#endif /* KRONECKERLINEAROPERATOR_HPP */
