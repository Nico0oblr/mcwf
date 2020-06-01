#ifndef SPARSEADDONS_H
#define SPARSEADDONS_H

template<typename Other>
SparseMatrix(const EigenBase<Other> & other)
:SparseMatrix(other.derived().sparseView()) {}

static SparseMatrix<_Scalar, _Options, _StorageIndex> Identity(Index rows,
							       Index cols) {
  SparseMatrix<_Scalar, _Options, _StorageIndex> mat(rows, cols);
  mat.setIdentity();
  return mat;
}

static SparseMatrix<_Scalar, _Options, _StorageIndex> Zero(Index rows,
							   Index cols) {
  SparseMatrix<_Scalar, _Options, _StorageIndex> mat(rows, cols);
  return mat;
}

Scalar trace() const {
  Scalar sum(0);
  for (int k = 0; k < outerSize(); ++k)
    sum += coeff(k,k);
  return sum;
}

double oneNorm() const {
  VectorXd cval = VectorXd::Zero(cols());
  for (int k = 0; k < outerSize(); ++k) {
    for (InnerIterator it(*this, k); it; ++it)
      {
	cval(it.col()) += std::abs(it.value());
      }
  }

  // return (this->cwiseAbs() * VectorXd::Ones(this->cols())).maxCoeff();
  return cval.maxCoeff();
}

void adjointInPlace() {*this = this->adjoint().eval();}

double infNorm() const {
  VectorXd rval = VectorXd::Zero(rows());
  for (int k = 0; k < outerSize(); ++k) {
    for (InnerIterator it(*this, k); it; ++it)
      {
	rval(it.row()) += std::abs(it.value());
      }
  }
  return rval.maxCoeff();
  // return (RowVectorXd::Ones(this->rows()) * this->cwiseAbs()).maxCoeff();
}

#endif /* SPARSEADDONS_H */
