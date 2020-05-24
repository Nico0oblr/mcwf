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

#endif /* SPARSEADDONS_H */
