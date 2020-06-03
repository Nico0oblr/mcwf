#ifndef LINEAROPERATIONS_HPP
#define LINEAROPERATIONS_HPP


template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
kroneckerOperator(const MatrixType & A, const MatrixType & B) {
  return std::make_unique<KroneckerLinearOperator<MatrixType>>(A, B);
}


template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
kroneckerOperator_IDRHS(const MatrixType & A, int dim) {
  return std::make_unique<KroneckerIDRHSLinearOperator<MatrixType>>(A, dim);
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
kroneckerOperator_IDLHS(const MatrixType & A, int dim) {
  return std::make_unique<KroneckerIDLHSLinearOperator<MatrixType>>(A, dim);
}

/*Operator+*/
template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
sumOperator(const LinearOperator<MatrixType> & A,
		 const LinearOperator<MatrixType> & B) {
  return std::make_unique<SumLinearOperator<MatrixType>>(A, B);
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

/* Operator- */
template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
operator-(const LinearOperator<MatrixType> & A,
	  const LinearOperator<MatrixType> & B) {
  return sumOperator(A, *(-1.0 * B));
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
operator-(const std::unique_ptr<LinearOperator<MatrixType>> & A,
	  const LinearOperator<MatrixType> & B) {
  return sumOperator(*A, *(-1.0 * B));
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
operator-(const LinearOperator<MatrixType> & A,
	  const std::unique_ptr<LinearOperator<MatrixType>> & B) {
  return sumOperator(A, *(-1.0 * B));
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
operator-(const std::unique_ptr<LinearOperator<MatrixType>> & A,
	  const std::unique_ptr<LinearOperator<MatrixType>> & B) {
  return sumOperator(*A, *(-1.0 * B));
}

/* Operator* */

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
operator*(const std::unique_ptr<LinearOperator<MatrixType>> & A,
	  typename MatrixType::Scalar sc) {
  auto tmp = A->clone();
  tmp->mult_by_scalar(sc);
  return tmp;
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
operator*(typename MatrixType::Scalar sc,
	  const std::unique_ptr<LinearOperator<MatrixType>> & A) {
  auto tmp = A->clone();
  tmp->mult_by_scalar(sc);
  return tmp;
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
operator*(const LinearOperator<MatrixType> & A,
	  typename MatrixType::Scalar sc) {
  auto tmp = A.clone();
  tmp->mult_by_scalar(sc);
  return tmp;
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
operator*(typename MatrixType::Scalar sc,
	       const LinearOperator<MatrixType> & A) {
  auto tmp = A.clone();
  tmp->mult_by_scalar(sc);
  return tmp;
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
multOperator(const LinearOperator<MatrixType> & A,
	     const LinearOperator<MatrixType> & B) {
  return std::make_unique<MultLinearOperator<MatrixType>>(A, B);
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
operator*(const LinearOperator<MatrixType> & A,
	  const LinearOperator<MatrixType> & B) {
  return multOperator(A, B);
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
operator*(const std::unique_ptr<LinearOperator<MatrixType>> & A,
	  const LinearOperator<MatrixType> & B) {
  return multOperator(*A, B);
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
operator*(const LinearOperator<MatrixType> & A,
	  const std::unique_ptr<LinearOperator<MatrixType>> & B) {
  return multOperator(A, *B);
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
operator*(const std::unique_ptr<LinearOperator<MatrixType>> & A,
	  const std::unique_ptr<LinearOperator<MatrixType>> & B) {
  return multOperator(*A, *B);
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
operatorize(const MatrixType & mat) {
  return std::make_unique<BareLinearOperator<MatrixType>>(mat);
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
doubleOperator(const LinearOperator<MatrixType> & A) {
  return DoubledLinearOperator<MatrixType>(A).clone();
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
doubleOperator(const std::unique_ptr<LinearOperator<MatrixType>> & A) {
  return DoubledLinearOperator<MatrixType>(*A).clone();
}

template<typename MatrixType>
std::vector<std::unique_ptr<LinearOperator<MatrixType>>>
doubleOperatorVector(const std::vector<std::unique_ptr<LinearOperator<MatrixType>>> & A) {
  std::vector<std::unique_ptr<LinearOperator<MatrixType>>> out;
  for (const auto & el : A) out.push_back(doubleOperator(el));
  return out;
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
powerOperator(const LinearOperator<MatrixType> & A, int p) {
  return std::make_unique<PowerLinearOperator<MatrixType>>(A, p);
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
powerOperator(const std::unique_ptr<LinearOperator<MatrixType>> & A, int p) {
  return std::make_unique<PowerLinearOperator<MatrixType>>(*A, p);
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
scale_and_add(const LinearOperator<MatrixType> & LHS,
	      const LinearOperator<MatrixType> & RHS,
	      typename MatrixType::Scalar a,
	      typename MatrixType::Scalar b) {
  return std::make_unique<SumLinearOperator<MatrixType>>(LHS, RHS, a, b);
}

template<typename MatrixType>
std::unique_ptr<LinearOperator<MatrixType>>
scale_rhs_and_add(const LinearOperator<MatrixType> & LHS,
		  const LinearOperator<MatrixType> & RHS,
		  typename MatrixType::Scalar a) {
  return std::make_unique<SumLinearOperator<MatrixType>>(LHS, RHS, 1.0, a);
}


#endif /* LINEAROPERATIONS_HPP */
