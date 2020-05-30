#ifndef ARNOLDIITERATION_HPP
#define ARNOLDIITERATION_HPP

#include "Common.hpp"
#include <unordered_map>
#include "MatrixExpApply.hpp"

namespace std{
  template<typename T>
  struct hash<std::complex<T> > {
    std::size_t operator()(const std::complex<T> & c) const {
      size_t seed  = std::hash<double>()(c.real());
      seed ^= std::hash<double>()(c.imag()) + 0x9e3779b9 + (seed<<6) + (seed>>2);
      return seed;
    }
  };
}  // namespace std


/*
  Implements a simple Arnoldi iteration without restarting or anything
  fancy. Allows for continuation of the iteration if desired.
  The idea is, that the user chechks some condition himself and asks for
  the right number of consecutive iterations.

  TODO: Make eigenvalue calculation optional
  TODO: Make eigenvector calculation optional
 */
template<typename _MatrixType>
struct ArnoldiIteration {
  using MatrixType = _MatrixType;
  
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
  typedef std::complex<RealScalar> ComplexScalar;
  using EigenvalueType =  Eigen::Matrix<ComplexScalar, ColsAtCompileTime, 1,
                                        Options&(~Eigen::RowMajor),
                                        MaxColsAtCompileTime, 1>;
  using EigenvectorType = Eigen::Matrix<ComplexScalar, RowsAtCompileTime,
                                        ColsAtCompileTime,
                                        Options, MaxRowsAtCompileTime,
                                        MaxColsAtCompileTime>;

  Eigen::Block<const EigenvectorType> H() const;
  Eigen::Block<const EigenvectorType> V() const;

  const EigenvectorType & eigenvectors() const;

  const EigenvalueType &eigenvalues() const;
  /*
    Maybe TODO: Create method to sort for converged vectors and return?
   */
  int nit() const;

  vec_t apply_exp(const vec_t & vec, int nconv) const {
    return V().topLeftCorner(V().rows(), nconv)
      * matrix_exponential(H().eval()).topLeftCorner(nconv, nconv)
      * vec_t::Unit(nconv, 0)
      * vec.norm();
  }

  // Empty constructor, if you want to call compute yourself.
  ArnoldiIteration();

  /*
    param A: Input matrix for which the eigenvalues should be calculated
    param nev: number of iterations to start out with
    param ncv: Internal Storage allocation --> ncv > nev_max must hold
  */
  ArnoldiIteration(const MatrixType & A, int nev, int ncv);

  ArnoldiIteration(const MatrixType & A, int nev, int ncv,
		   const EigenvectorType & v0);

  /*
    Continues with nev arnoldi iterations using the existing matrices.
  */
  void k_n_arnoldi(const MatrixType & A, int nev);

  /*
    If the Procecure was called using a Shift invert procedure, this
    calculates the residues of the generalized eigenvalue equation
    and returns them as map (eigenvalue, residue)

    residue = (lhs * ritz_vector - rhs * ritz_value * ritz_vector).norm()
  */
  std::unordered_map<ComplexScalar, std::pair<RealScalar, EigenvectorType> >
  generalized_residues(const MatrixType & lhs,
                       const MatrixType & rhs,
                       ComplexScalar sigma);
  
  std::unordered_map<ComplexScalar, EigenvectorType>
  transformed(ComplexScalar sigma);


  void restart();
  
protected:
  int nbr_iterations;
  int nbr_converged;
  EigenvectorType m_eivec;
  EigenvalueType m_eival;
  EigenvectorType m_H;
  EigenvectorType m_V;
  bool m_isInitialized;
  bool m_eigenvectorsOk;
};

/*
  Strategy:
  - start initial iteration
  - caluclate residuals for specific eigenvalues
  - continue iteration until result satisfactory
*/


template<typename MatrixType>
ArnoldiIteration<MatrixType>::ArnoldiIteration()
  : nbr_iterations(0),
    nbr_converged(0),
    m_eivec(),
    m_eival(),
    m_H(),
    m_V(),
    m_isInitialized(false),
    m_eigenvectorsOk(false) {}

template<typename MatrixType>
ArnoldiIteration<MatrixType>::ArnoldiIteration(const MatrixType & A,
                                               int nev, int ncv)
  : nbr_iterations(0),
    nbr_converged(0),
    m_eivec(),
    m_eival(EigenvalueType::Zero(ncv)),
    m_H(EigenvectorType::Zero(A.rows(), A.cols())),
    m_V(EigenvectorType::Zero(A.rows(), A.cols())),
    m_isInitialized(false),
    m_eigenvectorsOk(false) {
  m_V.col(0).setRandom();
  // m_V.col(0).setConstant({1.0, 1.0});
  m_V.col(0) /= m_V.col(0).norm();
  k_n_arnoldi(A, nev);
}

template<typename MatrixType>
ArnoldiIteration<MatrixType>::ArnoldiIteration(const MatrixType & A,
					       int nev, int ncv, const EigenvectorType & v0)
  : nbr_iterations(0),
    nbr_converged(0),
    m_eivec(),
    m_eival(EigenvalueType::Zero(ncv)),
    m_H(EigenvectorType::Zero(ncv + 1, ncv + 1)),
    m_V(EigenvectorType::Zero(A.rows(), ncv + 1)),
    m_isInitialized(false),
    m_eigenvectorsOk(false) {
  LOG(logINFO) << "using starting vector" << std::endl;
  m_V.col(0) = v0;
  m_V.col(0) /= m_V.col(0).norm();
  k_n_arnoldi(A, nev);
}

template<typename MatrixType>
Eigen::Block<const typename ArnoldiIteration<MatrixType>::EigenvectorType>
ArnoldiIteration<MatrixType>::H() const {
  return m_H.topLeftCorner(nbr_iterations, nbr_iterations);
}

template<typename MatrixType>
Eigen::Block<const typename ArnoldiIteration<MatrixType>::EigenvectorType>
ArnoldiIteration<MatrixType>::V() const {
  return m_V.topLeftCorner(m_V.rows(), nbr_iterations);
}


template<typename MatrixType>
void ArnoldiIteration<MatrixType>::k_n_arnoldi(const MatrixType & A,
                                               int nev) {
  if (nbr_iterations + nev >= m_H.rows()) {
    m_H.conservativeResize(2 * nev + m_H.rows() + 1,
                           2 * nev + m_H.cols());
    m_V.conservativeResize(m_V.rows(), 2 * nev + m_V.cols() + 1);
    LOG(logERROR) << "insufficient memory allocated" << std::endl;
  }

  for (int n = nbr_iterations; n < nbr_iterations + nev; ++n) {
    EigenvectorType v = A * m_V.col(n);
    for (int j = 0; j < n + 1; ++j) {
      m_H(j, n) = (m_V.col(j).adjoint() * v)(0, 0);
      v -= m_H(j, n) * m_V.col(j);
    }
    m_H(n + 1, n) = v.norm();
    m_V.col(n + 1) = v / m_H(n + 1, n);
  }

  nbr_iterations += nev;

  using eigenSolver = Eigen::ComplexEigenSolver<EigenvectorType>;
  eigenSolver eigen_solver(H(), true);

  auto eival = eigen_solver.eigenvalues();
  auto eivec = eigen_solver.eigenvectors();

  m_eival = eigen_solver.eigenvalues();
  m_eivec = V() * eigen_solver.eigenvectors();
  m_isInitialized = true;
  m_eigenvectorsOk = true;
}

template<typename MatrixType>
std::unordered_map<typename ArnoldiIteration<MatrixType>::ComplexScalar,
                   std::pair<typename ArnoldiIteration<MatrixType>::RealScalar,
			     typename ArnoldiIteration<MatrixType>::EigenvectorType> >
ArnoldiIteration<MatrixType>::generalized_residues(const MatrixType & lhs,
                                                   const MatrixType & rhs,
                                                   ComplexScalar sigma) {
  std::unordered_map<ComplexScalar, std::pair<RealScalar, EigenvectorType> > residue_map;
  for (int i = 0; i < eigenvalues().size(); ++i) {
    ComplexScalar eigen_value = 1.0 / eigenvalues()[i] + sigma;
    RealScalar residue = (lhs * eigenvectors().col(i)
                          - eigen_value * rhs * eigenvectors().col(i))
      .norm();
    residue_map[eigen_value] = std::make_pair(residue, eigenvectors().col(i));
    // LOG_VAR(residue);
  }
  return residue_map;
}

template<typename MatrixType>
std::unordered_map<typename ArnoldiIteration<MatrixType>::ComplexScalar,
		   typename ArnoldiIteration<MatrixType>::EigenvectorType>
ArnoldiIteration<MatrixType>::transformed(ComplexScalar sigma) {
  std::unordered_map<ComplexScalar, EigenvectorType> residue_map;
  for (int i = 0; i < eigenvalues().size(); ++i) {
    ComplexScalar eigen_value = 1.0 / eigenvalues()[i] + sigma;
    residue_map[eigen_value] = eigenvectors().col(i);
  }
  return residue_map;
}


template<typename MatrixType>
const typename ArnoldiIteration<MatrixType>::EigenvectorType &
ArnoldiIteration<MatrixType>::eigenvectors() const {
  assert(m_isInitialized
         && "EigenSolver is not initialized.");
  assert(m_eigenvectorsOk
         && "The eigenvectors have not been computed together with the eigenvalues.");
  return m_eivec;
}

template<typename MatrixType>
const typename ArnoldiIteration<MatrixType>::EigenvalueType &
ArnoldiIteration<MatrixType>::eigenvalues() const {
  assert(m_isInitialized && "EigenSolver is not initialized.");
  return m_eival;
}

template<typename MatrixType>
int ArnoldiIteration<MatrixType>::nit() const {
  return nbr_iterations;
}

template<typename MatrixType>
void ArnoldiIteration<MatrixType>::restart() {
  vec_t vec = vec_t::Zero(eigenvectors().rows());
  for (int i = 0; i < eigenvalues().size(); ++i) {
    vec += eigenvectors().col(i);
  }
  vec /= vec.norm();

  nbr_iterations = 0;
  nbr_converged = 0;
  m_eivec = EigenvectorType();
  m_eival = EigenvalueType();
  m_H.setZero();
  m_V.setZero();
  m_isInitialized = false;
  m_eigenvectorsOk = false;
  m_V.col(0) = vec;
}

vec_t exp_krylov(const spmat_t & A, const vec_t & vec, int nruns);

vec_t exp_krylov_alt(const spmat_t & A, const vec_t & vec, int nruns);

#endif //  ARNOLDIITERATION_HPP 
