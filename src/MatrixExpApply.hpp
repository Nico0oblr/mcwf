#ifndef MATRIXEXPAPPLY_HPP
#define MATRIXEXPAPPLY_HPP

#include "OneNormEst.hpp"
#include <unordered_map>

double _exact_inf_norm(const spmat_t & A);


static std::map<int, double>
_theta = {{1, 2.29e-16},
	  {2, 2.58e-8},
	  {3, 1.39e-5},
	  {4, 3.40e-4},
	  {5, 2.40e-3},
	  {6, 9.07e-3},
	  {7, 2.38e-2},
	  {8, 5.00e-2},
	  {9, 8.96e-2},
	  {10, 1.44e-1},
	  // # 11
	  {11, 2.14e-1},
	  {12, 3.00e-1},
	  {13, 4.00e-1},
	  {14, 5.14e-1},
	  {15, 6.41e-1},
	  {16, 7.81e-1},
	  {17, 9.31e-1},
	  {18, 1.09},
	  {19, 1.26},
	  {20, 1.44},
	  // # 21
	  {21, 1.62},
	  {22, 1.82},
	  {23, 2.01},
	  {24, 2.22},
	  {25, 2.43},
	  {26, 2.64},
	  {27, 2.86},
	  {28, 3.08},
	  {29, 3.31},
	  {30, 3.54},
	  // # The rest are from table 3.1 of
	  // # Computing the Action of the Matrix Exponential.
	  {35, 4.7},
	  {40, 6.0},
	  {45, 7.2},
	  {50, 8.5},
	  {55, 9.9}};

/*
  Information about an operator is lazily computed.
  The information includes the exact 1-norm of the operator,
  in addition to estimates of 1-norms of powers of the operator.
  This uses the notation of Computing the Action (2011).
  This class is specialized enough to probably not be of general interest
  outside of this module.
*/
template<typename MatrixType>
class LazyOperatorNormInfo {
  const MatrixType * m_A;
  double m_A_1_norm;
  int m_ell;
  int m_scale;
  std::unordered_map<int, double> m_d;
public:
  /*
    Provide the operator and some norm-related information.
    Parameters
    ----------
    A : linear operator
    The operator of interest.
    A_1_norm : float, optional
    The exact 1-norm of A.
    ell : int, optional
    A technical parameter controlling norm estimation quality.
    scale : int, optional
    If specified, return the norms of scale*A instead of A.
  */
  LazyOperatorNormInfo(const MatrixType & A,
		       double A_1_norm = -1.0,
		       int ell = 2,
		       int scale = 1)
    :m_A(&A), m_A_1_norm(A_1_norm), m_ell(ell), m_scale(scale), m_d() {}

  void set_scale(int scale) {m_scale = scale;}

  double onenorm() {
    if(m_A_1_norm < 0) m_A_1_norm = m_A->oneNorm();
    return m_scale * m_A_1_norm;
  }
  
  double d(int p) {
    if (m_d.find(p) == m_d.end()) {
      double est = onenormest_matrix_power(*m_A, p, m_ell);
      m_d.emplace(p, std::pow(est, (1.0 / p)));
    }
    return m_scale * m_d.at(p);
  }

  double alpha(int p) {
    return std::max(d(p), d(p+1));
  }
};

/*
  Compute the largest positive integer p such that p*(p-1) <= m_max + 1.
  Do this in a slightly dumb way, but safe and not too slow.
  Parameters
  ----------
  m_max : int
  A count related to bounds.
*/
int compute_p_max(int m_max);
/*
  A helper function for computing bounds.
  This is equation (3.10).
  It measures cost in terms of the number of required matrix products.
  Parameters
  ----------
  m : int
  A valid key of _theta.
  p : int
  A matrix power.
  norm_info : LazyOperatorNormInfo
  Information about 1-norms of related operators.
  Returns
  -------
  cost_div_m : int
  Required number of matrix products divided by m.
*/
long _compute_cost_div_m(int m, int p, LazyOperatorNormInfo<spmat_t> & norm_info);

  /*
    A helper function for the _expm_multiply_* functions.
    Parameters
    ----------
    norm_info : LazyOperatorNormInfo
        Information about norms of certain linear operators of interest.
    n0 : int
        Number of columns in the _expm_multiply_* B matrix.
    tol : float
        Expected to be
        :math:`2^{-24}` for single precision or
        :math:`2^{-53}` for double precision.
    m_max : int
        A value related to a bound.
    ell : int
        The number of columns used in the 1-norm approximation.
        This is usually taken to be small, maybe between 1 and 5.
    Returns
    -------
    best_m : int
        Related to bounds for error control.
    best_s : int
        Amount of scaling.
    Notes
    -----
    This is code fragment (3.1) in Al-Mohy and Higham (2011).
    The discussion of default values for m_max and ell
    is given between the definitions of equation (3.11)
    and the definition of equation (3.12).
  */
std::pair<int, long> _fragment_3_1(LazyOperatorNormInfo<spmat_t> & norm_info,
				  int n0, double tol, int m_max = 55,
				  int ell = 2);

/*
  A helper function for the _expm_multiply_* functions.
  Parameters
  ----------
  A_1_norm : float
  The precomputed 1-norm of A.
  n0 : int
  Number of columns in the _expm_multiply_* B matrix.
  m_max : int
  A value related to a bound.
  ell : int
  The number of columns used in the 1-norm approximation.
  This is usually taken to be small, maybe between 1 and 5.
  Returns
  -------
  value : bool
  Indicates whether or not the condition has been met.
  Notes
  -----
  This is condition (3.13) in Al-Mohy and Higham (2011).
*/
bool _condition_3_13(double A_1_norm, int n0, int m_max, int ell);

/*
  Compute the action of the matrix exponential at a single time point.
  Parameters
  ----------
  A : transposable linear operator
  The operator whose exponential is of interest.
  B : ndarray
  The matrix to be multiplied by the matrix exponential of A.
  t : float
  A time point.
  balance : bool
  Indicates whether or not to apply balancing.
  Returns
  -------
  F : ndarray
  :math:`e^{t A} B`
  Notes
  -----
  This is algorithm (3.2) in Al-Mohy and Higham (2011).
*/
spmat_t expm_multiply_simple(const spmat_t & A, const vec_t & B,
			     double t = 1.0);

spmat_t _expm_multiply_simple_core(const spmat_t & A,
				   const vec_t & B,
				   double t, std::complex<double> mu,
				   int m_star, long s,
				   double tol);

#endif /* MATRIXEXPAPPLY_HPP */
