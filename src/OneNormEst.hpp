#ifndef ONENORMEST_HPP
#define ONENORMEST_HPP

#include "Common.hpp"
#include "MatrixWrappers.hpp"

mat_t sign_round_up(const mat_t X);


// _sum_abs_axis0(X):
Eigen::VectorXd sum_abs_colwise(const mat_t & X);

// _max_abs_axis1(X)
Eigen::VectorXd max_abs_rowwise(const mat_t & X);

template<typename vector_type>
bool vectors_parallel(const vector_type & LHS,
		      const vector_type & RHS) {
  assert(LHS.size() == RHS.size());
  // TODO: Added abs to make compiler happy
  return (std::abs(LHS.dot(RHS)) == LHS.size());
}

// LOL at the name
bool every_col_of_X_is_parallel_to_a_col_of_Y(const mat_t & X,
					      const mat_t & Y);

bool column_needs_resampling(int i, const mat_t & X);
/*
  column i of X needs resampling if either
  it is parallel to a previous column of X or
  it is parallel to a column of Y
*/
bool column_needs_resampling(int i, const mat_t & X, const mat_t & Y);

Eigen::VectorXi randint(int low, int high, Eigen::Index size);

template<typename MatrixType>
void resample_column(int i, MatrixType & X) {
  Eigen::VectorXi ran = randint(0, 2, X.rows()) * 2;
  ran.array() -= 1;
  X.col(i) = ran.cast <typename MatrixType::Scalar> ();
}

// directly from PEP 485
bool close(double a, double b, double rtol = 1e-05, double atol = 1e-08);

bool less_than_or_close(double a, double b);

/*
    This is Algorithm 2.2.
    Parameters
    ----------
    A : ndarray or other linear operator
        A linear operator that can produce matrix products.
    AT : ndarray or other linear operator
        The transpose of A.
    t : int, optional
        A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.
    Returns
    -------
    g : sequence
        A non-negative decreasing vector
        such that g[j] is a lower bound for the 1-norm
        of the column of A of jth largest 1-norm.
        The first entry of this vector is therefore a lower bound
        on the 1-norm of the linear operator A.
        This sequence has length t.
    ind : sequence
        The ith entry of ind is the index of the column A whose 1-norm
        is given by g[i].
        This sequence of indices has length t, and its entries are
        chosen from range(n), possibly with repetition,
        where n is the order of the operator A.
    Notes
    -----
    This algorithm is mainly for testing.
    It uses the 'ind' array in a way that is similar to
    its usage in algorithm 2.4. This algorithm 2.2 may be easier to test,
    so it gives a chance of uncovering bugs related to indexing
    which could have propagated less noticeably to algorithm 2.4.
*/
template<typename M1, typename M2>
auto _algorithm_2_2(const M1 & A, const M2 & AT, int t) {

  int n = A.rows();
  Eigen::MatrixXd X = Eigen::MatrixXd::Constant(n, t, 1.0);

  if (t > 1) {
    for (int i = 1; i < X.cols(); ++i) {
      Eigen::VectorXi tmp = randint(0, 2, n) * 2;
      X.col(i) = tmp.cast<double>();
      X.col(i).array() -= 1.0;
    }
  }
    
  X /= static_cast<double>(n);

  // Iteratively improve the lower bounds.
  // Track extra things, to assert invariants for debugging.
  Eigen::VectorXd g_prev;
  Eigen::VectorXd h_prev;
  int k = 1;
  std::vector<int> ind(t);
  std::iota (std::begin(ind), std::end(ind), 0);

  Eigen::VectorXd g;
  while (true) {
    mat_t Y = A * X;
    g = sum_abs_colwise(Y);
    double best_j = g.maxCoeff();
    // Should be descending
    std::sort(g.data(), g.data() + g.size(), std::greater<>()); 
    mat_t S = sign_round_up(Y);
    mat_t Z = AT * S;
    Eigen::VectorXd h = max_abs_rowwise(Z);

    /*
      If this algorithm runs for fewer than two iterations,
      then its return values do not have the properties indicated
      in the description of the algorithm.
      In particular, the entries of g are not 1-norms of any
      column of A until the second iteration.
      Therefore we will require the algorithm to run for at least
      two iterations, even though this requirement is not stated
      in the description of the algorithm.
    */
    if (k >= 2) {
      // TEMP: real included to compile
      if (less_than_or_close(h.maxCoeff(),
			     Z.col(best_j).real().dot(X.col(best_j)))) {
	break;
      }
    }

    // TRANSLATE!
    {
      std::vector<int> y(h.size());
      std::size_t index(0);
      std::generate(std::begin(y), std::end(y), [&]{ return index++;});

      std::sort(std::begin(y), 
		std::end(y),
		[&](int i1, int i2) {return h(i1) > h(i2);});
      ind = std::vector<int>(y.begin(), y.begin() + t);
    }
    // ind = np.argsort(h)[::-1][:t];
    // 
    h = h(ind);

    for (int j = 0; j < t; ++j) {
      X.col(j) = Eigen::VectorXd::Unit(n, ind[j]);
    }

    // Check invariant (2.2).
    if (k >= 2) {
      assert(less_than_or_close(g_prev[0], h_prev[0])
	     && "invariant (2.2) is violated");
      assert(less_than_or_close(h_prev[0], g[0])
	     && "invariant (2.2) is violated");
    }

    // Check invariant (2.3).
    if (k >= 3) {
      for (int j = 0; j < t; ++j) {
	assert(less_than_or_close(g[j], g_prev[j])
	       && "invariant (2.3) is violated");
      }
    }

    // Update for the next iteration.
    g_prev = g;
    h_prev = h;
    k += 1;

  }
  // Return the lower bounds and the corresponding column indices.
  return std::make_pair(g, ind);
}

template<typename VectorType>
int eigen_argmax(const VectorType & vec) {
  int maxval = 0;
  double max = std::numeric_limits<double>::min();
  for (Eigen::Index i = 0; i < vec.size(); ++i) {
    if (max < vec(i)) {
      max = vec(i);
      maxval = i;
    }
  }
  return maxval;
}

template<typename T1, typename T2>
std::vector<int> in1d(const T1 & lhs, const T2 & rhs) {
  std::vector<int> dst;
  dst.insert(dst.end(), lhs.data(), lhs.data() + lhs.size());
  dst.insert(dst.end(), rhs.data(), rhs.data() + rhs.size());

  std::vector<int> y(dst.size());
  std::size_t n(0);
  std::generate(std::begin(y), std::end(y), [&]{ return n++;});
  std::stable_sort(std::begin(y), std::end(y),
		   [&](int i1, int i2) {return dst[i1] < dst[i2];});

  std::vector<int> out;
  for (size_type i = 0; i + 1 < y.size(); ++i) {
  if ((y[i] < lhs.size())
	&& (i + 1 < dst.size())
	&& (dst[y[i]] == dst[y[i + 1]])) {
      out.push_back(y[i]);
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<int> invert_indexer(std::vector<int> indexer, int size);

/*
  Compute a lower bound of the 1-norm of a sparse matrix.
  Parameters
  ----------
  A : ndarray or other linear operator
  A linear operator that can produce matrix products.
  AT : ndarray or other linear operator
  The transpose of A.
  t : int, optional
  A positive parameter controlling the tradeoff between
  accuracy versus time and memory usage.
  itmax : int, optional
  Use at most this many iterations.
  Returns
  -------
  est : float
  An underestimate of the 1-norm of the sparse matrix.
  v : ndarray, optional
  The vector such that ||Av||_1 == est*||v||_1.
  It can be thought of as an input to the linear operator
  that gives an output with particularly large norm.
  w : ndarray, optional
  The vector Av which has relatively large 1-norm.
  It can be thought of as an output of the linear operator
  that is relatively large in norm compared to the input.
  nmults : int, optional
  The number of matrix products that were computed.
  nresamples : int, optional
  The number of times a parallel column was observed,
  necessitating a re-randomization of the column.
  Notes
  -----
  This is algorithm 2.4.
*/
template<typename M1, typename M2>
double _onenormest_core(const M1 & A, const M2 & AT, int t, int itmax) {

  // This function is a more or less direct translation
  // of Algorithm 2.4 from the Higham and Tisseur (2000) paper.
  assert(itmax > 1 && "at least two iterations are required");
  assert(t > 0 && "at least one column is required");
  int n = A.rows();
  assert(t < n && "t should be smaller than the order of A");
  // Track the number of big*small matrix multiplications
  // and the number of resamplings.
  int nmults = 0;
  int nresamples = 0;
  
  // "We now explain our choice of starting matrix.  We take the first
  // column of X to be the vector of 1s [...] This has the advantage that
  // for a matrix with nonnegative elements the algorithm converges
  // with an exact estimate on the second iteration, and such matrices
  // arise in applications [...]"
  Eigen::MatrixXd X = Eigen::MatrixXd::Constant(n, t, 1.0);
  // "The remaining columns are chosen as rand{-1,1},
  // with a check for and correction of parallel columns,
  // exactly as for S in the body of the algorithm."
  if (t > 1) {
    for (int i = 1; i < t; ++i) {
      // These are technically initial samples, not resamples,
      // so the resampling count is not incremented.
      resample_column(i, X);
    }
    for (int i = 0; i < t; ++i) {
      while (column_needs_resampling(i, X)) {
	  resample_column(i, X);
	  nresamples += 1;
	}

    }
  }
  //  "Choose starting matrix X with columns of unit 1-norm."
  X /= static_cast<double>(n);
  //  "indices of used unit vectors e_j"
  Eigen::VectorXi ind_hist = Eigen::VectorXi::Zero(0);
  double est_old = 0.0;
  double est = 0.0;
  mat_t S = mat_t::Zero(n, t);
  int k = 1;
  Eigen::VectorXi ind;
  int ind_best = -1;
    
  while (true) {
    mat_t Y = A * X;
    nmults += 1;
    Eigen::VectorXd mags = sum_abs_colwise(Y);
    est = mags.maxCoeff();
    int best_j = eigen_argmax(mags);
    if ((est > est_old) || (k == 2)) {
      if (k >= 2){
	ind_best = ind[best_j];
      }
      // w = Y[:, best_j];
    }
    // (1)
    if ((k >= 2) && (est <= est_old)) {
      est = est_old;
      break;
    }
    est_old = est;
    mat_t S_old = S;
    if (k > itmax) break;
    S = sign_round_up(Y);
    //Clean up Y
    Y.resize(0 ,0);
    // # (2)
    if (every_col_of_X_is_parallel_to_a_col_of_Y(S, S_old)) break;
    if (t > 1) {
      // "Ensure that no column of S is parallel to another column of S
      //  or to a column of S_old by replacing columns of S by rand{-1,1}."
      for (int i = 0; i < t; ++i) {
	while (column_needs_resampling(i, S, S_old)) {
	  resample_column(i, S);
	  nresamples += 1;
	}
      }
    }

    S_old.resize(0, 0);
    // (3)
    mat_t Z = AT * S;
    nmults += 1;
    Eigen::VectorXd h = max_abs_rowwise(Z);
    Z.resize(0, 0);
    // (4)
    if ((k >= 2) && (h.maxCoeff() == h[ind_best])) break;
    // "Sort h so that h_first >= ... >= h_last
    // and re-order ind correspondingly."
    //
    // Later on, we will need at most t+len(ind_hist) largest
    // entries, so drop the rest
    // ind = np.argsort(h)[::-1][:t+len(ind_hist)].copy();
    {
      std::vector<int> y(h.size());
      std::size_t index(0);
      std::generate(std::begin(y), std::end(y), [&]{ return index++;});

      std::sort(std::begin(y), 
		std::end(y),
		[&](int i1, int i2) {return h(i1) > h(i2);});
      ind = Eigen::Map<Eigen::VectorXi>(y.data(), t + ind_hist.size());
    }
    h.resize(0);

    if (t > 1) {
      // (5)
      // Break if the most promising t vectors have been visited already.
      int mt = std::min(t, static_cast<int>(ind.size() - 1));
      if (in1d(ind(Eigen::seq(0, mt)), ind_hist).size() == mt)  break;
      // Put the most promising unvisited vectors at the front of the list
      // and put the visited vectors at the end of the list.
      // Preserve the order of the indices induced by the ordering of h.
      auto seen = in1d(ind, ind_hist);
      Eigen::VectorXi ind_new(ind.size());
      ind_new << ind(invert_indexer(seen, ind.size())), ind(seen);
      ind = ind_new;
    }
      
    for (int j = 0; j < t; ++j) {
      X.col(j) = Eigen::VectorXd::Unit(n, ind[j]);
    }

    // Take the first t elements and delete all overlap with ind_hist
    int mt = std::min(t, static_cast<int>(ind.size() - 1));
    Eigen::VectorXi new_ind = ind(Eigen::seq(0, mt))
      (invert_indexer(in1d(ind(Eigen::seq(0, mt)), ind_hist), mt));
    Eigen::VectorXi ind_hist_new(ind_hist.size() + new_ind.size());
    ind_hist_new << ind_hist, new_ind;
    ind_hist = ind_hist_new;
    k += 1;
  }
  return est;
}

template<typename MatrixType>
double onenormest(const MatrixType & A, int t = 2, int itmax = 5) {
  return _onenormest_core(A, A.adjoint(), t, itmax);
}

double onenormest_matrix_power(const spmat_t A, int power,
			       int t = 2, int itmax = 5);

#endif /* ONENORMEST_HPP */
