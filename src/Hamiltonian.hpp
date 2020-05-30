#ifndef HAMILTONAIN_HPP
#define HAMILTONAIN_HPP

#include "Common.hpp"
#include "Operators.hpp"
#include <memory>
#include "PadeExponential.hpp"

/*
  TODO: Take care of dimensionality
*/

inline spmat_t KroneckerProduct(const spmat_t & A, const spmat_t & B) {
  return Eigen::kroneckerProduct(mat_t(A), mat_t(B)).sparseView();
}

inline mat_t KroneckerProduct(const mat_t & A, const mat_t & B) {
  return Eigen::kroneckerProduct(A, B);
}

template<typename _MatrixType>
class Hamiltonian {
public:
  using MatrixType = _MatrixType;
  virtual MatrixType operator()(double time) const = 0;

  virtual vec_t propagate(double time, double dt,
			  const vec_t & state) = 0;

  virtual MatrixType propagator(double time, double dt) = 0;

  auto clone() const { return std::unique_ptr<Hamiltonian>(clone_impl()); }

  virtual bool is_time_dependent() const;

  virtual int dimension() const = 0;

  virtual void add(const MatrixType & mat) = 0;

  virtual void tensor(const MatrixType & mat, bool this_rhs = false) = 0;

protected:
  
  virtual Hamiltonian* clone_impl() const = 0;
};


template<typename _MatrixType>
class TimeIndependentHamiltonian :
  public Hamiltonian<_MatrixType> {
public:
  using this_t = TimeIndependentHamiltonian<_MatrixType>;
  using MatrixType = _MatrixType;
  
  MatrixType operator()(double /*time*/) const override {
    return m_hamiltonian;
  }

  vec_t propagate(double time, double dt,
		  const vec_t & state) override {
    return propagator(time, dt) * state;
  }

  MatrixType propagator(double /*time*/, double dt) override {
    if ((m_propagator.size() == m_hamiltonian.size())
	|| (std::abs(m_last_dt - dt) > tol)) {
      m_propagator = matrix_exponential_taylor(-1.0 * m_hamiltonian * dt, 4);
    }
    return m_propagator;
  }

  TimeIndependentHamiltonian(const MatrixType & hamiltonian)
    :m_hamiltonian(hamiltonian), m_propagator(), m_last_dt(0.0) {}

  TimeIndependentHamiltonian(const MatrixType & hamiltonian,
			     const MatrixType & propagator,
			     double last_dt)
    :m_hamiltonian(hamiltonian), m_propagator(propagator), m_last_dt(last_dt) {}

  // Arithmetic functions
  this_t & operator+=(const this_t & other) {
    reset();
    m_hamiltonian += other.m_hamiltonian;
    return *this;
  }

  this_t & operator-=(const this_t & other) {
    reset();
    m_hamiltonian -= other.m_hamiltonian;
    return *this;
  }

  this_t & operator*=(const this_t & other) {
    reset();
    m_hamiltonian *= other.m_hamiltonian;
    return *this;
  }

  this_t operator-() {
    this_t ham(-m_hamiltonian);
    ham.m_propagator = m_propagator.adjoint();
    ham.m_last_dt = m_last_dt;
    return ham;
  }

  this_t operator+() {
    return *this;
  }

  template<typename MatrixType2>
  explicit operator TimeIndependentHamiltonian<MatrixType2>() {
    TimeIndependentHamiltonian<MatrixType2> out(MatrixType2(m_hamiltonian),
						MatrixType2(m_propagator),
						m_last_dt);
    return out;
  }

  bool is_time_dependent() const override {return false;}

  void add(const MatrixType & mat) override {
    (*this) += TimeIndependentHamiltonian<MatrixType>(mat);
  }

  void tensor(const MatrixType & mat, bool this_rhs = false) override {
    reset();
    if(this_rhs) {
      m_hamiltonian = KroneckerProduct(mat, m_hamiltonian);
    } else {
      m_hamiltonian = KroneckerProduct(m_hamiltonian, mat);
    }
  }

  int dimension() const override {
    assert(m_hamiltonian.rows() == m_hamiltonian.cols());
    return m_hamiltonian.rows();
  }
  
private:

  void reset() {
    m_propagator.resize(0, 0);
    m_last_dt = 0.0;
  }
  
  MatrixType m_hamiltonian;
  MatrixType m_propagator;
  double m_last_dt;

protected:

  TimeIndependentHamiltonian* clone_impl()
    const override {return new TimeIndependentHamiltonian(*this);};
};


template<typename MatrixType>
MatrixType
fourth_oder_timeordered_exponential(const std::function<MatrixType(double)> & H,
				    double t, double dt) {
  static double c1 = (3.0 - 2.0 * std::sqrt(3.0)) / 12.0;
  static double c2 = (3.0 + 2.0 * std::sqrt(3.0)) / 12.0;
  MatrixType H1 = H(t + (0.5 - std::sqrt(3.0) / 6.0) * dt);
  MatrixType H2 = H(t + (0.5 + std::sqrt(3.0) / 6.0) * dt);
  MatrixType fac1 = expm(-1.0i * (c1 * H1 + c2 * H2) * dt);
  MatrixType fac2 = expm(-1.0i * (c2 * H1 + c1 * H2) * dt);
  return fac1 * fac2;
}

template<typename MatrixType>
vec_t
fourth_oder_timeordered_exponential(const std::function<MatrixType(double)> & H,
				    double t, double dt, const vec_t & state) {
  static double c1 = (3.0 - 2.0 * std::sqrt(3.0)) / 12.0;
  static double c2 = (3.0 + 2.0 * std::sqrt(3.0)) / 12.0;
  MatrixType H1 = H(t + (0.5 - std::sqrt(3.0) / 6.0) * dt);
  MatrixType H2 = H(t + (0.5 + std::sqrt(3.0) / 6.0) * dt);
  MatrixType fac1 = -1.0i * (c1 * H1 + c2 * H2) * dt;
  MatrixType fac2 = -1.0i * (c2 * H1 + c1 * H2) * dt;
  // std::cout << "fac1 norm: " << fac1.norm() << std::endl;
  // std::cout << "fac2 norm: " << fac2.norm() << std::endl;

  // vec_t apply_fac2 = apply_matrix_exponential_taylor(fac2, state, 4);
  // vec_t apply_fac1 = apply_matrix_exponential_taylor(fac1, apply_fac2, 4);
  vec_t apply_fac2 = expm(fac2) * state;
  vec_t apply_fac1 = expm(fac1) * apply_fac2;
  return apply_fac1;
}


template<typename _MatrixType>
class TimeDependentHamiltonian :
  public Hamiltonian<_MatrixType> {
public:
  using MatrixType = _MatrixType;
  using this_t = TimeDependentHamiltonian<MatrixType>;
  MatrixType operator()(double time) const override {
    return m_hamiltonian(time);
  }

  vec_t propagate(double time, double dt,
		  const vec_t & state) override {
    return fourth_oder_timeordered_exponential(m_hamiltonian, time, dt, state);
  }

  MatrixType propagator(double time, double dt) override {
    return fourth_oder_timeordered_exponential(m_hamiltonian, time, dt);
  }

  TimeDependentHamiltonian(const std::function<MatrixType(double)> & hamiltonian,
			   int dimension)
    : m_hamiltonian(hamiltonian), m_dimension(dimension) {}

  TimeDependentHamiltonian(const TimeIndependentHamiltonian<MatrixType> & ham)
    : m_dimension(ham.dimension()) {
    MatrixType value = ham(0.0);
    m_hamiltonian = [value](double /*time*/){return value;};
  }

  template<typename comp_func>
  this_t & compose_with(const TimeDependentHamiltonian<MatrixType> & other,
			const comp_func & comp) {
    auto f1 = m_hamiltonian;
    m_hamiltonian = [f1, other, comp](double time)
		    {return comp(f1(time), other(time));};
    return *this;
  }

  /*
    Explicitly overlodaed, so the lambda capture does not have to copy
    unnecessary data.
   */
  template<typename comp_func>
  this_t & compose_with(const TimeIndependentHamiltonian<MatrixType> & other,
			const comp_func & comp) {
    auto f1 = m_hamiltonian;
    MatrixType val = other(0.0);
    m_hamiltonian = [f1, val, comp](double time)
		    -> MatrixType {return comp(f1(time), val);};
    return *this;
  }

  template<typename HamiltonianType>
  this_t & operator+=(const HamiltonianType & other) {
    return compose_with(other, [](const MatrixType & A, const MatrixType & B)
			       {return A + B;});
  }

  template<typename HamiltonianType>
  this_t & operator-=(const HamiltonianType & other) {
    return compose_with(other, [](const MatrixType & A, const MatrixType & B)
			       {return A - B;});
  }

  template<typename HamiltonianType>
  this_t & operator*=(const HamiltonianType & other) {
    return compose_with(other, [](const MatrixType & A, const MatrixType & B)
			       {return A * B;});
  }

  this_t operator-() {
    auto f1 = m_hamiltonian;
    auto new_ham = [f1](double time){return - f1(time);};
    return this_t(new_ham);
  }

  this_t operator+() {
    return *this;
  }

  bool is_time_dependent() const override {return true;}

  void add(const MatrixType & mat) override {
    (*this) += TimeIndependentHamiltonian<MatrixType>(mat);
  }

  void tensor(const MatrixType & mat, bool this_rhs = false) override {
    auto f1 = m_hamiltonian;
    if(this_rhs) {
      m_hamiltonian = [f1, mat](double time)
		      {return KroneckerProduct(mat, f1(time));};
      m_dimension *= mat.rows();
    } else {
      m_hamiltonian = [f1, mat](double time)
		      {return KroneckerProduct(f1(time), mat);};
      m_dimension *= mat.rows();
    }
  }

  int dimension() const override {
    return m_dimension;
  }
  
private:
  std::function<MatrixType(double)> m_hamiltonian;
  int m_dimension;
  
protected:

  TimeDependentHamiltonian* clone_impl()
    const override {return new TimeDependentHamiltonian(*this);};
};

template<typename MatrixType>
TimeDependentHamiltonian<MatrixType>
tensor(TimeDependentHamiltonian<MatrixType> lhs,
       const TimeDependentHamiltonian<MatrixType> & rhs) {
  return lhs.compose_with(rhs, [] (const MatrixType & A, const MatrixType & B)
			       {return KroneckerProduct(A, B);});
}

template<typename MatrixType>
TimeDependentHamiltonian<MatrixType>
tensor(TimeDependentHamiltonian<MatrixType> lhs,
       const TimeIndependentHamiltonian<MatrixType> & rhs) {
  return lhs.compose_with(rhs, [] (const MatrixType & A, const MatrixType & B)
			       {return KroneckerProduct(A, B);});
}

template<typename MatrixType>
TimeDependentHamiltonian<MatrixType>
tensor(const TimeIndependentHamiltonian<MatrixType> & lhs,
       TimeDependentHamiltonian<MatrixType> rhs) {
  return rhs.compose_with(rhs, [] (const MatrixType & A, const MatrixType & B)
			       {return KroneckerProduct(B, A);});
}

template<typename MatrixType>
TimeIndependentHamiltonian<MatrixType>
tensor(const TimeIndependentHamiltonian<MatrixType> & lhs,
       const TimeIndependentHamiltonian<MatrixType> & rhs) {
  return TimeIndependentHamiltonian<MatrixType>
    (KroneckerProduct(lhs(0.0), rhs(0.0)));
}

// Operator+
template<typename MatrixType>
TimeIndependentHamiltonian<MatrixType>
operator+(TimeIndependentHamiltonian<MatrixType> lhs,
	  const TimeIndependentHamiltonian<MatrixType> & rhs) {
  lhs += rhs;
  return lhs;
}

template<typename MatrixType>
TimeDependentHamiltonian<MatrixType>
operator+(TimeDependentHamiltonian<MatrixType> lhs,
	  const TimeIndependentHamiltonian<MatrixType> & rhs) {
  lhs += rhs;
  return lhs;
}

template<typename MatrixType>
TimeDependentHamiltonian<MatrixType>
operator+(const TimeIndependentHamiltonian<MatrixType> & lhs,
	  const TimeDependentHamiltonian<MatrixType> & rhs) {
  TimeDependentHamiltonian<MatrixType> out(lhs);
  lhs += rhs;
  return lhs;
}

template<typename MatrixType>
TimeDependentHamiltonian<MatrixType>
operator+(TimeDependentHamiltonian<MatrixType> lhs,
	  const TimeDependentHamiltonian<MatrixType> & rhs) {
  lhs += rhs;
  return lhs;
}

// Operator-
template<typename MatrixType>
TimeIndependentHamiltonian<MatrixType>
operator-(TimeIndependentHamiltonian<MatrixType> lhs,
	  const TimeIndependentHamiltonian<MatrixType> & rhs) {
  lhs -= rhs;
  return lhs;
}

template<typename MatrixType>
TimeDependentHamiltonian<MatrixType>
operator-(TimeDependentHamiltonian<MatrixType> lhs,
	  const TimeIndependentHamiltonian<MatrixType> & rhs) {
  lhs -= rhs;
  return lhs;
}

template<typename MatrixType>
TimeDependentHamiltonian<MatrixType>
operator-(const TimeIndependentHamiltonian<MatrixType> & lhs,
	  const TimeDependentHamiltonian<MatrixType> & rhs) {
  TimeDependentHamiltonian<MatrixType> out(lhs);
  lhs -= rhs;
  return lhs;
}

template<typename MatrixType>
TimeDependentHamiltonian<MatrixType>
operator-(TimeDependentHamiltonian<MatrixType> lhs,
	  const TimeDependentHamiltonian<MatrixType> & rhs) {
  lhs -= rhs;
  return lhs;
}

// Operator*
template<typename MatrixType>
TimeIndependentHamiltonian<MatrixType>
operator*(TimeIndependentHamiltonian<MatrixType> lhs,
	  const TimeIndependentHamiltonian<MatrixType> & rhs) {
  lhs *= rhs;
  return lhs;
}

template<typename MatrixType>
TimeDependentHamiltonian<MatrixType>
operator*(TimeDependentHamiltonian<MatrixType> lhs,
	  const TimeIndependentHamiltonian<MatrixType> & rhs) {
  lhs *= rhs;
  return lhs;
}

template<typename MatrixType>
TimeDependentHamiltonian<MatrixType>
operator*(const TimeIndependentHamiltonian<MatrixType> & lhs,
	  const TimeDependentHamiltonian<MatrixType> & rhs) {
  TimeDependentHamiltonian<MatrixType> out(lhs);
  lhs *= rhs;
  return lhs;
}

template<typename MatrixType>
TimeDependentHamiltonian<MatrixType>
operator*(TimeDependentHamiltonian<MatrixType> lhs,
	  const TimeDependentHamiltonian<MatrixType> & rhs) {
  lhs *= rhs;
  return lhs;
}

template<typename MatrixType>
TimeIndependentHamiltonian<MatrixType> IdentityHamiltonian(int dimension) {
  MatrixType mat = MatrixType::Identity(dimension, dimension);
  return TimeIndependentHamiltonian<MatrixType>(mat);
}

#endif /* HAMILTONAIN_HPP */
