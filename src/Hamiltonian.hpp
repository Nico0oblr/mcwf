#ifndef HAMILTONAIN_HPP
#define HAMILTONAIN_HPP

#include "Common.hpp"
#include "Operators.hpp"
#include <memory>
#include "PadeExponential.hpp"
#include "LinearOperator.hpp"
#include "MatrixExpApply.hpp"
#include "ArnoldiIteration.hpp"

template<typename Hfunc>
auto fourth_order_timeordered_exponential(const Hfunc & H, double t, double dt) {
  static double c1 = (3.0 - 2.0 * std::sqrt(3.0)) / 12.0;
  static double c2 = (3.0 + 2.0 * std::sqrt(3.0)) / 12.0;
  auto H1 = H(t + (0.5 - std::sqrt(3.0) / 6.0) * dt)->eval();
  auto H2 = H(t + (0.5 + std::sqrt(3.0) / 6.0) * dt)->eval();
  auto fac1 = expm(-1.0i * (c1 * H1 + c2 * H2) * dt);
  auto fac2 = expm(-1.0i * (c2 * H1 + c1 * H2) * dt);
  return fac1 * fac2;
}

template<typename Hfunc>
vec_t fourth_order_timeordered_exponential(const Hfunc & H, double t,
					   double dt, const vec_t & state) {
  static double c1 = (3.0 - 2.0 * std::sqrt(3.0)) / 12.0;
  static double c2 = (3.0 + 2.0 * std::sqrt(3.0)) / 12.0;
  auto H1 = H(t + (0.5 - std::sqrt(3.0) / 6.0) * dt);
  auto H2 = H(t + (0.5 + std::sqrt(3.0) / 6.0) * dt);
  auto fac1 = scale_and_add(*H1, *H2, -1.0i * dt * c1, -1.0i * dt * c2);
  auto fac2 = scale_and_add(*H1, *H2, -1.0i * dt * c2, -1.0i * dt * c1);
  int krylov_dim = std::min(20, static_cast<int>(state.size() / 2));
  vec_t apply_fac2 = exp_krylov_apply(*fac2, state, krylov_dim);
  vec_t apply_fac1 = exp_krylov_apply(*fac1, apply_fac2, krylov_dim);
  return apply_fac1;
}

template<typename Hfunc>
vec_t fourth_order_timeordered_exponential(const Hfunc & H,
					   double dt, const vec_t & state) {
  int krylov_dim = std::min(20, static_cast<int>(state.size() / 2));
  auto exponent = -1.0i * dt * (*H);
  return exp_krylov_apply(*exponent, state, krylov_dim);
}

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
  using lo_ptr = std::unique_ptr<LinearOperator<MatrixType>>;
  
  virtual lo_ptr operator()(double time) const = 0;
  virtual vec_t propagate(double time, double dt, const vec_t & state) = 0;
  virtual MatrixType propagator(double time, double dt) = 0;
  virtual bool is_time_dependent() const = 0;
  virtual int dimension() const = 0;
  virtual void add(const MatrixType & mat) = 0;
  void add(const lo_ptr & mat) {add(*mat);}
  virtual void add(const LinearOperator<MatrixType> & mat) = 0;
  virtual void tensor(const MatrixType & mat, bool this_rhs = false) = 0;
  virtual void doubleMe() = 0;
  virtual ~Hamiltonian() = 0;

  auto clone() const { return std::unique_ptr<Hamiltonian>(clone_impl()); }
protected:
  
  virtual Hamiltonian* clone_impl() const = 0;
};

template<typename MatrixType>
Hamiltonian<MatrixType>::~Hamiltonian() {}

template<typename _MatrixType>
class TimeIndependentHamiltonian :
  public Hamiltonian<_MatrixType> {
public:
  using this_t = TimeIndependentHamiltonian<_MatrixType>;
  using MatrixType = _MatrixType;
  using lo_ptr = std::unique_ptr<LinearOperator<MatrixType>>;
  
  lo_ptr operator()(double /*time*/) const override
  {return m_hamiltonian->clone();}

  vec_t propagate(double time, double dt, const vec_t & state) override
  {
    return fourth_order_timeordered_exponential(m_hamiltonian, dt, state);
    // return propagator(time, dt) * state;
  }

  MatrixType propagator(double /*time*/, double dt) override {
    if ((m_propagator.size() == m_hamiltonian->size())
	|| (std::abs(m_last_dt - dt) > tol)) {
      m_propagator = expm(-1.0i * m_hamiltonian->eval() * dt);
    }
    return m_propagator;
  }

  TimeIndependentHamiltonian(const MatrixType & hamiltonian)
    :m_hamiltonian(operatorize(hamiltonian)), m_propagator(), m_last_dt(0.0) {}

  TimeIndependentHamiltonian(const MatrixType & hamiltonian,
			     const MatrixType & propagator,
			     double last_dt)
    :m_hamiltonian(operatorize(hamiltonian)),
     m_propagator(propagator), m_last_dt(last_dt) {}

  TimeIndependentHamiltonian(const lo_ptr & hamiltonian)
    :m_hamiltonian(hamiltonian->clone()), m_propagator(), m_last_dt(0.0) {}

  TimeIndependentHamiltonian(const LinearOperator<MatrixType> & hamiltonian)
    :m_hamiltonian(hamiltonian.clone()), m_propagator(), m_last_dt(0.0) {}

  template<typename MatrixType2>
  explicit operator TimeIndependentHamiltonian<MatrixType2>() {
    TimeIndependentHamiltonian<MatrixType2> out(MatrixType2(m_hamiltonian),
						MatrixType2(m_propagator),
						m_last_dt);
    return out;
  }

  bool is_time_dependent() const override {return false;}

  virtual void doubleMe() override
  {m_hamiltonian = doubleOperator(m_hamiltonian);}
  
  void add(const LinearOperator<MatrixType> & mat) override {
    reset();
    m_hamiltonian = m_hamiltonian + mat;
  }

  void add(const MatrixType & mat) override {
    reset();
    m_hamiltonian = m_hamiltonian + operatorize(mat);
  }

  void tensor(const MatrixType & mat, bool this_rhs = false) override {
    reset();
    if(this_rhs) {
      m_hamiltonian = kroneckerOperator(mat, m_hamiltonian->eval());
    } else {
      m_hamiltonian = kroneckerOperator(m_hamiltonian->eval(), mat);
    }
  }

  int dimension() const override {
    assert(m_hamiltonian->rows() == m_hamiltonian->cols());
    return m_hamiltonian->rows();
  }

  TimeIndependentHamiltonian(const TimeIndependentHamiltonian & other)
    : m_hamiltonian(other.m_hamiltonian->clone()),
      m_propagator(other.m_propagator),
      m_last_dt(other.m_last_dt) {}

  ~TimeIndependentHamiltonian() override {
    m_hamiltonian.reset();
    m_propagator.resize(0, 0);
  }
  
private:

  void reset() {
    // m_propagator.resize(0, 0);
    m_last_dt = 0.0;
  }
  
  lo_ptr m_hamiltonian;
  MatrixType m_propagator;
  double m_last_dt;

protected:

  TimeIndependentHamiltonian* clone_impl()
    const override {return new TimeIndependentHamiltonian(*this);}
};


template<typename _MatrixType>
class TimeDependentHamiltonian :
  public Hamiltonian<_MatrixType> {
public:
  using MatrixType = _MatrixType;
  using this_t = TimeDependentHamiltonian<MatrixType>;
  using lo_ptr = std::unique_ptr<LinearOperator<MatrixType>>;
  using ham_func = std::function<lo_ptr(double)>;

  lo_ptr operator()(double time) const override {
    return m_hamiltonian(time);
  }

  vec_t propagate(double time, double dt,
		  const vec_t & state) override {
    return fourth_order_timeordered_exponential(m_hamiltonian, time, dt, state);
  }

  MatrixType propagator(double time, double dt) override {
    return fourth_order_timeordered_exponential(m_hamiltonian, time, dt);
  }

  TimeDependentHamiltonian(const std::function<MatrixType(double)> & hamiltonian,
			   int dimension)
    : m_hamiltonian(), m_dimension(dimension) {
    m_hamiltonian = [hamiltonian](double time)
		    {return operatorize(hamiltonian(time));};
  }

  TimeDependentHamiltonian(const ham_func & hamiltonian,
			     int dimension)
    :m_hamiltonian(hamiltonian), m_dimension(dimension) {}

  bool is_time_dependent() const override {return true;}

  virtual void doubleMe() override {
    auto f1 = m_hamiltonian;
    m_hamiltonian = [f1](double t) {return doubleOperator(f1(t));};
  }

  void add(const MatrixType & mat) override {
    auto f1 = m_hamiltonian;
    m_hamiltonian = [f1, mat](double t) {return f1(t) + operatorize(mat);};
  }
  
  void add(const LinearOperator<MatrixType> & mat) override {

    struct TEMP {
      TEMP(const TEMP & other)
	:f(other.f), m(other.m->clone()) {}

      TEMP(const ham_func & cf, const LinearOperator<MatrixType> & cm)
	: f(cf), m(cm.clone()) {}

      lo_ptr operator()(double t) const {return f(t) + m;}      

      ham_func f;
      lo_ptr m;
    };
    
    m_hamiltonian = TEMP(m_hamiltonian, mat);
  }

  void tensor(const MatrixType & mat, bool this_rhs = false) override {
    auto f1 = m_hamiltonian;
    if(this_rhs) {
      m_hamiltonian = [f1, mat](double time)
		      {return kroneckerOperator(mat, f1(time)->eval());};
      m_dimension *= mat.rows();
    } else {
      m_hamiltonian = [f1, mat](double time)
		      {return kroneckerOperator(f1(time)->eval(), mat);};
      m_dimension *= mat.rows();
    }
  }

  int dimension() const override {
    return m_dimension;
  }

  ~TimeDependentHamiltonian() override = default;
  
private:
  ham_func m_hamiltonian;
  int m_dimension;
  
protected:

  TimeDependentHamiltonian* clone_impl()
    const override {return new TimeDependentHamiltonian(*this);}
};
#endif /* HAMILTONAIN_HPP */
