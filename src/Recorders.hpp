#ifndef RECORDERS_HPP
#define RECORDERS_HPP

#include "Common.hpp"

/*
  Mixin host for all Recorders
*/
template<typename _InformationType>
class RecorderHost {
public:
  using InformationType = _InformationType;
  virtual void record(const InformationType & /*state*/) {}
};

/*
  Abstraction of <observable>.
*/
double evaluate_impl(const vec_t & state, const mat_t & observable) {
  return state.dot(observable * state).real();
}
double evaluate_impl(const mat_t & density_matrix, const mat_t & observable) {
  return (density_matrix * observable).trace().real();
}

/*Mixes capabilities for MCWF run into the recorder*/
template<typename Base>
class MCWFMixin : public Base {
public:
  virtual void record(const vec_t & state) override {Base::record(state);}
  virtual void new_run_impl() {}

  void new_run() {
    new_run_impl();
    ++run_counter;
  }

  int n_runs() const {
    return run_counter;
  }

  MCWFMixin()
    :Base(), run_counter(0) {}

private:
  int run_counter;
};

/*Adds the recording of a vector of observables*/
template<typename Base>
class ObservableVectorMixin : public Base {
public:
  virtual void record(const typename Base::InformationType & info) override {
    Base::record(info);
    assert(m_observables.size() == m_records.size());
    for (int i = 0; i < m_observables.size(); ++i) {
      m_records[i].push_back(evaluate_impl(info, m_observables[i]));
    }
  }

  Eigen::VectorXd expval(int i) const {
    return Eigen::Map<const Eigen::VectorXd>(m_records.at(i).data(),
					     m_records.at(i).size());
  }
  
  ObservableVectorMixin(const std::vector<mat_t> & observables)
    :Base(), m_observables(observables), m_records(m_observables.size()) {}
    
protected:
  std::vector<mat_t> m_observables;
  std::vector<std::vector<double>> m_records;
};

/*Mixes the capability to properly evaluate distributions, 
  expectation values etc into the Recorder*/
template<typename Base>
class MCWFObservableVectorEvaluator : public Base {
public:
  Eigen::VectorXd expval(int i) const {
    return distribution(i).colwise().mean();
  }

  Eigen::MatrixXd distribution(int i) const {
    int data_size = Base::m_observables.at(i).size();
    int time_steps = data_size / Base::n_runs();
    assert(time_steps * Base::n_runs() == data_size);
    return Eigen::Map<const Eigen::MatrixXd>(Base::m_observables.at(i).data(),
					     Base::n_runs(), time_steps);
  }

  Eigen::VectorXd Var2(int i) const {
    return expval(i) - distribution(i).array().square().matrix().colwise().mean();
  }

  MCWFObservableVectorEvaluator(const std::vector<mat_t> & observables)
    :Base(observables) {}
};

/*Mixes a running average of the density matrix into a MCWF recorder*/
template<typename Base>
class MCWFDensityObserverMixin : public Base {
public:

  virtual void record(const vec_t & state) {
    Base::record(state);
    if (running_index < m_running_average.size()) {
      m_running_average.push_back(state * state.adjoint());
    } else {
      int current_runs = Base::n_runs();
      assert(current_runs > 1);
      m_running_average[running_index]
	= (m_running_average[running_index]
	   * (static_cast<double>(Base::n_runs()) - 1.0)
	   + state * state.adjoint()) / static_cast<double>(Base::n_runs());
    }
  }

  virtual void new_run_impl() {
    Base::new_run_impl();
    running_index = 0;
  }

  const std::vector<mat_t> & density_matrices() const {
    return m_running_average;
  }

  MCWFDensityObserverMixin()
    : Base(), running_index(0), m_running_average() {}
  
private:
  int running_index;
  std::vector<mat_t> m_running_average;
};


/*
  Abstaction for density matrix construction
*/
mat_t density_impl(const mat_t & density_matrix) {return density_matrix;}
mat_t density_impl(const vec_t & state) {return state * state.adjoint();}

/*
  Mixes the observation of the density matrix into a
  non-MCWF recorder. You can use it for MCWF runs, but
  you are going to run into memory issues.
*/
template<typename Base>
class DirectDensityObserverMixin : public Base {
public:

  virtual void record(const typename Base::InformationType & info) {
    Base::record(info);
    m_density_matrices.push_back(density_impl(info));
  }

  const std::vector<mat_t> & density_matrices() const {
    return m_density_matrices;
  }

  DirectDensityObserverMixin()
    : Base(), m_density_matrices() {}
  
private:
  std::vector<mat_t> m_density_matrices;
};

using MCWFObservableRecorder = MCWFObservableVectorEvaluator<ObservableVectorMixin<MCWFMixin<RecorderHost<vec_t>>>>;
using MCWFStateRecorder = MCWFDensityObserverMixin<MCWFMixin<RecorderHost<vec_t>>>;
using MCWFStateObservableRecorder = MCWFObservableVectorEvaluator<ObservableVectorMixin<MCWFDensityObserverMixin<MCWFMixin<RecorderHost<vec_t>>>>>;

using DirectStateRecorder = DirectDensityObserverMixin<RecorderHost<vec_t>>;
using DirectDmatRecorder = DirectDensityObserverMixin<RecorderHost<mat_t>>;

#endif /* RECORDERS_HPP */
