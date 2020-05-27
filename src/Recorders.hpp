#ifndef RECORDERS_HPP
#define RECORDERS_HPP

#include "Common.hpp"
#include <fstream>

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
double evaluate_impl(const vec_t & state, const calc_mat_t & observable);
double evaluate_impl(const calc_mat_t & density_matrix,
		     const calc_mat_t & observable);

/*Adds the recording of a vector of observables*/
template<typename Base>
class ObservableVectorMixin : public Base {
public:
  virtual void record(const typename Base::InformationType & info) override {
    Base::record(info);
    assert(m_observables.size() == m_records.size());
    for (size_type i = 0; i < m_observables.size(); ++i) {
      m_records[i].push_back(evaluate_impl(info, m_observables[i]));
    }
  }

  Eigen::VectorXd expval(int i) const {
    return Eigen::Map<const Eigen::VectorXd>(m_records.at(i).data(),
					     m_records.at(i).size());
  }

  size_type size() const {
    return m_records.size();
  }
  
  ObservableVectorMixin(const std::vector<calc_mat_t> & observables)
    :Base(), m_observables(observables), m_records(m_observables.size()) {}
    
protected:
  std::vector<calc_mat_t> m_observables;
  std::vector<std::vector<double>> m_records;
};

/*
  Abstaction for density matrix construction
*/
calc_mat_t density_impl(const calc_mat_t & density_matrix);
calc_mat_t density_impl(const vec_t & state);

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

  const std::vector<calc_mat_t> & density_matrices() const {
    return m_density_matrices;
  }

  DirectDensityObserverMixin()
    : Base(), m_density_matrices() {}
  
private:
  std::vector<calc_mat_t> m_density_matrices;
};

/* Defines an expectation value writer for Observable vector */
template<typename Base>
class ExpvalWriterMixin : public Base {
public:
  using Base::Base;

  void write(std::ostream & os) const {
    Eigen::IOFormat fmt(Eigen::StreamPrecision,
			Eigen::DontAlignCols, ", ", ", ", "", "", "", "");
    for (size_type i = 0; i < Base::size(); ++i) {
      os << Base::expval(i) << std::endl;
    }
  }

  void write(std::string filename) const {
    std::ofstream output(filename);
    write(output);
    output.close();
  }
};

/*Mixes capabilities for MCWF run into the recorder*/
class MCWFRecorder {
public:
  virtual void record(const vec_t & /*state*/,
		      size_type /*run_index*/,
		      size_type /*time_step*/) {}

  size_type n_runs() const {
    return m_runs;
  }

  MCWFRecorder(size_type runs)
    :m_runs(runs) {}

private:
  size_type m_runs;
};

/*Mixes the capability to properly evaluate distributions, 
  expectation values etc into the Recorder*/
template<typename Base>
class MCWFObservableVectorMixin : public Base {
public:

  virtual void record(const vec_t & state,
		      size_type run_index,
		      size_type time_step) override {
    Base::record(state, run_index, time_step);
    assert(m_observables.size() == m_records.size());
    for (size_type i = 0; i < m_observables.size(); ++i) {
      m_records[i][run_index].push_back(evaluate_impl(state, m_observables[i]));
    }
  }
  
  Eigen::VectorXd expval(int i) const {
    return distribution(i).colwise().mean();
  }

  Eigen::MatrixXd distribution(int index) const {
    assert(m_records.at(index).size() > 0 && "There is no data to distribute");
    size_type data_size = m_records.at(index).size();
    size_type time_steps = m_records.at(index).at(0).size();
    Eigen::MatrixXd mat(Base::n_runs(), time_steps);
    for (size_type i = 0; i < Base::n_runs(); ++i) {
      for (size_type j = 0; j < time_steps; ++j) {
	mat(i, j) = m_records[index][i][j];
      }
    }

    return mat;
  }

  Eigen::VectorXd Var2(int i) const {
    return expval(i) - distribution(i).array().square().matrix().colwise().mean();
  }

  size_type size() const {
    return m_records.size();
  }
  
  MCWFObservableVectorMixin(const std::vector<calc_mat_t> & observables,
			    int runs)
    :Base(runs), m_observables(observables), m_records(observables.size()) {
    for (size_type i = 0; i < observables.size(); ++i){
      m_records[i].resize(runs);
    }
  }

protected:
  std::vector<calc_mat_t> m_observables;
  std::vector<std::vector<std::vector<double>>> m_records;
};

/*Mixes a running average of the density matrix into a MCWF recorder*/
template<typename Base>
class MCWFDensityObserverMixin : public Base {
public:

  virtual void record(const vec_t & state,
		      size_type run_index,
		      size_type time_step) {
    Base::record(state, run_index, time_step);
#pragma omp critical
    if (m_running_average.size() < time_step + 1) {
      m_running_average.resize(time_step + 1);
    }

    calc_mat_t dmat = state * state.adjoint();
#pragma omp critical
    if (m_running_average.at(time_step).size() == 0) {
      m_running_average.at(time_step) = dmat;
    } else {
      m_running_average.at(time_step) += dmat;
    }
  }

  std::vector<calc_mat_t> density_matrices() const {
    std::vector<calc_mat_t> average(m_running_average.size());
    for (size_type i = 0; i < m_running_average.size(); ++i) {
      average[i] = m_running_average[i]
	/ static_cast<double>(Base::n_runs());
    }
    return average;
  }

  MCWFDensityObserverMixin(int runs)
    : Base(runs), running_index(0), m_running_average() {}
  
private:
  int running_index;
  std::vector<calc_mat_t> m_running_average;
};

using MCWFObservableRecorder = MCWFObservableVectorMixin<MCWFRecorder>;
using MCWFDmatRecorder = MCWFDensityObserverMixin<MCWFRecorder>;

using DirectStateRecorder = DirectDensityObserverMixin<RecorderHost<vec_t>>;
using DirectDmatRecorder = DirectDensityObserverMixin<RecorderHost<calc_mat_t>>;

using StateObservableRecorder = ObservableVectorMixin<RecorderHost<vec_t>>;
using DmatObservableRecorder = ObservableVectorMixin<RecorderHost<calc_mat_t>>;

#endif /* RECORDERS_HPP */
