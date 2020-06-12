#ifndef RECORDERS_HPP
#define RECORDERS_HPP

#include "Common.hpp"
#include "LinearOperator.hpp"
#include <fstream>
#include "NormalRecorders.hpp"

/*
  IDEA: Do not provide total number of runs in advance, but 
  register every run at the recorder, get an ID and store
  results with that ID.
*/


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

  virtual ~ExpvalWriterMixin() {}
};

/*Mixes capabilities for MCWF run into the recorder*/
class MCWFRecorder {
public:
  virtual void record(const vec_t & /*state*/,
		      size_type /*run_index*/,
		      size_type /*time_step*/) {}

  virtual int register_run() {}

  MCWFRecorder() {}
  
  virtual ~MCWFRecorder() = default;
  
protected:
  MCWFRecorder(const MCWFRecorder &) = default;
  MCWFRecorder& operator=(const MCWFRecorder &) = default;
  };

/*Mixes the capability to properly evaluate distributions, 
  expectation values etc into the Recorder*/
template<typename Base>
class MCWFObservableVectorMixin : public Base {
public:

  int register_run() override {
#pragma omp critical
    {
      for (auto & obsvec : m_records) obsvec.push_back({});
    }
    return m_records.back().size() - 1;
  }
  
  virtual void record(const vec_t & state,
		      size_type run_index,
		      size_type time_step) override {
    Base::record(state, run_index, time_step);
    assert(m_observables.size() == m_records.size());

#pragma omp critical
    for (size_type i = 0; i < m_observables.size(); ++i) {
      assert(m_records.at(i).at(run_index).size() == time_step);
      m_records.at(i).at(run_index).push_back(evaluate_impl(state, *m_observables[i]));
    }
  }
  
  Eigen::VectorXd expval(int i) const {
    return distribution(i).colwise().mean();
  }

  Eigen::VectorXd expval_squared(int i) const {
    return distribution(i).array().square().matrix().colwise().mean();
  }

  Eigen::MatrixXd distribution(int index) const {
    assert(m_records.at(index).size() > 0 && "There is no data to distribute");
    size_type data_size = m_records.at(index).size();
    size_type time_steps = m_records.at(index).at(0).size();
    Eigen::MatrixXd mat(m_records[index].size(), time_steps);
    for (size_type i = 0; i < m_records[index].size(); ++i) {
      for (size_type j = 0; j < m_records[index][i].size(); ++j) {
	mat(i, j) = m_records.at(index).at(i).at(j);
      }
    }

    return mat;
  }

  Eigen::VectorXd Var2(int i) const {
    return expval_squared(i)
      - Eigen::VectorXd(expval(i).array().pow(2.0).matrix());
  }

  size_type size() const {
    return m_records.size();
  }

  MCWFObservableVectorMixin<Base>&
  push_back(const LinearOperator<calc_mat_t> & obs) {
    m_observables.push_back(obs.clone());
    m_records.push_back({});
    return *this;
  }

  MCWFObservableVectorMixin(const std::vector<calc_mat_t> & observables)
    :Base(), m_observables(), m_records(observables.size()) {
    for (const calc_mat_t & mat: observables) {
      m_observables.push_back(BareLinearOperator<calc_mat_t>(mat).clone());
    }
  }

  MCWFObservableVectorMixin()
    :Base(), m_observables(), m_records() {}
  
  MCWFObservableVectorMixin(const std::vector<lo_ptr> & observables)
    :Base(), m_observables(), m_records(observables.size()) {
    for (const lo_ptr & mat: observables) {
      m_observables.push_back(mat->clone());
    }
  }

  virtual ~MCWFObservableVectorMixin() override {}
  
  MCWFObservableVectorMixin(const MCWFObservableVectorMixin<Base> & other)
    :Base(), m_observables(), m_records(other.m_records) {
    for (const auto & x : other.m_observables)
      m_observables.push_back(x->clone());
  }

  const std::vector<std::vector<std::vector<double>>> & data() const {
    return m_records;
  }
  
  // protected:
  std::vector<lo_ptr> m_observables;
  std::vector<std::vector<std::vector<double>>> m_records;
};

/*Mixes a running average of the density matrix into a MCWF recorder*/
template<typename Base>
class MCWFDensityObserverMixin : public Base {
public:

  int register_run() override {
#pragma omp critical
    {
      m_running_average.push_back(calc_mat_t());
    }
    return m_running_average.size() - 1;
  }
  
  virtual void record(const vec_t & state,
		      size_type run_index,
		      size_type time_step) {
    Base::record(state, run_index, time_step);
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
	/ static_cast<double>(m_running_average.size());
    }
    return average;
  }

  MCWFDensityObserverMixin()
    : Base(), m_running_average() {}

  virtual ~MCWFDensityObserverMixin() {}
  
private:
  std::vector<calc_mat_t> m_running_average;
};

class CorrelationRecorderMixin :
  public ObservableVectorMixin<RecorderHost<vec_t>> {
public:
  using Base = ObservableVectorMixin<RecorderHost<vec_t>>;
  using Base::Base;
  
  using Base::record;
  virtual void record(const vec_t & lhs, const vec_t & rhs) {
    assert(m_observables.size() == m_records.size());
#pragma omp critical
    for (size_type i = 0; i < m_observables.size(); ++i) {
      m_records.at(i).push_back(lhs.dot(*m_observables[i] * rhs).real());
    }
  }
};

class MCWFCorrelationRecorderMixin :
  public MCWFObservableVectorMixin<MCWFRecorder> {
public:
  using Base = MCWFObservableVectorMixin<MCWFRecorder>;
  using Base::Base;

  using Base::record;
  virtual void record(const vec_t & lhs,
		      const vec_t & rhs,
		      size_type run_index,
		      size_type /*time_step*/) {
    assert(m_observables.size() == m_records.size());
    for (size_type i = 0; i < m_observables.size(); ++i) {
      m_records[i][run_index].push_back(lhs.dot(*m_observables[i] * rhs).real());
    }
  }
};


using MCWFObservableRecorder = MCWFObservableVectorMixin<MCWFRecorder>;
using MCWFDmatRecorder = MCWFDensityObserverMixin<MCWFRecorder>;

using DirectStateRecorder = DirectDensityObserverMixin<RecorderHost<vec_t>>;
using DirectDmatRecorder = DirectDensityObserverMixin<RecorderHost<calc_mat_t>>;

using StateObservableRecorder = ObservableVectorMixin<RecorderHost<vec_t>>;
using DmatObservableRecorder = ObservableVectorMixin<RecorderHost<calc_mat_t>>;

#endif /* RECORDERS_HPP */
