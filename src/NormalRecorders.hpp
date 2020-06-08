#ifndef NORMALRECORDERS_HPP
#define NORMALRECORDERS_HPP

/*
  Mixin host for all Recorders
*/
template<typename _InformationType>
class RecorderHost {
public:
  using InformationType = _InformationType;
  virtual void record(const InformationType & /*state*/) {}
  RecorderHost() = default;
  virtual ~RecorderHost() = default;
  
protected:
  RecorderHost(const RecorderHost &) = default;
  RecorderHost& operator=(const RecorderHost &) = default;
};

/*
  Abstraction of <observable>.
*/
double evaluate_impl(const vec_t & state,
		     const LinearOperator<calc_mat_t> & observable);
double evaluate_impl(const calc_mat_t & density_matrix,
		     const LinearOperator<calc_mat_t> & observable);

/*Adds the recording of a vector of observables*/
template<typename Base>
class ObservableVectorMixin : public Base {
public:
  virtual void record(const typename Base::InformationType & info) override {
    Base::record(info);
    assert(m_observables.size() == m_records.size());
    for (size_type i = 0; i < m_observables.size(); ++i) {
      m_records[i].push_back(evaluate_impl(info, *m_observables[i]));
    }
  }

  Eigen::VectorXd expval(int i) const {
    return Eigen::Map<const Eigen::VectorXd>(m_records.at(i).data(),
					     m_records.at(i).size());
  }

  size_type size() const {
    return m_records.size();
  }

  ObservableVectorMixin<Base>&
  push_back(const LinearOperator<calc_mat_t> & obs) {
    m_observables.push_back(obs.clone());
    m_records.push_back({});
    return *this;
  }

  ObservableVectorMixin()
    :Base(), m_observables(), m_records() {}
  
  ObservableVectorMixin(const std::vector<calc_mat_t> & observables)
    :Base(), m_observables(), m_records(observables.size()) {
    for (const calc_mat_t & mat: observables) {
      m_observables.push_back(BareLinearOperator<calc_mat_t>(mat).clone());
    }
  }

  ObservableVectorMixin(const std::vector<lo_ptr> & observables)
    :Base(), m_observables(), m_records(observables.size()) {
    for (const lo_ptr & mat: observables) {
      m_observables.push_back(mat->clone());
    }
  }

  virtual ~ObservableVectorMixin() {}

  ObservableVectorMixin(const ObservableVectorMixin<Base> & other)
    :Base(), m_observables(), m_records(other.m_records) {
    for (const auto & x : other.m_observables)
      m_observables.push_back(x->clone());
  }
  
  // protected:
  std::vector<lo_ptr> m_observables;
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


  virtual ~DirectDensityObserverMixin() {}
  
private:
  std::vector<calc_mat_t> m_density_matrices;
};

#endif /* NORMALRECORDERS_HPP */
