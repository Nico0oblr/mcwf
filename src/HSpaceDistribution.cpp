#include "HSpaceDistribution.hpp"

vec_t HSpaceDistribution::draw() const {
  return m_states.at(linear_search(m_probabilities));
}

HSpaceDistribution::HSpaceDistribution(const std::vector<double> & probabilities,
		   const std::vector<vec_t> & states)
  :m_probabilities(probabilities),
   m_states(states) {}

HSpaceDistribution::HSpaceDistribution(const std::vector<double> & probabilities,
		   const std::vector<int> & states,
		   int dimension)
  :m_probabilities(probabilities) {
  assert(probabilities.size() == states.size());
  for (size_type i = 0; i < states.size(); ++i) {
    vec_t state = vec_t::Zero(dimension);
    state(states[i]) = 1.0;
    m_states.push_back(state);
  }
}

HSpaceDistribution::HSpaceDistribution(int dimension) {
  for (int i = 0; i < dimension; ++i) {
    m_probabilities.push_back(1.0 / static_cast<double>(dimension));
    vec_t state = vec_t::Zero(dimension);
    state(i) = 1.0;
    m_states.push_back(state);
  }
}

HSpaceDistribution & HSpaceDistribution::operator+=(const HSpaceDistribution & other) {
  std::vector<double> m_probabilities_old = m_probabilities;
  std::vector<vec_t> m_states_old = m_states;
  m_probabilities.clear();
  m_states.clear();
    
  for (size_type i = 0; i < m_states_old.size(); ++i) {
    for (size_type j = 0; j < other.m_states.size(); ++j) {
      m_states.push_back(Eigen::kroneckerProduct(m_states_old[i],
						 other.m_states[j]));
      m_probabilities.push_back(m_probabilities_old[i]
				* other.m_probabilities[j]);
    }
  }

  return *this;
}

calc_mat_t HSpaceDistribution::density_matrix() const {
  int dimension = m_states.back().size();
  calc_mat_t out = calc_mat_t::Zero(dimension, dimension);
  for (size_type i = 0; i < m_states.size(); ++i) {
    out = out + m_probabilities[i] * m_states[i] * m_states[i].adjoint();
  }
  return out;
}

HSpaceDistribution coherent_photon_state(double mean_photons, int dimension) {
  if (mean_photons < tol) return HSpaceDistribution({1.0}, {0}, dimension);
  std::cout << "mean photons: " << mean_photons << std::endl;
  vec_t state = vec_t::Zero(dimension);
  for (int i = 0; i < dimension; ++i) {
    state(i) = poisson(mean_photons, i);
  }
  state /= state.norm();
  std::cout << state.transpose().real() << std::endl;
  return HSpaceDistribution({1.0}, {state});
}
