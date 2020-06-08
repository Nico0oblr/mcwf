#include "Lindbladian.hpp"
#include "Operators.hpp"

calc_mat_t lindblad_term(const std::vector<calc_mat_t> & lindblad_operators,
			 const std::vector<scalar_t> & lindblad_amplitudes) {
  calc_mat_t nh_term;
  if (lindblad_operators.size() > 0) {
    nh_term = lindblad_operators[0].adjoint() * lindblad_operators[0]
      * lindblad_amplitudes[0];
    for (size_type i = 1; i < lindblad_operators.size(); ++i) {
      nh_term += lindblad_operators[i].adjoint() * lindblad_operators[i]
	* lindblad_amplitudes[i];
    }
  }
  return nh_term;
}

lo_ptr lindblad_term(const std::vector<lo_ptr> & lindblad_operators,
		     const std::vector<scalar_t> & lindblad_amplitudes) {
  lo_ptr nh_term = lindblad_operators[0]->adjoint()
    * lindblad_operators[0]
    * lindblad_amplitudes[0];
  for (size_type i = 1; i < lindblad_operators.size(); ++i) {
    nh_term = nh_term
      + lindblad_operators[i]->adjoint()
      * lindblad_operators[i]
      * lindblad_amplitudes[i];
  }
  return nh_term;
}


scalar_t bose_distribution(double temperature,
			   double frequency) {
  return 1.0 / ( std::exp(temperature * frequency) - 1.0);
}

Lindbladian thermalCavity(double n_b,
			  double frequency,
			  scalar_t gamma,
			  int dimension) {
  calc_mat_t A = annihilationOperator_sp(dimension);
  calc_mat_t A_t = creationOperator_sp(dimension);
  calc_mat_t n = numberOperator_sp(dimension);
  std::vector<scalar_t> ampl{gamma * (1.0 + n_b), gamma * n_b};
  return Lindbladian(TimeIndependentHamiltonian<calc_mat_t>(frequency * n),
		     {A, A_t}, ampl);
}

Lindbladian drivenCavity(double n_b,
			 double frequency,
			 double laser_frequency,
			 scalar_t gamma,
			 std::complex<double> amplitude,
			 int dimension,
			 int elec_dim) {
  std::cout << "ampltiude: " << amplitude << std::endl;
  auto A = kroneckerOperator_IDRHS(annihilationOperator_sp(dimension), elec_dim);
  auto A_t = kroneckerOperator_IDRHS(creationOperator_sp(dimension),
				     elec_dim);
  std::vector<lo_ptr> lindblad_op;
  lindblad_op.push_back(std::move(A));
  lindblad_op.push_back(std::move(A_t));
  std::vector<scalar_t> ampl{gamma * (1.0 + n_b), gamma * n_b};
  return Lindbladian(TimeDependentHamiltonian<calc_mat_t>
		     (DrivenCavityHamiltonian(frequency, laser_frequency,
					      amplitude, dimension, elec_dim),
		      dimension * elec_dim), lindblad_op, ampl);
}

Lindbladian::Lindbladian(const Hamiltonian<calc_mat_t> & system_hamiltonian,
			 const std::vector<calc_mat_t> & lindblad_operators,
			 const std::vector<scalar_t> & lindblad_amplitudes)
  :m_system_hamiltonian(system_hamiltonian.clone()),
   m_lindblad_operators(),
   m_lindblad_amplitudes(lindblad_amplitudes) {
  assert(lindblad_operators.size() == lindblad_amplitudes.size());
  for (const auto & lb : lindblad_operators)
    m_lindblad_operators.push_back(operatorize(lb));
}

Lindbladian::Lindbladian(const Hamiltonian<calc_mat_t> & system_hamiltonian,
			 const std::vector<lo_ptr> & lindblad_operators,
			 const std::vector<scalar_t> & lindblad_amplitudes)
  :m_system_hamiltonian(system_hamiltonian.clone()),
   m_lindblad_operators(),
   m_lindblad_amplitudes(lindblad_amplitudes) {
  assert(lindblad_operators.size() == lindblad_amplitudes.size());

  for (const auto & lb : lindblad_operators)
    m_lindblad_operators.push_back(lb->clone());
}

Lindbladian::Lindbladian(const Lindbladian & other)
  :m_system_hamiltonian(other.m_system_hamiltonian->clone()),
   m_lindblad_operators(),
   m_lindblad_amplitudes(other.m_lindblad_amplitudes) {
  for (const auto & lb : other.m_lindblad_operators)
    m_lindblad_operators.push_back(lb->clone());
}


std::unique_ptr<Hamiltonian<calc_mat_t>> Lindbladian::hamiltonian() const {
  std::unique_ptr<Hamiltonian<calc_mat_t>> hamiltonian_copy =
    m_system_hamiltonian->clone();
  hamiltonian_copy->add(- 0.5i * lindblad_term(m_lindblad_operators,
					       m_lindblad_amplitudes));
  return hamiltonian_copy;
}

Hamiltonian<calc_mat_t> & Lindbladian::system_hamiltonian() {
  return *m_system_hamiltonian;
}

void Lindbladian::add_subsystem(const calc_mat_t sub_hamiltonian) {
  int sub_dim = sub_hamiltonian.cols() / m_system_hamiltonian->dimension();
  for (size_type i = 0; i < m_lindblad_operators.size(); ++i) {
    m_lindblad_operators[i] = kroneckerOperator_IDRHS(m_lindblad_operators[i]->eval(),
						      sub_dim);
  }
  m_system_hamiltonian->tensor(calc_mat_t::Identity(sub_dim, sub_dim));
  m_system_hamiltonian->add(sub_hamiltonian);
}

calc_mat_t Lindbladian::operator()(double time, const calc_mat_t & density_matrix) const {
  calc_mat_t out = - 1.0i * (*(*m_system_hamiltonian)(time) * density_matrix
			     - density_matrix * (*(*m_system_hamiltonian)(time)));
  for (size_type i = 0; i < m_lindblad_operators.size(); ++i) {
    if (std::abs(m_lindblad_amplitudes[i]) < tol) continue;
    auto adj_op = m_lindblad_operators[i]->adjoint();
    auto num = 0.5 * adj_op * m_lindblad_operators[i];
    out += m_lindblad_amplitudes[i]
      * ((*m_lindblad_operators[i]) * density_matrix * (*adj_op)
	 - (*num) * density_matrix
	 - density_matrix * (*num));
  }
  return out;
}

std::unique_ptr<Hamiltonian<calc_mat_t>> Lindbladian::superoperator() const {
  int dimension = m_system_hamiltonian->dimension();
  
  struct SuperOperatorStruct {

    auto operator()(double time) const {
      calc_mat_t lhs = superoperator_left((*system_hamiltonian)(time)->eval(),
					  dimension);
      calc_mat_t rhs = superoperator_right((*system_hamiltonian)(time)->eval(),
					   dimension);
      calc_mat_t out = - 1.0i * (lhs - rhs);
      for (size_type i = 0; i < lindblad_operators.size(); ++i) {
	calc_mat_t adj_op = lindblad_operators[i]->adjoint()->eval();
	calc_mat_t mat1 = superoperator_left(lindblad_operators[i]->eval(),
					  dimension);
	calc_mat_t mat2 = superoperator_right(adj_op, dimension);
	calc_mat_t mat3 = superoperator_left(adj_op
					     * lindblad_operators[i]->eval(),
					  dimension);
	calc_mat_t mat4 = superoperator_right(adj_op
					      * lindblad_operators[i]->eval(),
					   dimension);
	out += lindblad_amplitudes[i] * (mat1 * mat2 - 0.5 * mat3 - 0.5 * mat4);
      }
      out = 1.0i * out;
      return operatorize(out);
    }
    
    std::unique_ptr<Hamiltonian<calc_mat_t>> system_hamiltonian;
    std::vector<lo_ptr> lindblad_operators;
    std::vector<scalar_t> lindblad_amplitudes;
    int dimension;

    SuperOperatorStruct(const SuperOperatorStruct & other)
      : system_hamiltonian(other.system_hamiltonian->clone()),
	lindblad_operators(),
	lindblad_amplitudes(other.lindblad_amplitudes),
	dimension(other.dimension) {
      for (const auto & lb : other.lindblad_operators)
	lindblad_operators.push_back(lb->clone());
    }

    SuperOperatorStruct(const std::unique_ptr<Hamiltonian<calc_mat_t>> & csystem_hamiltonian,
			const std::vector<lo_ptr> & clindblad_operators,
			const std::vector<scalar_t> & clindblad_amplitudes,
			int cdimension)
      : system_hamiltonian(csystem_hamiltonian->clone()),
	lindblad_operators(),
	lindblad_amplitudes(clindblad_amplitudes),
	dimension(cdimension) {
      for (const auto & lb : clindblad_operators)
	lindblad_operators.push_back(lb->clone());
    }
  };

  SuperOperatorStruct superoperator_struct(m_system_hamiltonian,
					   m_lindblad_operators,
					   m_lindblad_amplitudes,
					   dimension);
  
  if (m_system_hamiltonian->is_time_dependent()) {
    return TimeDependentHamiltonian<calc_mat_t>(superoperator_struct,
						dimension * dimension).clone();
  } else {
    return TimeIndependentHamiltonian<calc_mat_t>
      (superoperator_struct(0.0)).clone();
  }
}

Lindbladian::Lindbladian(const Hamiltonian<calc_mat_t> & system_hamiltonian,
			 const std::vector<calc_mat_t> & /*lindblad_operators*/,
			 const Eigen::MatrixXd & /*lindblad_matrix*/)
  :m_system_hamiltonian(system_hamiltonian.clone()) {
  assert(false);
  // assert((lindblad_matrix-lindblad_matrix.adjoint()).norm() < tol);
  /*Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(lindblad_matrix);
  mat_t U_adj = solver.eigenvectors().adjoint();
  Eigen::VectorXd gamma = solver.eigenvalues();
  for (int i = 0; i < gamma.size(); ++i) {
    m_lindblad_operators.push_back(U_adj(0, i) * lindblad_operators[0]);
    m_lindblad_amplitudes.push_back(gamma[i]);
    for (int j = 1; j < gamma.size(); ++j) {
      m_lindblad_operators[i] += U_adj(j, i) * lindblad_operators[j];
    }
  }
  std::cout << gamma << std::endl;*/
}


Lindbladian::Lindbladian(const Hamiltonian<calc_mat_t> & system_hamiltonian)
  :m_system_hamiltonian(system_hamiltonian.clone()) {}

Lindbladian::~Lindbladian() {
  m_system_hamiltonian.reset();
  m_lindblad_operators.resize(0);
  m_lindblad_amplitudes.resize(0);
}
