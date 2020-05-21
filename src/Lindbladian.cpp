#include "Lindbladian.hpp"
#include "Operators.hpp"

scalar_t bose_distribution(double temperature,
			   double frequency) {
  return 1.0 / ( std::exp(temperature * frequency) - 1.0);
}

Lindbladian thermalCavity(double n_b,
			  double frequency,
			  scalar_t gamma,
			  int dimension) {
  mat_t A = annihilationOperator(dimension);
  mat_t A_t = creationOperator(dimension);
  mat_t n = numberOperator(dimension);
  std::vector<scalar_t> ampl{gamma * (1.0 + n_b), gamma * n_b};
  return Lindbladian(frequency * n, {A, A_t}, ampl);
}

Lindbladian drivenCavity(double n_b,
			 double frequency,
			 scalar_t gamma,
			 std::complex<double> amplitude,
			 int dimension) {
  std::cout << "ampltiude: " << amplitude << std::endl;
  mat_t A = annihilationOperator(dimension);
  mat_t A_t = creationOperator(dimension);
  mat_t n = numberOperator(dimension);
  std::vector<scalar_t> ampl{gamma * (1.0 + n_b), gamma * n_b};
  std::cout << "amplitude: " << amplitude << std::endl;
  return Lindbladian(frequency * n + 0.5 * amplitude * A
		     + 0.5 * std::conj(amplitude) * A_t,
		     {A, A_t}, ampl);
}

Lindbladian::Lindbladian(const mat_t & system_hamiltonian,
	    const std::vector<mat_t> & lindblad_operators,
	    const std::vector<scalar_t> & lindblad_amplitudes)
  :m_system_hamiltonian(system_hamiltonian),
   m_lindblad_operators(lindblad_operators),
   m_lindblad_amplitudes(lindblad_amplitudes) {
  assert(lindblad_operators.size() == lindblad_amplitudes.size());
  calculate_nh_term();
}

bool Lindbladian::commutes() {
  return (m_nh_term * m_system_hamiltonian
	  - m_system_hamiltonian * m_nh_term).norm() < tol;
}

mat_t Lindbladian::hamiltonian() const {
  return m_system_hamiltonian - 0.5i * m_nh_term;
}

void Lindbladian::add_subsystem(const mat_t sub_hamiltonian) {
  assert(sub_hamiltonian.cols() >= m_system_hamiltonian.cols());
  int sub_dim = sub_hamiltonian.cols() / m_system_hamiltonian.cols();
  for (int i = 0; i < m_lindblad_operators.size(); ++i) {
    m_lindblad_operators[i] = tensor_identity(m_lindblad_operators[i],
					      sub_dim);
  }
  m_system_hamiltonian = tensor_identity(m_system_hamiltonian,
					 sub_dim);
  m_system_hamiltonian += sub_hamiltonian;
  calculate_nh_term();
}

void Lindbladian::calculate_nh_term() {
  if (m_lindblad_operators.size() > 0) {
    m_nh_term = m_lindblad_operators[0].adjoint() * m_lindblad_operators[0]
      * m_lindblad_amplitudes[0];
    for (int i = 1; i < m_lindblad_operators.size(); ++i) {
      m_nh_term += m_lindblad_operators[i].adjoint() * m_lindblad_operators[i]
	* m_lindblad_amplitudes[i];
    }
  }
}

mat_t Lindbladian::operator()(const mat_t & density_matrix) const {
  mat_t out = - 1.0i * (m_system_hamiltonian * density_matrix
			- density_matrix * m_system_hamiltonian);
  for (int i = 0; i < m_lindblad_operators.size(); ++i) {
    mat_t adj_op = m_lindblad_operators[i].adjoint();
    out += m_lindblad_amplitudes[i]
      * (m_lindblad_operators[i] * density_matrix * adj_op
	 - 0.5 * adj_op * m_lindblad_operators[i] * density_matrix
	 - 0.5 * density_matrix * adj_op * m_lindblad_operators[i]);
  }
  return out;
}

spmat_t Lindbladian::superoperator() const {
  int dimension = m_system_hamiltonian.cols();
  spmat_t out = - 1.0i * (superoperator_left(m_system_hamiltonian, dimension).sparseView()
			  - superoperator_right(m_system_hamiltonian, dimension).sparseView());
  for (int i = 0; i < m_lindblad_operators.size(); ++i) {
    mat_t adj_op = m_lindblad_operators[i].adjoint();
    spmat_t mat1 = superoperator_left(m_lindblad_operators[i], dimension).sparseView();
    spmat_t mat2 = superoperator_right(adj_op, dimension).sparseView();
    spmat_t mat3 = superoperator_left(adj_op * m_lindblad_operators[i], dimension).sparseView();
    spmat_t mat4 = superoperator_right(adj_op * m_lindblad_operators[i], dimension).sparseView();
    out += m_lindblad_amplitudes[i] * (mat1 * mat2 - 0.5 * mat3 - 0.5 * mat4);
  }
  return out;
}

Lindbladian::Lindbladian(const mat_t & system_hamiltonian,
			 const std::vector<mat_t> & lindblad_operators,
			 const Eigen::MatrixXd & lindblad_matrix)
  :m_system_hamiltonian(system_hamiltonian) {
  assert((lindblad_matrix-lindblad_matrix.adjoint()).norm() < tol);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(lindblad_matrix);
  mat_t U_adj = solver.eigenvectors().adjoint();
  Eigen::VectorXd gamma = solver.eigenvalues();
  for (int i = 0; i < gamma.size(); ++i) {
    m_lindblad_operators.push_back(U_adj(0, i) * lindblad_operators[0]);
    m_lindblad_amplitudes.push_back(gamma[i]);
    for (int j = 1; j < gamma.size(); ++j) {
      m_lindblad_operators[i] += U_adj(j, i) * lindblad_operators[j];
    }
  }
  calculate_nh_term();
  std::cout << gamma << std::endl;
}
