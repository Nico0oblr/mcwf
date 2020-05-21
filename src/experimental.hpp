namespace SpinOperators {
  mat_t spin_x(double total_spin) {
    int dimension = static_cast<int>(2 * total_spin) + 1;
    mat_t out = mat_t::Zero(dimension, dimension);

    for (int i = 0; i + 1 < dimension; ++i) {
      double m = i - total_spin;
      out(i, i + 1) = 0.5 * std::sqrt(total_spin * (total_spin + 1)
				      - m * (m + 1));
      out(i + 1, i) = 0.5 * std::sqrt(total_spin * (total_spin + 1)
				      - m * (m + 1));
    }
    return out;
  }

  mat_t spin_y(double total_spin) {
    int dimension = static_cast<int>(2 * total_spin) + 1;
    mat_t out = mat_t::Zero(dimension, dimension);

    for (int i = 0; i + 1 < dimension; ++i) {
      double m = i - total_spin;
      out(i, i + 1) = 0.5i * std::sqrt(total_spin * (total_spin + 1)
					- m * (m + 1));
      out(i + 1, i) = -0.5i * std::sqrt(total_spin * (total_spin + 1)
					- m * (m + 1));
    }
    return out;
  }

  mat_t spin_z(double total_spin) {
    int dimension = static_cast<int>(2 * total_spin) + 1;
    mat_t out = mat_t::Zero(dimension, dimension);

    for (int i = 0; i  < dimension; ++i) {
      out(i, i) = i - total_spin;
    }
    return out;
  }

  mat_t spin_plus(double total_spin) {
    return spin_x(total_spin) + 1.0i * spin_y(total_spin);
  }

  mat_t spin_minus(double total_spin) {
    return spin_x(total_spin) - 1.0i * spin_y(total_spin);
  }
}

Eigen::VectorXd gaussian_position(double R_x,
				  double R_y,
				  double R_z) {
  std::normal_distribution<double> gauss_x{0.0, R_x};
  std::normal_distribution<double> gauss_y{0.0, R_y};
  std::normal_distribution<double> gauss_z{0.0, R_z};
  Eigen::VectorXd pos(3);
  pos << gauss_x(mt_rand), gauss_y(mt_rand), gauss_z(mt_rand);
  return pos;
}

std::vector<Eigen::VectorXd> gaussian_position_vector(double R_x,
						      double R_y,
						      double R_z,
						      int sites) {
  std::vector<Eigen::VectorXd> positions;
  for (int i = 0; i < sites; ++i) {
    positions.push_back(gaussian_position(R_x, R_y, R_z));
  }
  return positions;
}

std::complex<double> dipole_interaction(int alpha, int alpha_pr,
					double gamma, double k,
					const Eigen::VectorXd & dist) {
  assert(alpha < 3);
  assert(alpha_pr < 3);
  double r = dist.norm();
  if (r < tol) {
    if (alpha == alpha_pr) return 1.0i * gamma / 2.0;
    return 0.0;
  }
  
  double kr = k * r;
  std::complex<double> phase = std::exp(1.0i * k * r);
  std::complex<double> A = phase * (-1.0 / kr - 1.0i / (kr * kr) + 1.0 / (kr * kr * kr));
  std::complex<double> B = phase * (1.0 / kr + 3.0i / (kr * kr) - 3.0 / (kr * kr * kr));
  std::complex<double> out = dist[alpha] * dist[alpha_pr] * B / (r * r);
  if (alpha == alpha_pr) out += A;
  return - 3.0 * gamma / 4.0 * out;
}

mat_t transition_operator(int excited_state, int dimension) {
  mat_t out = mat_t::Zero(dimension, dimension);
  out(0, excited_state) = 1.0;
  return out;
}

mat_t transition_operator_t(int excited_state, int dimension) {
  mat_t out = mat_t::Zero(dimension, dimension);
  out(excited_state, 0) = 1.0;
  return out;
}

Lindbladian atom_gas(int sites,
		     double gamma,
		     double transition_wv,
		     const Eigen::VectorXd & extent,
		     const Eigen::VectorXd & detuning,
		     const Eigen::VectorXd & rabi_frequency,
		     const Eigen::VectorXd & laser_vector) {
  assert(detuning.size() == 3);
  assert(rabi_frequency.size() == 3);
  assert(laser_vector.size() == 3);
  assert(extent.size() == 3);
  int dimension = std::pow(4, sites);
  std::vector<mat_t> b_x = operator_vector(transition_operator(1, 4), sites);
  std::vector<mat_t> b_t_x = operator_vector(transition_operator_t(1, 4), sites);
  std::vector<mat_t> b_y = operator_vector(transition_operator(2, 4), sites);
  std::vector<mat_t> b_t_y = operator_vector(transition_operator_t(2, 4), sites);
  std::vector<mat_t> b_z = operator_vector(transition_operator(3, 4), sites);
  std::vector<mat_t> b_t_z = operator_vector(transition_operator_t(3, 4), sites);
  std::vector<Eigen::VectorXd> atom_pos
    = gaussian_position_vector(extent[0], extent[1], extent[2], sites);
  std::vector<std::vector<mat_t>> b{b_x, b_y, b_z};
  std::vector<std::vector<mat_t>> b_t{b_t_x, b_t_y, b_t_z};

  mat_t hamiltonian(dimension, dimension);
  std::vector<mat_t> lindblad_ops;

  // First the diagonal terms
  for (int i = 0; i < sites; ++i) {
    std::complex<double> phase = std::exp(1.0i * atom_pos[i].dot(laser_vector));
    std::complex<double> phase_adj = std::exp(-1.0i * atom_pos[i].dot(laser_vector));
    for (int alpha = 0; alpha < 3; ++alpha) {
      hamiltonian -=  detuning[alpha] * b_t[alpha][i] * b[alpha][i];
      hamiltonian +=  rabi_frequency[alpha] * b_t[alpha][i] * phase;
      hamiltonian +=  rabi_frequency[alpha] * b[alpha][i] * phase_adj;
      lindblad_ops.push_back(b[alpha][i]);
    }
  }


  Eigen::MatrixXd lindblad_matrix = Eigen::MatrixXd::Zero(3 * sites, 3 * sites);
  // Now the interactions
  for (int i = 0; i < sites; ++i) {
    for (int j = 0; j < sites; ++j) {
      for (int alpha = 0; alpha < 3; ++alpha) {
	for (int alpha_pr = 0; alpha_pr < 3; ++alpha_pr) {
	  std::complex<double> G = dipole_interaction(alpha, alpha_pr,
						      gamma, transition_wv,
						      atom_pos[j]-atom_pos[i]);
	  lindblad_matrix(3 * i + alpha, 3 * j + alpha_pr) = 0.5 * std::imag(G);
	  if (i == j) continue;
	  hamiltonian += std::real(G) * b_t[alpha][i] * b[alpha_pr][j];
	}
      }
    }
  }

  std::cout << hamiltonian << std::endl;
  return Lindbladian(hamiltonian, lindblad_ops, lindblad_matrix);
}
