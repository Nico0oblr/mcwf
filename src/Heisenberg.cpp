#include "Heisenberg.hpp"

spmat_t Heisenberg_spin_projector(int sites, double spin) {
  int dimension = std::pow(2, sites);
  spmat_t spinoperator = 0.5 * sum_operator_sp(pauli_z(), sites);
  vec_t spinoperator_d = spinoperator.diagonal();
  
  std::vector<int> projection_basis;
  for (int i = 0; i < dimension; ++i) {
    if (std::abs(spin - spinoperator_d(i).real()) < tol) {
      projection_basis.push_back(i);
    }
  }


  using T = Eigen::Triplet<scalar_t>;
  std::vector<T> tripletList;
  spmat_t projection = mat_t::Zero(projection_basis.size(), dimension);
  for (size_type i = 0; i < projection_basis.size(); ++i) {
    tripletList.push_back(T(i, projection_basis[i], 1.0));
    // projection.row(i) = vec_t::Unit(dimension, projection_basis[i]);
  }

  projection.setFromTriplets(tripletList.begin(), tripletList.end());
  return projection;
}

mat_t pauli_x() {
  mat_t out(2,2);
  out << 0, 1,
      1, 0;
  return out;
}

mat_t pauli_y() {
  mat_t out(2,2);
  out << 0, -1i,
    1i, 0;
  return out;
}

mat_t pauli_z() {
  mat_t out(2,2);
  out << 1, 0,
      0, -1;
  return out;
}

std::vector<mat_t> pauli_x_vector(int sites) {
  return operator_vector(pauli_x(), sites);
}

std::vector<mat_t> pauli_y_vector(int sites) {
  return operator_vector(pauli_y(), sites);
}

std::vector<mat_t> pauli_z_vector(int sites) {
  return operator_vector(pauli_z(), sites);
}

mat_t pauli_z_total(int sites) {
  return sum_operator(pauli_z(), sites);
}

mat_t pauli_squared_total(int sites) {
  return sum_operator(pauli_x() * pauli_x()
		      + pauli_y() * pauli_y()
		      + pauli_z() * pauli_z(), sites);
}

mat_t HeisenbergChain(int sites,
		      double Jx, double Jy, double Jz,
		      bool periodic) {
  if (sites == 0) {
    mat_t out(1,1);
    out(0,0) = 1.0;
    return out;
  }
  
  assert(sites > 0);
  std::vector<mat_t> x_matrices = pauli_x_vector(sites);
  std::vector<mat_t> y_matrices = pauli_y_vector(sites);
  std::vector<mat_t> z_matrices = pauli_z_vector(sites);

  mat_t ham = mat_t::Zero(z_matrices.back().cols(), z_matrices.back().rows());
  for (int i = 0; i + 1 < sites; ++i) {
    ham += 0.25 * Jx * (x_matrices[i] * x_matrices[i + 1]);
    ham += 0.25 * Jy * (y_matrices[i] * y_matrices[i + 1]);
    ham += 0.25 * Jz * (z_matrices[i] * z_matrices[i + 1]);
  }

  if (periodic && sites > 2) {
    ham += 0.25 * Jx * (x_matrices[sites - 1] * x_matrices[0]);
    ham += 0.25 * Jy * (y_matrices[sites - 1] * y_matrices[0]);
    ham += 0.25 * Jz * (z_matrices[sites - 1] * z_matrices[0]);
  }

  // ham += ham.adjoint().eval();
  return ham - mat_t::Identity(ham.cols(), ham.rows());
}

spmat_t HeisenbergChain_sp(int sites,
			   double Jx, double Jy, double Jz,
			   bool periodic) {
  if (sites == 0) {
    spmat_t out(1,1);
    out.coeffRef(0,0) = 1.0;
    return out;
  }

  int dimension = std::pow(2, sites);
  assert(sites > 0);
  mat_t ham = spmat_t::Zero(dimension, dimension);
  for (int i = 0; i + 1 < sites; ++i) {
    ham += 0.25 * Jx * (n_th_subsystem_sp(pauli_x(), i, sites)
			* n_th_subsystem_sp(pauli_x(), i + 1, sites));
    ham += 0.25 * Jz * (n_th_subsystem_sp(pauli_y(), i, sites)
			* n_th_subsystem_sp(pauli_y(), i + 1, sites));
    ham += 0.25 * Jy * (n_th_subsystem_sp(pauli_z(), i, sites)
			* n_th_subsystem_sp(pauli_z(), i + 1, sites));
  }

  if (periodic && sites > 2) {
    ham += 0.25 * Jx * (n_th_subsystem_sp(pauli_x(), sites - 1, sites)
			* n_th_subsystem_sp(pauli_x(), 0, sites));
    ham += 0.25 * Jz * (n_th_subsystem_sp(pauli_y(), sites - 1, sites)
			* n_th_subsystem_sp(pauli_y(), 0, sites));
    ham += 0.25 * Jy * (n_th_subsystem_sp(pauli_z(), sites - 1, sites)
			* n_th_subsystem_sp(pauli_z(), 0, sites));
  }

  // ham += ham.adjoint().eval();
  return ham - spmat_t::Identity(ham.cols(), ham.rows());
}
