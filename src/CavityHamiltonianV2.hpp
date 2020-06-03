#include "Hamiltonian.hpp"
#include "Lindbladian.hpp"

spmat_t Hubbard_site_operator(int photon_dimension,
			      int sites,
			      int site,
			      double coupling,
			      double hopping,
			      double hubbardU,
			      const spmat_t & proj);

spmat_t ST_decomp_exp(int photon_dimension,
		      int sites,
		      double coupling,
		      double hopping,
		      double hubbardU,
		      const spmat_t & proj,
		      double dt);

vec_t ST_decomp_exp_apply(int photon_dimension,
			    int sites,
			    double coupling,
			    double hopping,
			    double hubbardU,
			    const spmat_t & proj,
			    double dt, const vec_t & state);


template<typename Derived>
vec_t apply_tensor_id(const Eigen::EigenBase<Derived> & mat, const vec_t & vec) {
  vec_t out = vec_t::Zero(vec.size());
  int mat_dim = mat.rows();
  int vec_dim = vec.size();
  int codim = vec_dim / mat_dim;
  assert(codim * mat_dim == vec_dim);

  for (int i = 0; i < mat.cols(); ++i) {
    out += Eigen::kroneckerProduct(mat.derived().col(i),
				   vec(Eigen::seq(i * codim, (i + 1) * codim - 1)));
  }
  return out;
}


struct DiagonalizedMatrix {

  void set_from(const calc_mat_t & matrix);
  
  DiagonalizedMatrix & operator=(const calc_mat_t & other);
  
  mat_t exp(std::complex<double> factor) const;
  
  vec_t exp_apply(std::complex<double> factor, const vec_t & vec);
  
  const calc_mat_t & operator()() const;
  
  calc_mat_t self;
  vec_t D;
  mat_t V;
  mat_t V_inv;
};

class CavityHamiltonianV2 : public TimeDependentHamiltonian<calc_mat_t> {
  using Base = TimeDependentHamiltonian<calc_mat_t>;
public:

  const double m_frequency;
  const double m_laser_frequency;
  const double m_laser_amplitude;
  const int m_elec_dim;
  const int m_dimension;
  const calc_mat_t m_light_matter;
  const double m_dt;
  const double m_gamma;
  const double m_n_b;
  size_type m_order;

private:
  mat_t m_X;
  mat_t m_e_X;
  DiagonalizedMatrix m_Y;
  DiagonalizedMatrix m_first_comm;
  DiagonalizedMatrix m_second_comm;
  DiagonalizedMatrix m_third_comm;
  DiagonalizedMatrix m_fourth_comm;
  DiagonalizedMatrix m_fifth_comm;
  
  double m_Y_norm;
  double m_first_norm;
  double m_second_norm;
  double m_third_norm;
  double m_fourth_norm;
  double m_fifth_norm;

  
public:
  double driving_term(double t) const;

  CavityHamiltonianV2(double frequency, double laser_frequency,
		      double laser_amplitude, int elec_dim, int dimension,
		      const calc_mat_t & light_matter,
		      double dt, double gamma = 0.0, double n_b = 0.0);
    
  calc_mat_t propagator(double t, double dt) override;
  
  vec_t BCH_propagate(double t, double dt, const vec_t & state);

  vec_t ST_propagate(double t, double dt, const vec_t & state);

  vec_t propagate(double t, double dt, const vec_t & state) override;

  void set_order(int order);

  CavityHamiltonianV2* clone_impl() const override;
  
};

struct CavityLindbladian : public Lindbladian {

  using Base = Lindbladian;
  CavityLindbladian(double frequency, double laser_frequency,
		    double laser_amplitude, int elec_dim, int dimension,
		    const calc_mat_t & light_matter,
		    double dt, double gamma, double n_b);
     
  std::unique_ptr<Hamiltonian<calc_mat_t>> hamiltonian() const override;
  
  const CavityHamiltonianV2 & hamiltonian_expl() const;
  
  CavityHamiltonianV2 & hamiltonian_expl();
  
  ~CavityLindbladian() override;
  
  CavityHamiltonianV2 mcwf_hamiltonian;
};
