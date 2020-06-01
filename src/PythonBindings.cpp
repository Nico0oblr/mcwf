#define EIGEN_SPARSEMATRIX_PLUGIN "SparseAddons.h"
#define MAKE_SHARED
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/iostream.h>
#include <pybind11/complex.h>

#include "Operators.hpp"
#include "Common.hpp"
#include "EigenCommon.hpp"
#include "Hamiltonian.hpp"
#include "LightMatterSystem.hpp"
#include "Recorders.hpp"
#include "HSpaceDistribution.hpp"

#include "toy_spin_model.hpp"
#include "HubbardModel.hpp"

// Solvers
#include "direct_closed_solver.hpp"
#include "direct_solver.hpp"
#include "mcwf_functions.hpp"
#include "runge_kutta_solver.hpp"


#include "CavityHamiltonian.hpp"
#include "CavityHamiltonianV2.hpp"
#include "PadeExponential.hpp"
#include "OneNormEst.hpp"
#include "MatrixExpApply.hpp"

#include "ArnoldiIteration.hpp"
#include "Kron.hpp"
#include "LinearOperator.hpp"

namespace py = pybind11;


template<typename T>
void declare_arnoldi(py::module & mod, const std::string & typestr) {
  std::string pyclass_name = std::string("ArnoldiIteration") + typestr;
  py::class_<ArnoldiIteration<T>>(mod, pyclass_name.c_str())
    .def("H", &ArnoldiIteration<T>::H)
    .def("V", &ArnoldiIteration<T>::V)
    .def("nit", &ArnoldiIteration<T>::nit)
    .def("eigenvectors", &ArnoldiIteration<T>::eigenvectors)
    .def("eigenvalues", &ArnoldiIteration<T>::eigenvalues)
    .def("k_n_arnoldi", &ArnoldiIteration<T>::k_n_arnoldi)
    .def("restart", &ArnoldiIteration<T>::restart)
    .def("apply_exp", &ArnoldiIteration<T>::apply_exp)
    .def(py::init<const T &, int, int>())
    .def(py::init<const T &, int, int, const vec_t &>())
    .def(py::init<>());
}

PYBIND11_MODULE(mcwf, m) {  
  m.doc() = "mcwf simulation bindings"; // optional module docstring

  /*Operators.hpp*/
  m.def("creationOperator", &creationOperator, "");
  m.def("annihilationOperator", &annihilationOperator, "");
  m.def("numberOperator", &numberOperator, "");
  m.def("creationOperator_sp", &creationOperator, "");
  m.def("annihilationOperator_sp", &annihilationOperator, "");
  m.def("numberOperator_sp", &numberOperator, "");
  m.def("exchange_interaction", &exchange_interaction, "");
  m.def("J0", &J0, "");
  m.def("nth_subsystem", &nth_subsystem, "");
  m.def("n_th_subsystem_sp", &n_th_subsystem_sp, "");
  m.def("sum_operator_sp", &sum_operator_sp, "");
  m.def("operator_vector", &operator_vector, "");
  m.def("sum_operator", &sum_operator, "");
  m.def("pauli_x", &pauli_x, "");
  m.def("pauli_y", &pauli_y, "");
  m.def("pauli_z", &pauli_z, "");
  m.def("pauli_x_vector", &pauli_x_vector, "");
  m.def("pauli_y_vector", &pauli_y_vector, "");
  m.def("pauli_z_vector", &pauli_z_vector, "");
  m.def("pauli_z_total", &pauli_z_total, "");
  m.def("pauli_squared_total", &pauli_squared_total, "");
  m.def("HeisenbergChain", &HeisenbergChain, "");
  m.def("L_p", &L_p, "");
  m.def("L_c_m", &L_c_m, "");
  /*Common.hpp*/
  m.def("add_vectors", &add_vectors, "");
  m.def("linear_search", &linear_search<std::vector<double>>, "");
  m.def("linear_search", &linear_search<Eigen::VectorXd>, "");
  m.def("binomial", &binomial, "");
  m.def("factorial", &factorial, "");
  m.def("poisson", &poisson, "");
  m.def("minus_one_power", &minus_one_power, "");
  /*HubbardModel.hpp*/

  /*Hamiltonian.hpp*/
  py::class_<Hamiltonian<calc_mat_t>> (m, "Hamiltonian");
  py::class_<TimeIndependentHamiltonian<calc_mat_t>, Hamiltonian<calc_mat_t>>
    (m, "TimeIndependentHamiltonian")
    .def(py::init<const calc_mat_t &>())
    .def(py::init<const calc_mat_t &, const calc_mat_t &, int>())
    .def("add", py::overload_cast<const spmat_t &>(&TimeIndependentHamiltonian<calc_mat_t>::add))
    .def("add", py::overload_cast<const LinearOperator<spmat_t> &>(&TimeIndependentHamiltonian<calc_mat_t>::add))
    .def("propagate", &TimeIndependentHamiltonian<calc_mat_t>::propagate)
    .def("propagator", &TimeIndependentHamiltonian<calc_mat_t>::propagator)
    .def("__call__", &TimeIndependentHamiltonian<calc_mat_t>::operator());

  py::class_<TimeDependentHamiltonian<calc_mat_t>, Hamiltonian<calc_mat_t>>
    (m, "TimeDependentHamiltonian")
    .def(py::init<const std::function<calc_mat_t(double)> &, int>())
    // .def(py::init<const TimeIndependentHamiltonian<calc_mat_t> &>())
    .def("add", py::overload_cast<const spmat_t &>(&TimeDependentHamiltonian<calc_mat_t>::add))
    .def("add", py::overload_cast<const LinearOperator<spmat_t> &>(&TimeDependentHamiltonian<calc_mat_t>::add))
    .def("propagate", &TimeDependentHamiltonian<calc_mat_t>::propagate)
    .def("propagator", &TimeDependentHamiltonian<calc_mat_t>::propagator)
    .def("__call__", &TimeDependentHamiltonian<calc_mat_t>::operator());
  /*LightMatterSystem.hpp*/
  py::class_<LightMatterSystem>
    (m, "LightMatterSystem")
    .def("elec_dim", &LightMatterSystem::elec_dim)
    .def("hamiltonian", py::overload_cast<>(&LightMatterSystem::hamiltonian,
					    py::const_))
    .def("hamiltonian", py::overload_cast<>(&LightMatterSystem::hamiltonian));

  /*Lindbladian.hpp*/
  py::class_<Lindbladian>
    (m, "Linbladian")
    .def("hamiltonian", &Lindbladian::hamiltonian)
    .def("system_hamiltonian", &Lindbladian::system_hamiltonian,
	 py::return_value_policy::reference_internal)
    .def("add_subsystem", &Lindbladian::add_subsystem)
    .def("__call__", &Lindbladian::operator())
    .def("superoperator", &Lindbladian::superoperator)
    .def(py::init<const Hamiltonian<calc_mat_t> &,
	 const std::vector<calc_mat_t> &,
	 const std::vector<scalar_t> &>())
    .def(py::init<const Hamiltonian<calc_mat_t> &,
	 const std::vector<calc_mat_t> &,
	 const Eigen::MatrixXd &>())
    .def(py::init<const Lindbladian &>());
  m.def("bose_distribution", &bose_distribution);
  m.def("thermalCavity", &thermalCavity);
  m.def("drivenCavity", &drivenCavity);

  /*HubbardModel*/
  auto mhub = m.def_submodule("HubbardOperators");
  mhub.def("c_up_t", &HubbardOperators::c_up_t);
  mhub.def("c_down_t", &HubbardOperators::c_down_t);
  mhub.def("c_up", &HubbardOperators::c_up);
  mhub.def("c_down", &HubbardOperators::c_down);
  mhub.def("n_down", &HubbardOperators::n_down);
  mhub.def("n_up", &HubbardOperators::n_up);
  m.def("Hubbard_hamiltonian", &Hubbard_hamiltonian);
  m.def("Hubbard_light_matter", &Hubbard_light_matter);
  m.def("Hubbard_light_matter_sp", &Hubbard_light_matter_sp);
  m.def("get_spin_sector", &get_spin_sector);
  m.def("HubbardNeelState", &HubbardNeelState);
  m.def("HubbardNeelState_sp", &HubbardNeelState_sp);
  m.def("DimerGroundState", &DimerGroundState);
  m.def("HubbardProjector", &HubbardProjector);
  m.def("HubbardProjector_sp", &HubbardProjector_sp);

  /*Solvers*/
  auto msolvers = m.def_submodule("Solvers");
  msolvers.def("direct_closed_observable", &direct_closed_observable,
	       py::call_guard<py::scoped_ostream_redirect,
	       py::scoped_estream_redirect>());
  msolvers.def("direct_closed_two_time_correlation",
	       &direct_closed_two_time_correlation,
	       py::call_guard<py::scoped_ostream_redirect,
	       py::scoped_estream_redirect>());
  msolvers.def("observable_direct", &observable_direct,
	       py::call_guard<py::scoped_ostream_redirect,
	       py::scoped_estream_redirect>());
  msolvers.def("two_time_correlation_direct", &two_time_correlation_direct,
	       py::call_guard<py::scoped_ostream_redirect,
	       py::scoped_estream_redirect>());
  msolvers.def("observable_kutta", &observable_kutta,
	       py::call_guard<py::scoped_ostream_redirect,
	       py::scoped_estream_redirect>());
  msolvers.def("observable_calc", &observable_calc,
	       py::call_guard<py::scoped_ostream_redirect,
	       py::scoped_estream_redirect>());
  msolvers.def("two_time_correlation", &two_time_correlation,
	       py::call_guard<py::scoped_ostream_redirect,
	       py::scoped_estream_redirect>());

  /*Recorders*/
  // Observable recorder
  py::class_<RecorderHost<calc_mat_t>> (m, "RecorderHost_mat");
  py::class_<RecorderHost<vec_t>> (m, "RecorderHost_vec");
  py::class_<MCWFRecorder> (m, "MCWFRecorder");
    
  py::class_<MCWFObservableRecorder, MCWFRecorder>(m, "MCWFObservableRecorder")
    .def("expval", &MCWFObservableRecorder::expval)
    .def("distribution", &MCWFObservableRecorder::distribution)
    .def(py::init<const std::vector<calc_mat_t> & , int>())
    .def(py::pickle
	 (
	  [](const MCWFObservableRecorder &p) { // __getstate__
	    std::vector<calc_mat_t> evaluated_obs;
	    for (const auto & obs : p.m_observables)
	      evaluated_obs.push_back(obs->eval());
	    return py::make_tuple(evaluated_obs,
				  p.n_runs(),
				  p.m_records);
	  },
	  [](py::tuple t) { // __setstate__
	    if (t.size() != 3) {
	      throw std::runtime_error("Invalid state!");
	    }	      
	    MCWFObservableRecorder tmp(t[0].cast<std::vector<calc_mat_t>>(),
				       t[1].cast<int>());
	    tmp.m_records = t[2].cast<std::vector<std::vector<std::vector<double>>>>();
	    return tmp;
	  }
	  ));
  py::class_<StateObservableRecorder, RecorderHost<vec_t>>
    (m, "StateObservableRecorder")
    .def("expval", &StateObservableRecorder::expval)
    .def(py::init<const std::vector<calc_mat_t> &>())
    .def(py::pickle
	 (
	  [](const StateObservableRecorder &p) { // __getstate__
	    std::vector<calc_mat_t> evaluated_obs;
	    for (const auto & obs : p.m_observables)
	      evaluated_obs.push_back(obs->eval());
	    return py::make_tuple(evaluated_obs,
				  p.m_records);
	  },
	  [](py::tuple t) { // __setstate__
	    if (t.size() != 2) {
	      throw std::runtime_error("Invalid state!");
	    }	      
	    StateObservableRecorder tmp(t[0].cast<std::vector<calc_mat_t>>());
	    tmp.m_records = t[1].cast<std::vector<std::vector<double>>>();
	    return tmp;
	  }
	  ));
  py::class_<DmatObservableRecorder, RecorderHost<calc_mat_t>>
    (m, "DmatObservableRecorder")
    .def("expval", &DmatObservableRecorder::expval)
    .def(py::init<const std::vector<calc_mat_t> &>())
    .def(py::pickle
	 (
	  [](const DmatObservableRecorder &p) { // __getstate__
	    std::vector<calc_mat_t> evaluated_obs;
	    for (const auto & obs : p.m_observables)
	      evaluated_obs.push_back(obs->eval());
	    return py::make_tuple(evaluated_obs,
				  p.m_records);
	  },
	  [](py::tuple t) { // __setstate__
	    if (t.size() != 2) {
	      throw std::runtime_error("Invalid state!");
	    }	      
	    DmatObservableRecorder tmp(t[0].cast<std::vector<calc_mat_t>>());
	    tmp.m_records = t[1].cast<std::vector<std::vector<double>>>();
	    return tmp;
	  }
	  ));

  // Density matrix recorder
  py::class_<MCWFDmatRecorder, MCWFRecorder>(m, "MCWFDmatRecorder")
    .def("density_matrices", &MCWFDmatRecorder::density_matrices)
    .def(py::init<int>());
  py::class_<DirectStateRecorder, RecorderHost<vec_t>>(m, "DirectStateRecorder")
    .def("density_matrices", &DirectStateRecorder::density_matrices);
  py::class_<DirectDmatRecorder, RecorderHost<calc_mat_t>>
    (m, "DirectDmatRecorder")
    .def("density_matrices", &DirectDmatRecorder::density_matrices);
    
  /*toy_spin_model.hpp*/
  m.def("J0_n", &J0_n);
  m.def("toy_modelize", &toy_modelize);
    
  /*EigenCommon*/
  m.def("tensor_identity", [](const spmat_t & op, int dim)
	-> spmat_t {return tensor_identity(op, dim);});
  m.def("tensor_identity", [](const Eigen::Ref<const mat_t> & op, int dim)
	-> mat_t {return tensor_identity(op, dim);});
  m.def("tensor_identity_LHS", [](const spmat_t & op, int dim)
	-> spmat_t {return tensor_identity_LHS(op, dim);});
  m.def("tensor_identity_LHS", [](const Eigen::Ref<const mat_t> & op, int dim)
	-> mat_t {return tensor_identity_LHS(op, dim);});
    
  m.def("double_matrix", [](const spmat_t & op)
	-> spmat_t {return double_matrix(op);});
  m.def("double_matrix", [](const Eigen::Ref<const mat_t> & op)
	-> mat_t {return double_matrix(op);});

  m.def("matrix_exponential_taylor", [](const spmat_t & op, int order)
	-> spmat_t {return matrix_exponential_taylor(op, order);});
  m.def("matrix_exponential_taylor", [](const Eigen::Ref<const mat_t> & op,
					int order)
	-> mat_t {return matrix_exponential_taylor(op, order);});
  m.def("apply_matrix_exponential_taylor",
	[](const spmat_t & op, const vec_t & state, int order)
	-> spmat_t {return apply_matrix_exponential_taylor(op, state, order);});
  m.def("apply_matrix_exponential_taylor",
	[](const Eigen::Ref<const mat_t> & op, const vec_t & state, int order)
	-> mat_t {return apply_matrix_exponential_taylor(op, state, order);});
  m.def("matrix_exponential", [](const Eigen::Ref<const mat_t> & op)
	-> mat_t {return matrix_exponential(op);});

  // Superoperator methods
  m.def("superoperator_left", [](const spmat_t & op, int dimension)
	-> spmat_t {return superoperator_left(op, dimension);});
  m.def("superoperator_right", [](const spmat_t & op, int dimension)
	-> spmat_t {return superoperator_right(op, dimension);});
  m.def("unstack_matrix", [](const spmat_t & op)
	-> spmat_t {return unstack_matrix(op);});
  m.def("restack_vector", &restack_vector);
  m.def("superoperator_left", [](const Eigen::Ref<const mat_t> & op,
				 int dimension)
	-> mat_t {return superoperator_left(op, dimension);});
  m.def("superoperator_right", [](const Eigen::Ref<const mat_t> & op,
				  int dimension)
	-> mat_t {return superoperator_right(op, dimension);});
  m.def("unstack_matrix", [](const Eigen::Ref<const mat_t> & op)
	-> mat_t {return unstack_matrix(op);});

    
  /*HSpaceDistribution*/
  py::class_<HSpaceDistribution>(m, "HSpaceDistribution")
    .def("draw", &HSpaceDistribution::draw)
    .def(py::init<const std::vector<double> &, const std::vector<vec_t> &>())
    .def(py::init<const std::vector<double> &,
	 const std::vector<int> &, int>())
    .def(py::init<int>())
    .def(py::self += py::self)
    .def(py::pickle
	 (
	  [](const HSpaceDistribution &p) { // __getstate__
	    return py::make_tuple(p.m_probabilities,
				  p.m_states);
	  },
	  [](py::tuple t) { // __setstate__
	    if (t.size() != 2) {
	      throw std::runtime_error("Invalid state!");
	    }	      
	    return HSpaceDistribution(t[0].cast<std::vector<double>>(),
				      t[1].cast<std::vector<vec_t>>());
	  }
	  ));
	   
  m.def("coherent_photon_state", &coherent_photon_state);
  m.def("set_num_threads", [](int num_threads) {
			     omp_set_dynamic(0);
			     omp_set_num_threads(num_threads);
			   });
  m.def("get_max_threads", []() {
			     return omp_get_max_threads();
			   });

  /*CavityHamiltonian*/
  py::class_<CavityHamiltonian, TimeDependentHamiltonian<calc_mat_t>>
    (m, "CavityHamiltonian")
    .def(py::init<double, double, double, int, int ,
	 const calc_mat_t, double>())
    .def("propagate", &CavityHamiltonian::propagate)
    .def("propagator", &CavityHamiltonian::propagator);

  py::class_<CavityHamiltonianV2, TimeDependentHamiltonian<calc_mat_t>>
    (m, "CavityHamiltonianV2")
    .def(py::init<double, double, double, int, int ,
	 const calc_mat_t, double, double, double>())
    .def(py::init<double, double, double, int, int ,
	 const calc_mat_t, double>())
    .def("propagate", &CavityHamiltonianV2::propagate)
    .def("BCH_propagate", &CavityHamiltonianV2::BCH_propagate)
    .def("ST_propagate", &CavityHamiltonianV2::ST_propagate)
    .def("propagator", &CavityHamiltonianV2::propagator)
    .def("set_order", &CavityHamiltonianV2::set_order)
    .def_readonly("frequency", &CavityHamiltonianV2::m_frequency)
    .def_readonly("laser_frequency", &CavityHamiltonianV2::m_laser_frequency)
    .def_readonly("laser_amplitude", &CavityHamiltonianV2::m_laser_amplitude)
    .def_readonly("elec_dim", &CavityHamiltonianV2::m_elec_dim)
    .def_readonly("dimension", &CavityHamiltonianV2::m_dimension)
    .def_readonly("light_matter", &CavityHamiltonianV2::m_light_matter)
    .def_readonly("dt", &CavityHamiltonianV2::m_dt)
    .def_readonly("gamma", &CavityHamiltonianV2::m_gamma)
    .def_readonly("n_b", &CavityHamiltonianV2::m_n_b)
    .def(py::pickle
	 (
	  [](const CavityHamiltonianV2 &p) { // __getstate__
	    return py::make_tuple(p.m_frequency,
				  p.m_laser_frequency,
				  p.m_laser_amplitude,
				  p.m_elec_dim,
				  p.m_dimension,
				  p.m_light_matter,
				  p.m_dt,
				  p.m_gamma,
				  p.m_n_b,
				  p.m_order);
	  },
	  [](py::tuple t) { // __setstate__
	    if (t.size() != 10) {
	      throw std::runtime_error("Invalid state!");
	    }	      
	    CavityHamiltonianV2 distro(t[0].cast<double>(),
				       t[1].cast<double>(),
				       t[2].cast<double>(),
				       t[3].cast<int>(),
				       t[4].cast<int>(),
				       t[5].cast<calc_mat_t>(),
				       t[6].cast<double>(),
				       t[7].cast<double>(),
				       t[8].cast<double>());
	    distro.set_order(t[9].cast<int>());
	    return distro;
	  }
	  ));

  py::class_<CavityLindbladian, Lindbladian>
    (m, "CavityLindbladian")
    .def(py::init<double, double, double, int, int, const calc_mat_t &,
	 double, double, double>())
    .def("hamiltonian",&CavityLindbladian::hamiltonian)
    .def("system_hamiltonian", &CavityLindbladian::system_hamiltonian,
	 py::return_value_policy::reference_internal)
    .def(py::pickle
	 (
	  [](const CavityLindbladian &p) { // __getstate__
	    return py::make_tuple(p.hamiltonian_expl().m_frequency,
				  p.hamiltonian_expl().m_laser_frequency,
				  p.hamiltonian_expl().m_laser_amplitude,
				  p.hamiltonian_expl().m_elec_dim,
				  p.hamiltonian_expl().m_dimension,
				  p.hamiltonian_expl().m_light_matter,
				  p.hamiltonian_expl().m_dt,
				  p.hamiltonian_expl().m_gamma,
				  p.hamiltonian_expl().m_n_b,
				  p.hamiltonian_expl().m_order);
	  },
	  [](py::tuple t) { // __setstate__
	    if (t.size() != 10) {
	      throw std::runtime_error("Invalid state!");
	    }	      
	    CavityLindbladian distro(t[0].cast<double>(),
				     t[1].cast<double>(),
				     t[2].cast<double>(),
				     t[3].cast<int>(),
				     t[4].cast<int>(),
				     t[5].cast<calc_mat_t>(),
				     t[6].cast<double>(),
				     t[7].cast<double>(),
				     t[8].cast<double>());
	    distro.hamiltonian_expl().set_order(t[9].cast<int>());
	    return distro;
	  }
	  ));
  /*PadeExponential.hpp*/
  m.def("onenorm_power", &onenorm_power, "");
  m.def("_ell", &_ell, "");
  m.def("expm", &expm, "");

  m.def("Hubbard_site_operator", &Hubbard_site_operator);
  m.def("ST_decomp_exp", &ST_decomp_exp);
  m.def("ST_decomp_exp_apply", &ST_decomp_exp_apply);

  /*OneNormEst*/
  // m.def("_algorithm_2_2", &_algorithm_2_2<spmat_t, spmat_t>);
  m.def("onenormest", [](const spmat_t & mat) {return onenormest(mat);});
  m.def("onenormestOperator", [](const LinearOperator<spmat_t> & mat)
			      {return onenormest(mat);});
  m.def("randint", &randint);
  // m.def("close", &close);
  m.def("less_than_or_close", &less_than_or_close);
  m.def("in1d", &in1d<Eigen::MatrixXd, Eigen::MatrixXd>);
  m.def("invert_indexer", &invert_indexer);
  m.def("algorithm_2_2", [](const spmat_t & A, const spmat_t & B, int t)
			 {return _algorithm_2_2(A, B, t);});
  m.def("onenormest_matrix_power", [](const spmat_t & mat, int p) {
				     return onenormest_matrix_power(mat, p);});
  m.def("onenormest_operator_power", [](const LinearOperator<spmat_t> & mat,
					int p) {
				       return onenormest_matrix_power(mat, p);});


  /*Expm apply*/
  m.def("expm_multiply_simple", [](const spmat_t & A, const vec_t & vec)
				{return expm_multiply_simple(A, vec);});
  m.def("expm_multiply_simple_operator",
	[](const LinearOperator<spmat_t> & A, const vec_t & vec)
	{return expm_multiply_simple(A, vec);});
  m.def("expm_multiply_simple_core", &_expm_multiply_simple_core);
  m.def("condition_3_13", &_condition_3_13);
  m.def("fragment_3_1", &_fragment_3_1);
  m.def("compute_cost_div_m", &_compute_cost_div_m);
  m.def("compute_p_max", &compute_p_max);
  m.def("exact_onenorm", [](const spmat_t & mat) {return mat.oneNorm();});
  m.def("exact_infnorm", [](const spmat_t & mat) {return mat.infNorm();});

  py::class_<LazyOperatorNormInfo<spmat_t>>(m, "LazyOperatorNormInfo")
    .def("set_scale", &LazyOperatorNormInfo<spmat_t>::set_scale)
    .def("onenorm", &LazyOperatorNormInfo<spmat_t>::onenorm)
    .def("d", &LazyOperatorNormInfo<spmat_t>::d)
    .def("alpha", &LazyOperatorNormInfo<spmat_t>::alpha)
    .def(py::init<const LinearOperator<spmat_t> &, double>());

  /*ArnoldiIteration.hpp*/
  declare_arnoldi<spmat_t>(m, "");

  m.def("exp_krylov", &exp_krylov);
  m.def("exp_krylov_alt", &exp_krylov_alt);
  m.def("kroneckerApply", &kroneckerApply);
  m.def("kroneckerApply_id", &kroneckerApply_id);
  m.def("kroneckerApply_LHS", &kroneckerApply_LHS);
  m.def("kroneckerApplyLazy", &kroneckerApply);

  py::class_<LinearOperator<spmat_t>>(m, "LinearOperator")
    .def("apply_to", &LinearOperator<spmat_t>::apply_to)
    .def("applied_to", &LinearOperator<spmat_t>::applied_to)
    .def("eval", &LinearOperator<spmat_t>::eval)
    .def("rows", &LinearOperator<spmat_t>::rows)
    .def("mult_by_scalar", &LinearOperator<spmat_t>::mult_by_scalar)
    .def("adjoint", &LinearOperator<spmat_t>::adjoint)
    .def("adjointInPlace", &LinearOperator<spmat_t>::adjointInPlace)
    .def("clone", &LinearOperator<spmat_t>::clone)
    .def("__mul__", [](const LinearOperator<spmat_t> & a,
		       const LinearOperator<spmat_t> & b) {
		      return a * b;
		    }, py::is_operator())
    .def("__sub__", [](const LinearOperator<spmat_t> & a,
		       const LinearOperator<spmat_t> & b) {
		      return a - b;
		    }, py::is_operator())
    .def("__add__", [](const LinearOperator<spmat_t> & a,
		       const LinearOperator<spmat_t> & b) {
		      return a + b;
		    }, py::is_operator())
    .def("__rmul__", [](const LinearOperator<spmat_t> & b,
			const Eigen::Matrix<std::complex<double>, 1, Eigen::Dynamic> & a) -> vec_t {
		       // return a.applied_to(b.transpose());
		       return a * b;
		     }, py::is_operator())
    .def("__rmul__", [](const LinearOperator<spmat_t> & b,
			const mat_t & a) -> mat_t {
		       return a * b;
		     }, py::is_operator())
    .def("__mul__", [](const LinearOperator<spmat_t> & a,
		       const mat_t & b) -> mat_t {
		      return a * b;
		    }, py::is_operator())
    .def("__mul__", [](const LinearOperator<spmat_t> & a,
		       const std::complex<double> & b) {
		      auto copy = a.clone();
		      copy->mult_by_scalar(b);
		      return copy;
		    }, py::is_operator())
    .def("__rmul__", [](const LinearOperator<spmat_t> & a,
			const std::complex<double> & b) {
		       auto copy = a.clone();
		       copy->mult_by_scalar(b);
		       return copy;
		     }, py::is_operator());
  m.def("sumOperator", &sumOperator<spmat_t>);
    
  /*LinearOperator*/
  m.def("Hubbard_light_matter_Operator", &Hubbard_light_matter_Operator);
  m.def("operatorize", &operatorize<spmat_t>);
  m.def("kroneckerOperator", &kroneckerOperator<spmat_t>);
  m.def("kroneckerOperator_IDRHS", &kroneckerOperator_IDRHS<spmat_t>);
  m.def("kroneckerOperator_IDLHS", &kroneckerOperator_IDLHS<spmat_t>);
  m.def("doubleOperator", [](const LinearOperator<spmat_t>& mat)
			  {return doubleOperator(mat);});
  m.def("powerOperator", [](const LinearOperator<spmat_t>& mat, int p)
			 {return powerOperator(mat, p);});
  m.def("scale_and_add", &scale_and_add<spmat_t>);
  m.def("scale_rhs_and_add", &scale_rhs_and_add<spmat_t>);
  declare_arnoldi<LinearOperator<spmat_t>>(m, "Operator");

  py::class_<SumLinearOperator<spmat_t>, LinearOperator<spmat_t>>(m, "SumLinearOperator");
  py::class_<MultLinearOperator<spmat_t>, LinearOperator<spmat_t>>(m, "MultLinearOperator");
  py::class_<KroneckerLinearOperator<spmat_t>, LinearOperator<spmat_t>>(m, "KroneckerLinearOperator");
  py::class_<KroneckerIDRHSLinearOperator<spmat_t>, LinearOperator<spmat_t>>(m, "KroneckerIDRHSLinearOperator");
  py::class_<KroneckerIDLHSLinearOperator<spmat_t>, LinearOperator<spmat_t>>(m, "KroneckerIDLHSLinearOperator");
  py::class_<DoubledLinearOperator<spmat_t>, LinearOperator<spmat_t>>(m, "DoubledLinearOperator");
  py::class_<PowerLinearOperator<spmat_t>, LinearOperator<spmat_t>>(m, "PowerLinearOperator");
}
