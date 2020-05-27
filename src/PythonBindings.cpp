#define EIGEN_SPARSEMATRIX_PLUGIN "SparseAddons.h"
#define MAKE_SHARED
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/iostream.h>

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

// Temporary
#include "tests.hpp"

#include "CavityHamiltonian.hpp"
#include "CavityHamiltonianV2.hpp"

namespace py = pybind11;

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
    /*Common.hpp*/
    m.def("add_vectors", &add_vectors, "");
    m.def("linear_search", &linear_search<std::vector<double>>, "");
    m.def("linear_search", &linear_search<Eigen::VectorXd>, "");
    m.def("binomial", &binomial, "");
    m.def("poisson", &poisson, "");
    m.def("minus_one_power", &minus_one_power, "");
    /*HubbardModel.hpp*/

    /*Hamiltonian.hpp*/
    py::class_<Hamiltonian<calc_mat_t>> (m, "Hamiltonian");
    py::class_<TimeIndependentHamiltonian<calc_mat_t>, Hamiltonian<calc_mat_t>>
      (m, "TimeIndependentHamiltonian")
      .def(py::init<const calc_mat_t &>())
      .def(py::init<const calc_mat_t &, const calc_mat_t &, int>())
      .def("add", &TimeIndependentHamiltonian<calc_mat_t>::add)
      .def("propagate", &TimeIndependentHamiltonian<calc_mat_t>::propagate)
      .def("propagator", &TimeIndependentHamiltonian<calc_mat_t>::propagator)
      .def("__call__", &TimeIndependentHamiltonian<calc_mat_t>::operator());

    py::class_<TimeDependentHamiltonian<calc_mat_t>, Hamiltonian<calc_mat_t>>
      (m, "TimeDependentHamiltonian")
      .def(py::init<const std::function<calc_mat_t(double)> &, int>())
      .def(py::init<const TimeIndependentHamiltonian<calc_mat_t> &>())
      .def("add", &TimeDependentHamiltonian<calc_mat_t>::add)
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
    m.def("get_spin_sector", &get_spin_sector);
    m.def("HubbardNeelState", &HubbardNeelState);
    m.def("DimerGroundState", &DimerGroundState);
    m.def("HubbardProjector", &HubbardProjector);

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
      .def(py::init<const std::vector<calc_mat_t> & , int>());
    py::class_<StateObservableRecorder, RecorderHost<vec_t>>
      (m, "StateObservableRecorder")
      .def("expval", &StateObservableRecorder::expval)
      .def(py::init<const std::vector<calc_mat_t> &>());
    py::class_<DmatObservableRecorder, RecorderHost<calc_mat_t>>
      (m, "DmatObservableRecorder")
      .def("expval", &DmatObservableRecorder::expval)
      .def(py::init<const std::vector<calc_mat_t> &>());

    // Density matrix recorder
    py::class_<MCWFDmatRecorder, MCWFRecorder>(m, "MCWFDmatRecorder")
      .def("density_matrices", &MCWFDmatRecorder::density_matrices)
      .def(py::init<int>());
    py::class_<DirectStateRecorder, RecorderHost<vec_t>>(m, "DirectStateRecorder")
      .def("density_matrices", &DirectStateRecorder::density_matrices);
    py::class_<DirectDmatRecorder, RecorderHost<calc_mat_t>>
      (m, "DirectDmatRecorder")
      .def("density_matrices", &DirectDmatRecorder::density_matrices);
    
    // Test binding
    m.def("run_tests", &run_tests);

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
    
    /*HSpaceDistribution*/
    py::class_<HSpaceDistribution>(m, "HSpaceDistribution")
      .def("draw", &HSpaceDistribution::draw)
      .def(py::init<const std::vector<double> &, const std::vector<vec_t> &>())
      .def(py::init<const std::vector<double> &,
	   const std::vector<int> &, int>())
      .def(py::init<int>())
      .def(py::self += py::self);
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
      .def("propagator", &CavityHamiltonianV2::propagator)
      .def("set_order", &CavityHamiltonianV2::set_order);

    py::class_<CavityLindbladian, Lindbladian>
      (m, "CavityLindbladian")
      .def(py::init<double, double, double, int, int, const calc_mat_t &,
	   double, double, double>())
      .def("hamiltonian",&CavityLindbladian::hamiltonian)
      .def("system_hamiltonian", &CavityLindbladian::system_hamiltonian,
	   py::return_value_policy::reference_internal);
}
/*
.def("__mul__", [](const Vector2 &a, float b) {
    return a * b;
}, py::is_operator())

*/
