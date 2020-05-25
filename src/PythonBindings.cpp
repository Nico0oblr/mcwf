#define EIGEN_SPARSEMATRIX_PLUGIN "SparseAddons.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>

#include "Operators.hpp"
#include "Common.hpp"
#include "EigenCommon.hpp"
#include "Hamiltonian.hpp"
#include "LightMatterSystem.hpp"
#include "HubbardModel.hpp"
#include "Recorders.hpp"

// Solvers
#include "direct_closed_solver.hpp"
#include "direct_solver.hpp"
#include "mcwf_functions.hpp"
#include "runge_kutta_solver.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mcwf, m) {
    m.doc() = "mcwf simulation bindings"; // optional module docstring

    /*Operators.hpp*/
    m.def("creationOperator", &creationOperator, "");
    m.def("annihilationOperator", &annihilationOperator, "");
    m.def("numberOperator", &annihilationOperator, "");
    m.def("creationOperator_sp", &creationOperator, "");
    m.def("annihilationOperator_sp", &annihilationOperator, "");
    m.def("numberOperator_sp", &annihilationOperator, "");
    m.def("exchange_interaction", &exchange_interaction, "");
    m.def("J0", &J0, "");
    m.def("nth_subsystem", &nth_subsystem, "");
    m.def("operator_vector", &operator_vector, "");
    m.def("sum_operator", &sum_operator, "");
    m.def("pauli_x", &sum_operator, "");
    m.def("pauli_y", &sum_operator, "");
    m.def("pauli_z", &sum_operator, "");
    m.def("pauli_x_vector", &sum_operator, "");
    m.def("pauli_y_vector", &sum_operator, "");
    m.def("pauli_z_vector", &sum_operator, "");
    m.def("pauli_z_total", &sum_operator, "");
    m.def("pauli_squared_total", &sum_operator, "");
    m.def("HeisenbergChain", &sum_operator, "");
    /*Common.hpp*/
    m.def("add_vectors", &add_vectors, "");
    m.def("linear_search", &linear_search<std::vector<double>>, "");
    m.def("linear_search", &linear_search<Eigen::VectorXd>, "");
    m.def("binomial", &binomial, "");
    m.def("poisson", &poisson, "");
    m.def("minus_one_power", &minus_one_power, "");
    /*HubbardModel.hpp*/

    /*Hamiltonian.hpp*/
    py::class_<TimeIndependentHamiltonian<calc_mat_t>>
      (m, "TimeIndependentHamiltonian")
      .def(py::init<const calc_mat_t &>())
      .def(py::init<const calc_mat_t &, const calc_mat_t &, int>())
      .def("propagate", &TimeIndependentHamiltonian<calc_mat_t>::propagate)
      .def("propagator", &TimeIndependentHamiltonian<calc_mat_t>::propagator)
      .def("__call__", &TimeIndependentHamiltonian<calc_mat_t>::operator());

    py::class_<TimeDependentHamiltonian<calc_mat_t>>
      (m, "TimeDependentHamiltonian")
      .def(py::init<const std::function<calc_mat_t(double)> &, int>())
      .def(py::init<const TimeIndependentHamiltonian<calc_mat_t> &>())
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
    m.def("HubbardProjector", &HubbardProjector);

    /*Solvers*/
    auto msolvers = m.def_submodule("Solvers");
    msolvers.def("direct_closed_observable", &direct_closed_observable);
    msolvers.def("direct_closed_two_time_correlation",
		 &direct_closed_two_time_correlation);
    msolvers.def("observable_direct", &observable_direct);
    msolvers.def("two_time_correlation_direct", &two_time_correlation_direct);
    msolvers.def("observable_kutta", &observable_kutta);
    msolvers.def("observable_calc", &observable_calc);
    msolvers.def("two_time_correlation", &two_time_correlation);

    /*Recorders*/
    // Observable recorder
    py::class_<MCWFObservableRecorder>(m, "MCWFObservableRecorder")
      .def("expval", &MCWFObservableRecorder::expval)
      .def("distribution", &MCWFObservableRecorder::distribution)
      .def(py::init<const std::vector<calc_mat_t> & , int>());
    py::class_<StateObservableRecorder>(m, "StateObservableRecorder")
      .def("expval", &StateObservableRecorder::expval)
      .def(py::init<const std::vector<calc_mat_t> &>());
    py::class_<DmatObservableRecorder>(m, "DmatObservableRecorder")
      .def("expval", &DmatObservableRecorder::expval)
      .def(py::init<const std::vector<calc_mat_t> &>());

    // Density matrix recorder
    py::class_<MCWFDmatRecorder>(m, "MCWFDmatRecorder")
      .def("density_matrices", &MCWFDmatRecorder::density_matrices)
            .def(py::init<int>());
    py::class_<DirectStateRecorder>(m, "DirectStateRecorder")
      .def("density_matrices", &DirectStateRecorder::density_matrices);
    py::class_<DirectDmatRecorder>(m, "DirectDmatRecorder")
      .def("density_matrices", &DirectDmatRecorder::density_matrices);
}
/*
.def("__mul__", [](const Vector2 &a, float b) {
    return a * b;
}, py::is_operator())

*/
