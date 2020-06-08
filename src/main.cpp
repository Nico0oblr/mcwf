#include <fstream>
#include <omp.h>
#include <chrono>

#include "argparse.hpp"
#include "Common.hpp"
#include "Operators.hpp"
#include "HSpaceDistribution.hpp"
#include "Lindbladian.hpp"

#include "mcwf_functions.hpp"
#include "runge_kutta_solver.hpp"
#include "direct_solver.hpp"
#include "direct_closed_solver.hpp"

#include "HubbardModel.hpp"
#include "toy_spin_model.hpp"
#include "Hamiltonian.hpp"
#include "Recorders.hpp"
#include "LightMatterSystem.hpp"

int main(int argc, char ** argv) {
  std::cout << "Starting program" << std::endl;
  /*System creation*/
  YamlOrCMD parser(argv, argc, "config.yaml");
  int num_threads = parser.parse<int>("num_threads");
  omp_set_dynamic(0);
  omp_set_num_threads(num_threads);
  size_type runs = parser.parse<size_type>("runs");
  double dt = parser.parse<double>("dt");
  double time = parser.parse<double>("time");
  LightMatterSystem lms = parse_system(parser);
  std::string observable_name = parser.parse<std::string>("observable");
  calc_mat_t observable = parse_observable(lms, observable_name);
  HSpaceDistribution state_distro = parse_initial_distribution(lms, parser);

  Eigen::VectorXd n_averaged;
  std::string method = parser.parse<std::string>("method");
  double time1 = parser.parse<double>("time1");
  if (method == "mcwf_correlation") {
    /*n_averaged = two_time_correlation(lms.system, state_distro,
				      time1, time, dt, runs, observable,
				      observable).colwise().mean();*/
  } else if (method == "direct_correlation") {
    /*n_averaged = two_time_correlation_direct(lms.system, state_distro,
					     time1, time, dt, observable,
					     observable);*/
  } else if (method == "direct_closed_correlation") {
    /*n_averaged = direct_closed_two_time_correlation(lms.hamiltonian(),
						    state_distro.draw(),
						    time1, time, dt, observable,
						    observable);*/
  } else if (method == "mcwf") {
    ExpvalWriterMixin<MCWFObservableRecorder> recorder({observable});
    observable_calc(lms.system, state_distro, time, dt, runs, recorder);
    recorder.write("results.csv");
    return 0;
  } else if (method == "runge_kutta") {
    ExpvalWriterMixin<DmatObservableRecorder> recorder({observable});
    observable_kutta(lms.system, state_distro, time, dt, recorder);
    recorder.write("results.csv");
    return 0;
  } else if (method == "direct") {
    ExpvalWriterMixin<DmatObservableRecorder> recorder({observable});
    observable_direct(lms.system, state_distro, time, dt, recorder);
    recorder.write("results.csv");
    return 0;
  } else if (method == "direct_closed") {
    ExpvalWriterMixin<StateObservableRecorder> recorder({observable});
    direct_closed_observable(*lms.system.m_system_hamiltonian,
			     state_distro.draw(), time, dt, recorder);
    recorder.write("results.csv");
    return 0;
  } else if (method == "compare") {
    MCWFDmatRecorder mcwf_recorder;
    DirectDmatRecorder direct_recorder;
    DirectDmatRecorder kutta_recorder;
    auto start = std::chrono::high_resolution_clock::now(); 
    observable_kutta(lms.system, state_distro, time, dt, kutta_recorder);
    auto start_mcwf = std::chrono::high_resolution_clock::now(); 
    observable_calc(lms.system, state_distro, time, dt, runs, mcwf_recorder);
    auto start_direct = std::chrono::high_resolution_clock::now(); 
    observable_direct(lms.system, state_distro, time, dt, direct_recorder);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::vector<calc_mat_t> direct_dmat = direct_recorder.density_matrices();
    std::vector<calc_mat_t> kutta_dmat = kutta_recorder.density_matrices();
    std::vector<calc_mat_t> mcwf_dmat = mcwf_recorder.density_matrices();
    assert(mcwf_dmat.size() == kutta_dmat.size());
    assert(mcwf_dmat.size() == direct_dmat.size());

    for (size_type i = 0; i < mcwf_dmat.size(); ++i) {
      std::cout << "time step " << i << std::endl;
      std::cout << "mcwf-kutta difference: "
		<< calc_mat_t((mcwf_dmat[i] - kutta_dmat[i])
			      * (mcwf_dmat[i] - kutta_dmat[i])).trace()
		<< std::endl;
      std::cout << "mcwf-direct difference: "
		<< calc_mat_t((mcwf_dmat[i] - direct_dmat[i])
			      * (mcwf_dmat[i] - direct_dmat[i])).trace()
		<< std::endl;
      std::cout << "kutta-direct difference: "
		<< calc_mat_t((kutta_dmat[i] - direct_dmat[i])
			      * (kutta_dmat[i] - direct_dmat[i])).trace()
		<< std::endl;
      std::cout << "==========" << std::endl;
    }

    auto kutta_duration = std::chrono::duration_cast
      <std::chrono::milliseconds>(start_mcwf - start);
    auto mcwf_duration = std::chrono::duration_cast
      <std::chrono::milliseconds>(start_direct - start_mcwf);
    auto direct_duration = std::chrono::duration_cast
      <std::chrono::milliseconds>(end - start_direct);
    std::cout << "kutta_duration: " << kutta_duration.count() << std::endl;
    std::cout << "mcwf_duration: " << mcwf_duration.count() << std::endl;
    std::cout << "direct_duration: " << direct_duration.count() << std::endl;
  
    return 0;
  } else {
    assert(false && "Calculation method not found");
  }
  
  /*Average data and write to file*/
  std::ofstream output("results.csv");
  Eigen::IOFormat fmt(Eigen::StreamPrecision,
		      Eigen::DontAlignCols, ", ", ", ", "", "", "", "");
  output << n_averaged.format(fmt) << std::endl;
  output.close();
  }
