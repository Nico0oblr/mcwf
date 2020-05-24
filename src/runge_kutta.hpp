#ifndef RUNGE_KUTTA_HPP
#define RUNGE_KUTTA_HPP

#include <Eigen/Dense>
#include <vector>
#include <iostream>

class RungeKuttaSolver {

public:
  template<typename value_type>
  value_type
  perform_step(double t,
	       double dt,
	       const value_type & y,
	       const std::function<value_type(double,
					      const value_type &)> & func) {
    std::vector<value_type> ks;
    for (int i = 0; i + 1 < m_butcher_tableau.cols(); ++i) {
      value_type argument = y;
      for (int j = 1; j < i; ++j) {
	argument += dt * m_butcher_tableau(i, j) * ks[j];
      }
      ks.push_back(func(t + dt * m_butcher_tableau(i, 0), argument));
    }

    value_type result = y;
    for (int i = 1; i < m_butcher_tableau.cols(); ++i) {
      result += dt * m_butcher_tableau(m_butcher_tableau.rows() - 1, i) * ks[i - 1];
    }

    if (m_butcher_tableau.rows() > m_butcher_tableau.cols()) {
      value_type check_result = y;
      for (int i = 1; i < m_butcher_tableau.cols(); ++i) {
	check_result += dt * m_butcher_tableau(m_butcher_tableau.rows() - 2, i) * ks[i - 1];
      }
      std::cout << (result - check_result).norm() << std::endl;
    }

    return result;
  }

  RungeKuttaSolver(Eigen::MatrixXd butcher_tableau);

  private:
   Eigen::MatrixXd m_butcher_tableau;
};

RungeKuttaSolver build_runge_kutta_4();

RungeKuttaSolver cash_karp();

RungeKuttaSolver dormand_price();

template<typename value_type>
value_type
rungeKuttaStep(double dt,
	       const value_type & y,
	       const std::function<value_type(const value_type &)> & func) {
  value_type k1 = dt * func(y); 
  value_type k2 = dt * func(y + 0.5 * k1); 
  value_type k3 = dt * func(y + 0.5 * k2); 
  value_type k4 = dt * func(y + k3); 
  return y + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

#endif /* RUNGE_KUTTA_HPP */
