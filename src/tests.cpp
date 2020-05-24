#include "tests.hpp"

#include "Common.hpp"
#include "Operators.hpp"
#include "HubbardModel.hpp"

void superoperator_test(int dimension) {
  mat_t vec = Eigen::MatrixXd::Random(dimension, dimension);
  mat_t op = Eigen::MatrixXd::Random(dimension, dimension);
  mat_t result_lhs1 = restack_vector(superoperator_left(op, dimension) * unstack_matrix(vec), dimension);
  mat_t result_lhs2 = op * vec;
  mat_t result_rhs1 = restack_vector(superoperator_right(op, dimension) * unstack_matrix(vec), dimension);
  mat_t result_rhs2 = vec * op;

  /* Should all be zero/small */
  assert((restack_vector(unstack_matrix(vec), dimension) - vec).norm() < tol);
  assert((unstack_matrix(vec) - unstack_matrix_alt(vec)).norm() < tol);
  assert((result_lhs1 - result_lhs2).norm() < tol);
  assert((result_rhs1 - result_rhs2).norm() < tol);
}

void function_tests() {
  assert(factorial(4) == 4 * 3 * 2);
  assert(factorial(5) == 5 * 4 * 3 * 2);
  assert(binomial(5, 2) == 10);
  assert(binomial(2, 2) == 1);
  assert(binomial(2, 1) == 2);
  assert(binomial(2, 0) == 1);
  assert(minus_one_power(2) == 1);
  assert(minus_one_power(0) == 1);
  assert(minus_one_power(1) == -1);
  assert(minus_one_power(-1) == -1);
  assert(minus_one_power(-2) == 1);
  assert(minus_one_power(-5) == -1);

  assert(std::abs(L_p(2.0, 0.05, 0) - 0.998335) < 1e-5);
  assert(std::abs(L_p(2.0, 0.05, 1) - 0.333) < 1e-5);
  assert(std::abs(L_p(2.0, 0.05, 2) - 0.199857) < 1e-5);
  assert(std::abs(L_c_m(1.5, 0.05, 0, 1) -
		  0.25 * (2.0*L_p(1.5, 0.05, 0)+L_p(1.5, 0.05, -2)
			  -2.0*L_p(1.5, 0.05, 1)-2.0*L_p(1.5, 0.05, -1)
			  +L_p(1.5, 0.05, 2))) < tol);
  assert(std::abs(L_c_m(1.5, 0.05, 0, 0) - L_p(1.5, 0.05, 0)) < tol);
}

void hubbard_tests() {
  assert((HubbardOperators::c_up_t() * HubbardOperators::c_up()
	  -HubbardOperators::n_up()).norm() < tol);
  assert((HubbardOperators::c_down_t() * HubbardOperators::c_down()
	  -HubbardOperators::n_down()).norm() < tol);
  assert((HubbardOperators::c_up_t()
	  * HubbardOperators::c_up_t()).norm() < tol);
  assert((HubbardOperators::c_down_t()
	  * HubbardOperators::c_down_t()).norm() < tol);
  assert((HubbardOperators::c_up()
	  * HubbardOperators::c_up()).norm() < tol);
  assert((HubbardOperators::c_down()
	  * HubbardOperators::c_down()).norm()< tol);
}

void run_tests() {
  std::cout << "Running tests" << std::endl;
  superoperator_test(10);
  hubbard_tests();
  function_tests();
  std::cout << "Done" << std::endl;
}
