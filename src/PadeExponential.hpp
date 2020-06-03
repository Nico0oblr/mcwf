#ifndef PADEEXPONENTIAL_HPP
#define PADEEXPONENTIAL_HPP

#include "Common.hpp"

Eigen::SparseMatrix<double> spmat_abs(const spmat_t & A);

template<typename MatrixType>
struct PadePair {
  MatrixType U, V;
};

template<typename MatrixType>
struct ExpmPadeHelper {
  using PT = PadePair<MatrixType>;

  MatrixType m_A, m_A2, m_A4, m_A6, m_A8, m_A10;
  double m_d4, m_d6, m_d8, m_d10;
  ExpmPadeHelper(const MatrixType & mat)
    :m_A(mat), m_A2(), m_A4(), m_A6(), m_A8(), m_A10(),
     m_d4(-1.0), m_d6(-1.0), m_d8(-1.0), m_d10(-1.0) {
    m_A.makeCompressed();
  }

  spmat_t id() const {
    return spmat_t::Identity(A().rows(), A().cols());
  }
  
  const spmat_t & A() const {return m_A;}

  const spmat_t & A2() {
    if (m_A2.size() == 0) {
      m_A2 = A() * A();
    }
    return m_A2;
  }

  const spmat_t & A4() {
    if (m_A4.size() == 0) {
      m_A4 = A2() * A2();
    }
    return m_A4;
  }

  const spmat_t & A6() {
    if (m_A6.size() == 0) {
      m_A6 = A4() * A2();
    }
    return m_A6;
  }

  const spmat_t & A8() {
    if (m_A8.size() == 0) {
      m_A8 = A4() * A4();
    }
    return m_A8;
  }

  const spmat_t & A10() {
    if (m_A10.size() == 0) m_A10 = A8() * A2();
    return m_A10;
  }

  double d4() {
    if (m_d4 < 0.0) m_d4 = std::pow(A4().oneNorm(), 1.0 / 4.0);
    return m_d4;
  }

  double d6() {
    if (m_d6 < 0.0) m_d6 = std::pow(A6().oneNorm(), 1.0 / 6.0);
    return m_d6;
  }

  double d8() {
    if (m_d8 < 0.0) m_d8 = std::pow(A8().oneNorm(), 1.0 / 8.0);
    return m_d8;
  }

  double d10() {
    if (m_d10 < 0.0) m_d10 = std::pow(A10().oneNorm(), 1.0 / 10.0);
    return m_d10;
  }

  PT pade3() {
    std::vector<double> b{120., 60., 12., 1.};
    return {A() * (b[3] * A2() + b[1] * id()), b[2] * A2() + b[0] * id()};
  }

  PT pade5() {
    std::vector<double> b{30240., 15120., 3360., 420., 30., 1.};
    return {A() * (b[5] * A4() + b[3] * A2() + b[1] * id()),
	    b[4] * A4() + b[2] * A2() + b[0] * id()};
  }
  
  PT pade7() {
    std::vector<double> b{17297280., 8648640., 1995840., 277200.,
			  25200., 1512., 56., 1.};
    return {A() * (b[7] * A6() + b[5] * A4() + b[3] * A2() + b[1] * id()),
	    b[6] * A6() + b[4] * A4() + b[2] * A2() + b[0] * id()};
  }

  PT pade9() {
    std::vector<double> b{17643225600., 8821612800., 2075673600.,
			  302702400., 30270240., 2162160., 110880.,
			  3960., 90., 1.};
    return {A() * (b[9] * A8() + b[7] * A6()
		   + b[5] * A4() + b[3] * A2() + b[1] * id()),
	    b[8] * A8() + b[6] * A6() + b[4] * A4()
	    + b[2] * A2() + b[0] * id()};
  }

  PT pade13_scaled(double s) {
    std::vector<double> b{64764752532480000., 32382376266240000.,
			  7771770303897600.,
			  1187353796428800., 129060195264000.,
			  10559470521600., 670442572800., 33522128640.,
			  1323241920., 40840800., 960960.,
			  16380., 182., 1.};
    MatrixType B = A() * std::pow(2.0, -s);
    MatrixType B2 = A2() * std::pow(2.0, (-2.0 * s));
    MatrixType B4 = A4() * std::pow(2.0, (-4.0 * s));
    MatrixType B6 = A6() * std::pow(2.0, (-6.0 * s));
    MatrixType U2 = B6 * (b[13] * B6 + b[11] * B4 + b[9] * B2);
    MatrixType U = B * (U2 + b[7] * B6 + b[5] * B4 + b[3] * B2 + b[1] * id());
    MatrixType V2 = B6 * (b[12] * B6 + b[10] * B4 + b[8] * B2);
    MatrixType V = V2 + b[6] * B6 + b[4] * B4 + b[2] * B2 + b[0] * id();
    return {U, V};
  }
  
};

double onenorm_power(const spmat_t & A, size_type power);

int _ell(const spmat_t & A, int m);

spmat_t solve_P_Q(const PadePair<spmat_t> & p);

spmat_t expm(const spmat_t & A);

#endif /* PADEEXPONENTIAL_HPP */
