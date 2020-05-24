#ifndef SPARSEBASEADDONS_HPP
#define SPARSEBASEADDONS_HPP

Scalar trace() const {
  Scalar sum(0);
  for (int k = 0; k < outerSize(); ++k)
    sum += coeff(k,k);
  return sum;
}

#endif /* SPARSEBASEADDONS_HPP */
