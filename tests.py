#!/usr/bin/env python3

import numpy as np
from mcwf import *
import pytest
import scipy
from scipy.sparse.linalg import norm
from scipy.sparse import rand
from scipy.sparse.linalg import expm as sc_expm
from scipy.sparse.linalg import norm

def mat_almost_equal(A, B, tol = 1e-12):
    if scipy.sparse.issparse(A):
        A = A.todense()
    if scipy.sparse.issparse(B):
        B = B.todense()
    __tracebackhide__ = True
    distance = np.linalg.norm(A - B)
    if distance != pytest.approx(0.0, abs = tol):
        pytest.fail("Matrices not equal: {}".format(distance))

def vec_almost_equal(A, B, tol = 1e-12):
    A = A.reshape((np.prod(A.shape), 1))
    B = B.reshape((np.prod(B.shape), 1))
    assert A.shape == B.shape
    mat_almost_equal(A, B, tol)
        
def spmat_almost_equal(A, B, tol = 1e-12):
    mat_almost_equal(A.todense(), B.todense(), tol)
        
def test_hubbard_projector():
    mat_almost_equal(HubbardProjector(2, 1, 1), HubbardProjector_sp(2, 1, 1))
    mat_almost_equal(HubbardProjector(3, 2, 1), HubbardProjector_sp(3, 2, 1))
    mat_almost_equal(HubbardProjector(2, 3, 1), HubbardProjector_sp(2, 3, 1))
    mat_almost_equal(HubbardProjector(4, 2, 2), HubbardProjector_sp(4, 2, 2))

def test_matrix_exponential():
    for i in range(5):
        mat = np.random.rand(20, 20)
        mat_almost_equal(expm(mat), matrix_exponential(mat), 1e-8)
        mat_almost_equal(expm(mat), sc_expm(mat), 1e-8)


def test_neel_state():
    for sites in [2, 4]:
        projector = HubbardProjector(sites, sites // 2, sites // 2)
        state = HubbardNeelState_sp(sites, projector).draw()
        spin_up = projector @ sum_operator_sp(HubbardOperators.n_up(), sites) @ projector.conj().T
        spin_down = projector @ sum_operator_sp(HubbardOperators.n_down(), sites) @ projector.conj().T
        assert np.abs(state.dot(spin_up @ state)) == pytest.approx(sites // 2)
        assert np.abs(state.dot(spin_down @ state)) == pytest.approx(sites // 2)

def test_norm_est():
    samples = 20
    dimension = 100
    powers = 4
    for i in range(samples):
        for power in range(0, powers):
            mat = np.random.rand(dimension, dimension)
            + 1.0j * np.random.rand(dimension, dimension)
            spmat = scipy.sparse.csr_matrix(mat, dtype = np.complex128)
            assert onenormest_matrix_power(spmat, power) == pytest.approx(np.linalg.norm(np.linalg.matrix_power(mat, power) , ord = 1))
        

def test_auxiliary():
    n = 10
    lst = [0, 1, 2, 3, 4]
    assert len(invert_indexer(np.array(lst), n)) == (n - len(lst))
    lst2 = [0, 5, 4, 3, 2, 1]
    assert len(in1d(np.array(lst), np.array(lst2))) == 5
    assert len(in1d(np.array(lst2), np.array(lst))) == 5

def test_p_max_default():
    m_max = 55
    expected_p_max = 8
    observed_p_max = compute_p_max(m_max)
    assert observed_p_max == expected_p_max

def test_onenorm():
    for i in range(4):
        mat = np.random.rand(50, 50) + 1.0j * np.random.rand(50, 50)
        assert exact_onenorm(mat) == pytest.approx(np.linalg.norm(mat, ord = 1))

def test_infnorm():
    for i in range(4):
        mat = np.random.rand(50, 50) + 1.0j * np.random.rand(50, 50)
        assert exact_infnorm(mat) == pytest.approx(np.linalg.norm(mat, ord = np.inf))
    b = np.array([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
    b = b.reshape((3, 3))
    assert np.linalg.norm(b, np.inf) == 9
    assert exact_infnorm(b) == 9

def test_p_max_range():
    for m_max in range(1, 55+1):
        p_max = compute_p_max(m_max)
        assert p_max * (p_max - 1) <= m_max + 1
        p_too_big = p_max + 1
        assert p_too_big * (p_too_big - 1) > m_max + 1

def _test_expm_aux():
    import importlib.util
    spec = importlib.util.spec_from_file_location("module.name", "/usr/lib/python3.8/site-packages/scipy/sparse/linalg/_expm_multiply.py")
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)

    dimension = 100
    samples = 10
    for i in range(samples):
        mat = np.random.rand(dimension, dimension) + 1.0j * np.random.rand(dimension, dimension)
        vec = np.random.rand(dimension, 1) + 1.0j * np.random.rand(dimension, 1)
        mat_onenorm = np.linalg.norm(mat, ord = 1)
        tol = 2 ** -53

        # Testing condition 3_13
        for m_max in range(1, 10):
            assert foo._condition_3_13(mat_onenorm, 1, m_max, 2) == condition_3_13(mat_onenorm, 1, m_max, 2)

        # Testing norm info
        expected_norminfo = foo.LazyOperatorNormInfo(mat)
        infact_norminfo = LazyOperatorNormInfo(mat, -1.0)
        assert expected_norminfo.onenorm() == infact_norminfo.onenorm()
        for p in range(1, 7):
            assert expected_norminfo.d(p) == pytest.approx(infact_norminfo.d(p))
            assert expected_norminfo.alpha(p) == pytest.approx(infact_norminfo.alpha(p))
            # Testing cost computation
            for m in range(1, 30):
                assert compute_cost_div_m(m, p, infact_norminfo) == pytest.approx(foo._compute_cost_div_m(m, p, expected_norminfo))

        assert fragment_3_1(infact_norminfo, 1, tol, 55, 2) == foo._fragment_3_1(expected_norminfo, 1, tol, 55, 2)
        

def test_exp_apply():
    dimension = 20
    mat = rand(dimension, dimension, density = 0.1) + 1.0j * rand(dimension, dimension, density = 0.1)
    vec = rand(dimension, 1, density = 0.1) + 1.0j * rand(dimension, 1, density = 0.1)
    vec_almost_equal(expm_multiply_simple(mat, vec.todense()),
                     scipy.sparse.linalg.expm_multiply(mat, vec).todense())


def arnoldi_iteration(A, b, n: int):
    """Computes a basis of the (n + 1)-Krylov subspace of A: the space
    spanned by {b, Ab, ..., A^n b}.

    Arguments
      A: m × m array
      b: initial vector (length m)
      n: dimension of Krylov subspace, must be >= 1
    
    Returns
      Q: m x (n + 1) array, the columns are an orthonormal basis of the
        Krylov subspace.
      h: (n + 1) x n array, A on basis Q. It is upper Hessenberg.  
    """
    m = A.shape[0]
    h = np.zeros((n + 1, n), dtype = np.complex128)
    Q = np.zeros((m, n + 1), dtype = np.complex128)
    q = (b / np.linalg.norm(b)).ravel()  # Normalize the input vector
    Q[:, 0] = q  # Use it as the first Krylov vector

    for k in range(n):
        v = A.dot(q)  # Generate a new candidate vector
        for j in range(k + 1):  # Subtract the projections on previous vectors
            h[j, k] = np.dot(Q[:, j].conj(), v)
            v = v - h[j, k] * Q[:, j]

        h[k + 1, k] = np.linalg.norm(v)
        eps = 1e-12  # If v is shorter than this threshold it is the zero vector
        if h[k + 1, k] > eps:  # Add the produced vector to the list, unless
            q = v / h[k + 1, k]  # the zero vector is produced.
            Q[:, k + 1] = q
        else:  # If that happens, stop iterating.
            return Q, h
    return Q[:,:n], h[:n,:]
    
def test_arnoldi():
    samples = 5
    dimension = 1000
    density = 0.01
    niter = 20
    for i in range(samples):
        mat = rand(dimension, dimension, density = density) + 1.0j * rand(dimension, dimension, density = density)
        vec = rand(dimension, 1, density = 0.1) + 1.0j * rand(dimension, 1, density = 0.1)
        mat = 0.01 * mat
        vec /= np.linalg.norm(vec.todense())
        iteration = ArnoldiIteration(mat, niter, niter, vec.todense())
        Q, h = arnoldi_iteration(mat.toarray(),
                                 vec.toarray(), niter)
        mat_almost_equal(Q, iteration.V())
        mat_almost_equal(h, iteration.H())
        mat_almost_equal(iteration.V().T.conj()
                         @ mat.todense()
                         @ iteration.V(), iteration.H())
        x = iteration.V().T.conj() @ iteration.V()
        x[x < 1e-10] = 0
        assert np.count_nonzero(x - np.diag(np.diagonal(x))) == 0
        res1 = iteration.apply_exp(vec.todense(), niter)
        res2 = expm_multiply_simple(mat, vec.todense())
        #res1 = res1.reshape((res1.size, 1))
        #res2 = res1.reshape((res2.size, 1))
        vec_almost_equal(res1, res2)

def test_Kronecker():
    dimension1 = 20
    dimension2 = 30
    density = 0.1
    A = rand(dimension1, dimension1, density = density)
    B = rand(dimension2, dimension2, density = density)
    kron = tensor_identity(A, dimension2) @ tensor_identity_LHS(B, dimension1)
    vec = rand(dimension1 * dimension2, 1, density = 1.0)
    vec_almost_equal(tensor_identity_LHS(B, dimension1) @ vec,
                     kroneckerApply_LHS(B, dimension1, vec.todense()))
    vec_almost_equal(tensor_identity(A, dimension2) @ vec,
                     kroneckerApply_id(A, dimension2, vec.todense()))
    vec_almost_equal(kron @ vec, kroneckerApply(A, B, vec.todense()))

def test_operators_eval():
    dimension = 20
    density = 0.5
    mat1 = rand(dimension, dimension, density = density)\
        + 1.0j * rand(dimension, dimension, density = density)
    mat2 = rand(dimension, dimension, density = density)\
        + 1.0j * rand(dimension, dimension, density = density)
    mat1 /= np.linalg.norm(mat1.todense())
    mat2 /= np.linalg.norm(mat2.todense())
    op1 = operatorize(mat1)
    op2 = operatorize(mat2)
    vec = rand(dimension, 1, density = 1.0)\
        + 1.0j * rand(dimension, 1, density = 1.0)
    vec /= np.linalg.norm(vec.todense())

    assert op1.eval().shape == mat1.shape
    assert (0.5 * op1).eval().shape == mat1.shape
    spmat_almost_equal((0.5 * op1).eval(), 0.5 * mat1)
    spmat_almost_equal((op1 + op2).eval(), mat1 + mat2)
    vec_almost_equal((op1 + op2) * vec.todense(), (mat1 + mat2) @ vec)
    vec_almost_equal((0.5 * op1) * vec.todense(), (0.5 * mat1) @ vec)
    vec_almost_equal(vec.todense().T * (op1 + op2), vec.T @ (mat1 + mat2))
    vec_almost_equal(vec.todense().T * (0.5 * op1), vec.T @ (0.5 * mat1))
    mat_almost_equal((op1 + op2) * mat1.todense(), (mat1 + mat2) @ mat1)
    mat_almost_equal((0.5 * op1) * mat1.todense(), (0.5 * mat1) @ mat1)
    mat_almost_equal(mat1.todense() * (op1 + op2), mat1 @ (mat1 + mat2))
    mat_almost_equal(mat1.todense() * (0.5 * op1), mat1 @ (0.5 * mat1))
    
    spmat_almost_equal((op1 - op2).eval(), mat1 - mat2)
    vec_almost_equal((op1 - op2) * vec.todense(), (mat1 - mat2) @ vec)
    mat_almost_equal((op1 - op2) * mat1.todense(), (mat1 - mat2) @ mat1)
    # Testing multiplication
    mat_almost_equal((op1 * op2).eval(), mat1 @ mat2)
    mat_almost_equal((op1.adjoint() * op2).eval(), mat1.T.conj() @ mat2)
    mat_almost_equal((op1 * op2.adjoint()).eval(), mat1 @ mat2.T.conj())
    vec_almost_equal((op1 * op2) * vec.todense(),
                     (op1 * op2).eval() @ vec.todense())
    vec_almost_equal((op1 * op2) * vec.todense(), (mat1 @ mat2) @ vec)
    vec_almost_equal((op1.adjoint() * op2) * vec.todense(),
                     (mat1.T.conj() @ mat2) @ vec)

    vec_almost_equal(scale_and_add(op1, op2, 3.0, 2.0j).eval(),
                     (mat1 * 3.0 + mat2 * 2.0j))

    dimension2 = 30
    
    vect = rand(dimension * dimension2, 1, density = 1.0)\
        + 1.0j * rand(dimension * dimension2, 1, density = 1.0)
    mat3 = rand(dimension2, dimension, density = density)\
        + 1.0j * rand(dimension2, dimension, density = density)

def test_operators2():
    dimension = 20
    density = 0.5
    mat1 = rand(dimension, dimension, density = density)\
        + 1.0j * rand(dimension, dimension, density = density)
    mat2 = rand(dimension, dimension, density = density)\
        + 1.0j * rand(dimension, dimension, density = density)
    mat1 /= np.linalg.norm(mat1.todense())
    mat2 /= np.linalg.norm(mat2.todense())
    op1 = operatorize(mat1)
    op2 = operatorize(mat2)
    vec = rand(dimension, 1, density = 1.0)\
        + 1.0j * rand(dimension, 1, density = 1.0)
    vec /= np.linalg.norm(vec.todense())
    op1 = operatorize(mat1)
    op2 = operatorize(mat2)
    vecdoub = rand(2 * dimension, 1, density = 1.0)\
        + 1.0j * rand(2 * dimension, 1, density = 1.0)
    vecdoub /= np.linalg.norm(vecdoub.todense())

    mat_almost_equal(doubleOperator(op1).eval(),
                     np.kron(np.eye(2), mat1.todense()))
    vec_almost_equal(doubleOperator(op1) * vecdoub.todense(),
                     np.kron(np.eye(2), mat1.todense()) @ vecdoub)

def test_operator_memory():
    dimension = 20
    density = 0.5
    mat1 = rand(dimension, dimension, density = density)\
        + 1.0j * rand(dimension, dimension, density = density)
    op1 = operatorize(mat1)
    tmp = op1.clone(); tmp.eval()
    tmp = (op1 + op1).clone(); tmp.eval()
    tmp = (op1 * op1).clone(); tmp.eval()
    tmp = (0.5 * op1).clone(); tmp.eval()

def test_math():
  assert factorial(4) == 4 * 3 * 2
  assert factorial(5) == 5 * 4 * 3 * 2
  assert binomial(5, 2) == 10
  assert binomial(2, 2) == 1
  assert binomial(2, 1) == 2
  assert binomial(2, 0) == 1
  assert minus_one_power(2) == 1
  assert minus_one_power(0) == 1
  assert minus_one_power(1) == -1
  assert minus_one_power(-1) == -1
  assert minus_one_power(-2) == 1
  assert minus_one_power(-5) == -1

def test_hubbard_operators():
    assert np.linalg.norm(HubbardOperators.c_up_t() @ HubbardOperators.c_up_t()) == pytest.approx(0.0)
    assert np.linalg.norm(HubbardOperators.c_down_t() @ HubbardOperators.c_down_t()) == pytest.approx(0.0)
    assert np.linalg.norm(HubbardOperators.c_up() @ HubbardOperators.c_up()) == pytest.approx(0.0)
    assert np.linalg.norm(HubbardOperators.c_down() @ HubbardOperators.c_down()) == pytest.approx(0.0)

    mat_almost_equal(HubbardOperators.c_up_t() @ HubbardOperators.c_up(),
	                HubbardOperators.n_up())
    mat_almost_equal(HubbardOperators.c_down_t() @ HubbardOperators.c_down(),
                        HubbardOperators.n_down())

def test_light_matter_aux():
    assert L_p(2.0, 0.05, 0) == pytest.approx(0.998335, abs = 1e-5)
    assert L_p(2.0, 0.05, 1) == pytest.approx(0.333, abs = 1e-5)
    assert L_p(2.0, 0.05, 2) == pytest.approx(0.199857, abs = 1e-5)
    assert L_c_m(1.5, 0.05, 0, 1) == pytest.approx(0.25 * (2.0*L_p(1.5, 0.05, 0)+L_p(1.5, 0.05, -2) -2.0*L_p(1.5, 0.05, 1)-2.0*L_p(1.5, 0.05, -1) + L_p(1.5, 0.05, 2)))
    assert L_c_m(1.5, 0.05, 0, 0) == pytest.approx(L_p(1.5, 0.05, 0))

def test_superoperator():
    dimension = 10
    vec = np.random.rand(dimension, dimension)
    op = np.random.rand(dimension, dimension)
    test1 = superoperator_left(op, dimension)
    tmp1 = (superoperator_left(op, dimension) @ unstack_matrix(vec)).todense()
    result_lhs1 = restack_vector(tmp1, dimension)
    result_lhs2 = op @ vec
    tmp2 = superoperator_right(op, dimension) @ unstack_matrix(vec).todense()
    result_rhs1 = restack_vector(tmp2, dimension)
    result_rhs2 = vec @ op

    mat_almost_equal(restack_vector(unstack_matrix(vec).todense(), dimension), vec)
    mat_almost_equal(result_lhs1, result_lhs2)
    mat_almost_equal(result_rhs1, result_rhs2)

